from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset

from torchmeta.utils.data.dataset import CombinationMetaDataset
import random
import warnings
from torch.utils.data.sampler import RandomSampler
import ast


class OHTMSampler(RandomSampler):
    def __init__(self, data_source, tasks, batch_size):
        self.tasks = tasks
        self.batch_size = batch_size
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        # Temporarily disable the warning if the length of the length of the
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(OHTMSampler, self).__init__(data_source,
                                              replacement=True)

    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        num_classes = len(self.data_source.dataset)
        if len(self.tasks):
            x = self.tasks
            for _ in range(len(self.tasks)):
                y = random.sample(x, 1)
                x = [item for item in x if item not in y]
                yield tuple(y[0])
            for _ in range(int(self.batch_size-len(self.tasks))):
                yield tuple(random.sample(range(num_classes), num_classes_per_task))
        else:
            for _ in range(self.batch_size):
                yield tuple(random.sample(range(num_classes), num_classes_per_task))


class MetaDatasetRandomSampler(OHTMSampler):
    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        if len(self.tasks):
            x = self.tasks
            for _ in range(len(self.tasks)):
                y = random.sample(x, 1)
                x = [item for item in x if item not in y]
                yield tuple(y[0])
            for _ in range(int(self.batch_size-len(self.tasks))):
                source = random.randrange(len(self.data_source.dataset.sources))
                num_classes = len(self.data_source.dataset._class_datasets[source])
                offset = self.data_source.dataset._cum_num_classes[source]
                indices = random.sample(range(num_classes), num_classes_per_task)
                yield tuple(index + offset for index in indices)
        else:

            for _ in range(self.batch_size):
                source = random.randrange(len(self.data_source.dataset.sources))
                num_classes = len(self.data_source.dataset._class_datasets[source])
                offset = self.data_source.dataset._cum_num_classes[source]
                indices = random.sample(range(num_classes), num_classes_per_task)
                yield tuple(index + offset for index in indices)


class MetaDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, task=None):
        if collate_fn is None:
            collate_fn = no_collate

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            sampler = OHTMSampler(dataset, task, batch_size)
        shuffle = False

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
                                             shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                             num_workers=num_workers, collate_fn=collate_fn,
                                             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class BatchMetaCollate(object):

    def __init__(self, collate_fn):
        super().__init__()
        self.collate_fn = collate_fn

    def collate_task(self, task):
        if isinstance(task, TorchDataset):
            return self.collate_fn([task[idx] for idx in range(len(task))])
        elif isinstance(task, OrderedDict):
            return OrderedDict([(key, self.collate_task(subtask))
                                for (key, subtask) in task.items()])
        else:
            raise NotImplementedError()

    def __call__(self, batch):
        return self.collate_fn([self.collate_task(task) for task in batch])


def no_collate(batch):
    return batch


class BatchMetaDataLoaderOHTM(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, task=None, use_batch_collate=True):
        if use_batch_collate:
            collate_fn = BatchMetaCollate(default_collate)
        else:
            collate_fn = default_collate
            sampler = MetaDatasetRandomSampler(dataset, task, batch_size)

        super(BatchMetaDataLoaderOHTM, self).__init__(dataset,
                                                      batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                                                      collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                                                      timeout=timeout, worker_init_fn=worker_init_fn, task=task)


class OHTM(object):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=1,
                 pin_memory=True, hard_mining_threshold=32, buffer_threshold=50,
                 use_batch_collate=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.hard_mining_threshold = hard_mining_threshold
        self.buffer_threshold = buffer_threshold
        self.index = 0
        self.use_batch_collate = use_batch_collate

    def init_metalearner(self, metalearner):
        self.metalearner = metalearner

    def get_hardest_scores(self):
        if self.index >= self.hard_mining_threshold:
            hardest_tasks = [ast.literal_eval(task) for task in list(self.metalearner.hardest_task.keys())[
                :int(self.batch_size/2)]]
        else:
            hardest_tasks = []
        return hardest_tasks

    def prune_task_buffer(self):
        if len(self.metalearner.hardest_task.items()) > self.buffer_threshold:
            for item in list(self.metalearner.hardest_task.keys())[self.buffer_threshold:]:
                self.metalearner.hardest_task.pop(item)

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        self.metalearner.hardest_task = OrderedDict({k: v for k,
                                                     v in sorted(self.metalearner.hardest_task.items(), key=lambda item: item[1])})
        self.prune_task_buffer()
        task_for_batch = self.get_hardest_scores()
        for batch in BatchMetaDataLoaderOHTM(self.dataset,
                                             batch_size=self.batch_size,
                                             shuffle=self.shuffle,
                                             num_workers=self.num_workers,
                                             pin_memory=self.pin_memory, task=task_for_batch, use_batch_collate=self.use_batch_collate):
            return batch
            break
