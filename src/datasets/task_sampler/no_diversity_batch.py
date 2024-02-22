from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset

from torchmeta.utils.data.dataset import CombinationMetaDataset
import random
import warnings
from itertools import combinations
from torch.utils.data.sampler import RandomSampler
from itertools import chain


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source, batch_size):
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
            super(CombinationRandomSampler, self).__init__(data_source,
                                                           replacement=True)

    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        num_classes = len(self.data_source.dataset)
        x = []
        for i in range(self.batch_size):
            x.append(random.sample(range(num_classes), num_classes_per_task))
        for _ in combinations(range(num_classes), num_classes_per_task):
            random.shuffle(x)
            for j in range(self.batch_size):
                y = tuple(random.sample(x[j], num_classes_per_task))
                yield y


class MetaDatasetRandomSampler(CombinationRandomSampler):
    def __iter__(self):
        num_classes_per_source = list(map(len, self.data_source.dataset._class_datasets))
        num_classes_per_task = self.data_source.num_classes_per_task
        iterator = chain(*[combinations(range(num_classes), num_classes_per_task)
                           for num_classes in num_classes_per_source])
        x = []
        for _ in range(self.batch_size):
            source = random.randrange(len(self.data_source.dataset.sources))
            num_classes = len(self.data_source.dataset._class_datasets[source])
            offset = self.data_source.dataset._cum_num_classes[source]
            indices = random.sample(range(num_classes), num_classes_per_task)
            x.append(list(index + offset for index in indices))
        for _ in iterator:
            random.shuffle(x)
            for j in range(self.batch_size):
                y = tuple(random.sample(x[j], num_classes_per_task))
                yield y


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


class MetaDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        if collate_fn is None:
            collate_fn = no_collate

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            sampler = CombinationRandomSampler(dataset, batch_size)
        shuffle = False

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
                                             shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                             num_workers=num_workers, collate_fn=collate_fn,
                                             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class BatchMetaDataLoaderNDB(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None, use_batch_collate=True):
        if use_batch_collate:
            collate_fn = BatchMetaCollate(default_collate)
        else:
            collate_fn = default_collate
            sampler = MetaDatasetRandomSampler(dataset, batch_size)

        super(BatchMetaDataLoaderNDB, self).__init__(dataset,
                                                     batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                                     batch_sampler=batch_sampler, num_workers=num_workers,
                                                     collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                                                     timeout=timeout, worker_init_fn=worker_init_fn)
