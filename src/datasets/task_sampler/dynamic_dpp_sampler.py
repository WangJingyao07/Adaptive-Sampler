from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset
import torch
from torchmeta.utils.data.dataset import CombinationMetaDataset
import random
import warnings
import numpy as np
from dppy.finite_dpps import FiniteDPP
from torch.utils.data.sampler import RandomSampler
from src.datasets.task_sampler.disjoint_sampler import DisjointMetaDataloader


def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.
    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)
    return prototypes


class DPPSampler(RandomSampler):
    def __init__(self, data_source, batch_size, DPP=None):
        self.DPP = DPP
        self.rng = np.random.RandomState(1)
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
            super(DPPSampler, self).__init__(data_source,
                                             replacement=True)

    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        num_classes = len(self.data_source.dataset)
        if self.DPP is None:
            for _ in range(self.batch_size):
                yield tuple(random.sample(range(num_classes), num_classes_per_task))
        else:
            for _ in range(self.batch_size):
                try:
                    self.DPP.sample_exact_k_dpp(size=num_classes_per_task, random_state=self.rng)
                    yield tuple(self.DPP.list_of_samples[-1])
                except Exception:
                    tasks = []
                    for i in range(int(num_classes_per_task/5)):
                        self.DPP.sample_exact_k_dpp(size=5, random_state=self.rng)
                        tasks += self.DPP.list_of_samples[-1]
                    yield tuple(tasks)


class MetaDatasetRandomSampler(DPPSampler):
    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        if self.DPP is None:
            for _ in range(self.batch_size):
                source = random.randrange(len(self.data_source.dataset.sources))
                num_classes = len(self.data_source.dataset._class_datasets[source])
                offset = self.data_source.dataset._cum_num_classes[source]
                indices = random.sample(range(num_classes), num_classes_per_task)
                yield tuple(index + offset for index in indices)
        else:
            for _ in range(self.batch_size):
                try:
                    self.DPP.sample_exact_k_dpp(size=num_classes_per_task, random_state=self.rng)
                    yield tuple(self.DPP.list_of_samples[-1])
                except Exception:
                    tasks = []
                    for i in range(int(num_classes_per_task/5)):
                        self.DPP.sample_exact_k_dpp(size=5, random_state=self.rng)
                        tasks += self.DPP.list_of_samples[-1]
                    yield tuple(tasks)


class MetaDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, DPP=None):
        if collate_fn is None:
            collate_fn = no_collate

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            sampler = DPPSampler(dataset, batch_size, DPP)
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


class BatchMetaDataLoaderdDPP(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, DPP=None, use_batch_collate=True):
        if use_batch_collate:
            collate_fn = BatchMetaCollate(default_collate)
        else:
            collate_fn = default_collate
            sampler = MetaDatasetRandomSampler(dataset, batch_size)

        super(BatchMetaDataLoaderdDPP, self).__init__(dataset,
                                                      batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                                                      collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                                                      timeout=timeout, worker_init_fn=worker_init_fn, DPP=DPP)


class dDPP(object):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True, num_ways=5,
                 dpp_threshold=500, model_name='protonet', use_batch_collate=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_ways = num_ways
        self.index = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dpp_threshold = dpp_threshold
        self.model_name = model_name
        self.use_batch_collate = use_batch_collate
        self.model = None
        self.DPP = None
        self.loader = BatchMetaDataLoaderdDPP(self.dataset,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              num_workers=self.num_workers,
                                              pin_memory=self.pin_memory, DPP=self.DPP, use_batch_collate=self.use_batch_collate)

    def init_metalearner(self, metalearner):
        self.metalearner = metalearner

    def get_task_embedding(self):
        task_embedding = {}
        for batch in DisjointMetaDataloader(self.dataset,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=8,
                                            pin_memory=True,
                                            use_batch_collate=self.use_batch_collate):
            with torch.no_grad():
                if self.model_name == 'cnaps':
                    for task_id, (train_inputs, train_targets, task) \
                            in enumerate(zip(*batch['train'])):
                        _, prototypes = self.metalearner.model(
                            train_inputs.to(device=self.device),
                            train_targets.to(device=self.device),
                            train_inputs.to(device=self.device))
                        for class_id, index in enumerate(task):
                            task_embedding[str(index.item())] = np.array(
                                prototypes[class_id].cpu().tolist())
                elif self.model_name == 'matching_networks':
                    train_inputs, train_targets, tasks = batch['train']
                    _, _, train_embeddings = self.metalearner.model(
                        train_inputs.to(device=self.device), train_targets.to(device=self.device),
                        train_inputs.to(device=self.device), train_targets.to(device=self.device), self.dataset.num_classes_per_task)
                    prototypes = get_prototypes(train_embeddings.to(
                        device='cpu'), train_targets, self.num_ways)
                    for task_id, task in enumerate(tasks):
                        for class_id, index in enumerate(task):
                            task_embedding[str(index.item())] = np.array(
                                prototypes[task_id][class_id].cpu().tolist())
                elif self.model_name == 'metaoptnet':
                    for task_id, (train_inputs, train_targets, task) \
                            in enumerate(zip(*batch['train'])):
                        train_inputs = train_inputs.to(device=self.device)
                        train_targets = train_targets.to(device=self.device)
                        prototypes = self.metalearner.model(
                            train_inputs.to(device=self.device), train_targets.to(
                                device=self.device),
                            train_inputs.to(device=self.device), train_targets.to(device=self.device), dpp=True)
                        for class_id, index in enumerate(task):
                            task_embedding[str(index.item())] = np.array(
                                prototypes.squeeze(0)[class_id].cpu().tolist())
                elif self.model_name in ['maml', 'reptile']:
                    for task_id, (train_inputs, train_targets, task) \
                            in enumerate(zip(*batch['train'])):
                        _, prototypes = self.metalearner.model(
                            train_inputs.to(device=self.device))
                        for class_id, index in enumerate(task):
                            task_embedding[str(index.item())] = np.array(
                                prototypes[class_id].cpu().tolist())
                else:
                    train_inputs, train_targets, tasks = batch['train']
                    _, train_embeddings = self.metalearner.model(
                        train_inputs.to(device=self.device))
                    prototypes = get_prototypes(train_embeddings.to(
                        device='cpu'), train_targets, self.num_ways)
                    for task_id, task in enumerate(tasks):
                        for class_id, index in enumerate(task):
                            task_embedding[str(index.item())] = np.array(
                                prototypes[task_id][class_id].cpu().tolist())
        return np.array(list(task_embedding.values()))

    def get_diverse_tasks(self):
        Phi = self.get_task_embedding()
        return FiniteDPP('likelihood', **{'L': Phi.dot(Phi.T)})

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index % 500 == 0:
            self.DPP = self.get_diverse_tasks()
            self.loader = BatchMetaDataLoaderdDPP(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.shuffle,
                                                  num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory, DPP=self.DPP, use_batch_collate=self.use_batch_collate)
        for batch in self.loader:
            return batch
            break
