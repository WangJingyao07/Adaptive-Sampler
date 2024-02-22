from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset
from glob import glob
import torch
from torchmeta.utils.data.dataset import CombinationMetaDataset
import warnings
from itertools import combinations
from torch.utils.data.sampler import RandomSampler
import numpy as np
from src.datasets.task_sampler.disjoint_sampler import DisjointMetaDataloader
from dppy.finite_dpps import FiniteDPP


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


class CombinationRandomSamplerStaticDDP(RandomSampler):
    def __init__(self, data_source, dataset_name):
        self.model = self.init_static_model(data_source, dataset_name)
        Phi = self.get_task_embedding(data_source, dataset_name)
        self.rng = np.random.RandomState(1)
        self.DPP = FiniteDPP('likelihood', **{'L': Phi.dot(Phi.T)})
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
            super(CombinationRandomSamplerStaticDDP, self).__init__(data_source,
                                                                    replacement=True)

    def init_static_model(self, data_source, dataset_name):
        if dataset_name in ["sinusoid"]:
            from src.maml.model import ModelMLPSinusoid
            model = ModelMLPSinusoid()
            ways_path = "" if data_source.num_classes_per_task == 5 else "_20"
            for model_path in enumerate(glob(f"maml_{dataset_name}{ways_path}/0/*/config.json")):
                with open(model_path, 'rb') as f:
                    model.load_state_dict(torch.load(f, map_location=torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')))
        else:
            from src.protonet.model import Protonet_Omniglot, Protonet_MiniImagenet
            model = Protonet_Omniglot() if dataset_name == 'omniglot' else Protonet_MiniImagenet()
            ways_path = "" if data_source.num_classes_per_task == 5 else "_20"
            for model_path in enumerate(glob(f"protonet_{dataset_name}{ways_path}/0/*/config.json")):
                with open(model_path, 'rb') as f:
                    model.load_state_dict(torch.load(f, map_location=torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')))
        return model

    def get_task_embedding(self, data_source, dataset_name):
        task_embedding = {}
        for batch in DisjointMetaDataloader(data_source,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            use_batch_collate=dataset_name != 'meta_dataset'):
            train_inputs, train_targets, tasks = batch['train']
            with torch.no_grad():
                is_classification_task = (not train_targets.dtype.is_floating_point)
                if is_classification_task:
                    train_embeddings, _ = self.model(train_inputs)
                    prototypes = get_prototypes(train_embeddings, train_targets,
                                                data_source.num_classes_per_task)
                else:
                    _, prototypes = self.model(train_inputs)
            for task_id, task in enumerate(tasks):
                for class_id, index in enumerate(task):
                    task_embedding[str(index.item())] = np.array(
                        prototypes[task_id][class_id].cpu().tolist())
        return np.array(list(task_embedding.values()))

    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        num_classes = len(self.data_source.dataset)
        for _ in combinations(range(num_classes), num_classes_per_task):
            self.DPP.sample_exact_k_dpp(size=num_classes_per_task, random_state=self.rng)
            yield tuple(self.DPP.list_of_samples[-1])


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
                 worker_init_fn=None, dataset_name="omniglot"):
        if collate_fn is None:
            collate_fn = no_collate

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            sampler = CombinationRandomSamplerStaticDDP(dataset, dataset_name)
        shuffle = False

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
                                             shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                             num_workers=num_workers, collate_fn=collate_fn,
                                             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class sDPP(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None,
                 dataset_name="omniglot", use_batch_collate=True):
        if use_batch_collate:
            collate_fn = BatchMetaCollate(default_collate)
        else:
            collate_fn = default_collate

        super(sDPP, self).__init__(dataset,
                                   batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                   batch_sampler=batch_sampler, num_workers=num_workers,
                                   collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                                   timeout=timeout, worker_init_fn=worker_init_fn, dataset_name=dataset_name)
