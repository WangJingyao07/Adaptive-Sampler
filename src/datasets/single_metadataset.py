import os
import tensorflow.compat.v1 as tf

from torchmeta.utils.data import CombinationMetaDataset, ClassDataset

from src.datasets.meta_dataset.reader import Reader
from src.datasets.meta_dataset.dataset_spec import load_dataset_spec
from src.datasets.meta_dataset.learning_spec import Split
from src.datasets.meta_dataset.decoder import ImageDecoder
import torch
from collections import OrderedDict
from itertools import islice


tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(112)

SOURCES = {
    'train': ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
              "vgg_flower", "traffic_sign", "mscoco"],
    'val': ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
            "vgg_flower", "traffic_sign", "mscoco"],
    'test': ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
             "vgg_flower", "traffic_sign", "mscoco"]
}


class SingleMetaDataset(CombinationMetaDataset):
    def __init__(
        self,
        root,
        source,
        num_ways,
        num_shots,
        num_shots_test,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None
    ):
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test

        dataset = SingleMetaDatasetClassDataset(
            root,
            source,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            shuffle_buffer_size=1000,
        )
        CombinationMetaDataset.__init__(
            self,
            dataset,
            num_ways,
            target_transform=None,
            dataset_transform=None
        )

    def __getitem__(self, index):
        return self.dataset[index]


class SingleMetaDatasetClassDataset(ClassDataset):
    def __init__(
        self,
        root,
        source,
        num_ways=5,
        num_shots=1,
        num_shots_test=15,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        shuffle_buffer_size=None
    ):
        self.root = os.path.expanduser(os.path.join(root, source))
        self.source = source
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test
        super().__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=None
        )
        if self.meta_train:
            split = Split.TRAIN
        elif self.meta_val:
            split = Split.VALID
        elif self.meta_test:
            split = Split.TEST
        else:
            raise ValueError('Unknown split')
        if source not in SOURCES[self.meta_split]:
            raise ValueError(f'The source `{source}` is not in the list of '
                             f'sources for the `{self.meta_split}` split: '
                             f'{SOURCES[self.meta_split]}')

        image_decoder = ImageDecoder(image_size=84, data_augmentation=None)

        def image_decode(example_string, source_id):
            image = image_decoder(example_string)
            return tf.transpose(image, (2, 0, 1))

        dataset_spec = load_dataset_spec(self.root)
        reader = Reader(
            dataset_spec,
            split=split,
            shuffle_buffer_size=shuffle_buffer_size,
            read_buffer_size_bytes=None,
            num_prefetch=0,
            num_to_take=-1,
            num_unique_descriptions=0
        )
        class_datasets = reader.construct_class_datasets()
        class_datasets = [dataset.map(image_decode) for dataset in class_datasets]
        self._class_datasets = class_datasets

    def get_images(self, index, images_needed):
        images_np = list(
            islice(self._class_datasets[index].as_numpy_iterator(), images_needed))
        while len(images_np) != images_needed:
            images_np += images_np[:(images_needed-len(images_np))]
        images = [torch.from_numpy(image) for image in images_np]
        return images

    def __getitem__(self, index):
        support_images, query_images = [], []
        targets = torch.randperm(self.num_ways).unsqueeze(1)
        tasks = torch.tensor(index).unsqueeze(1)
        for class_id in index:
            images = self.get_images(class_id, self.num_shots+self.num_shots_test)
            support_images.extend(images[:self.num_shots])
            query_images.extend(images[self.num_shots:])

        support_images = torch.stack(support_images, dim=0)
        support_labels = targets.repeat((1, self.num_shots)).view(-1)
        support_tasks = tasks.repeat((1, self.num_shots)).view(-1)

        query_images = torch.stack(query_images, dim=0)
        query_labels = targets.repeat((1, self.num_shots_test)).view(-1)
        query_tasks = tasks.repeat((1, self.num_shots_test)).view(-1)

        return OrderedDict([
            ('train', (support_images, support_labels, support_tasks)),
            ('test', (query_images, query_labels, query_tasks))
        ])

    @property
    def num_classes(self):
        return len(self._class_datasets)
