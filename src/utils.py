import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from collections import namedtuple, OrderedDict
from src.datasets import Omniglot, MiniImagenet, MetaDataset, SingleMetaDataset, TieredImagenet, Sinusoid, Harmonic, SinusoidAndLine
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision import transforms

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                       'meta_test_dataset model loss_function')


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(logits, targets):
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
                             for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
                              for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()


class ToTensor1D(object):
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def init_metadataset_data(name="meta-dataset", sub_dataset_name=None):
    if name == "meta-dataset":
        train_set = ['ilsvrc_2012', 'omniglot', 'aircraft',
                     'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
        validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                          'mscoco']
        test_set = ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
                    "vgg_flower", "traffic_sign", "mscoco"]
    else:
        train_set = [sub_dataset_name]
        validation_set = [sub_dataset_name]
        test_set = [sub_dataset_name]

    return train_set, validation_set, test_set


def get_benchmark_by_name(model_name,
                          name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          image_size=84,
                          hidden_size=None,
                          metaoptnet_embedding='ResNet',
                          metaoptnet_head='SVM-CS',
                          use_augmentations=False,
                          sub_dataset_name=None):
    """Get dataset, model and loss function"""
    from src.maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
    from src.reptile.model import ModelConvOmniglot as ModelConvOmniglotReptile
    from src.reptile.model import ModelConvMiniImagenet as ModelConvMiniImagenetReptile
    from src.protonet.model import Protonet_Omniglot, Protonet_MiniImagenet
    from src.protonet.metalearners.loss import prototypical_loss
    from src.matching_networks.model import MatchingNetwork
    from src.cnaps.model import Cnaps
    from src.cnaps.metalearners.loss import CNAPsLoss
    from src.metaoptnet.model import MetaOptNet
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)

    # Classification

    if name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = []
        if use_augmentations:
            transform.append(transforms.RandomCrop(image_size, padding=8))
            transform.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            transform.append(transforms.RandomHorizontalFlip())

        transform.append(Resize(image_size))
        transform.append(ToTensor())
        transform = Compose(transform)
        try:
            meta_train_dataset = Omniglot(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          class_augmentations=class_augmentations,
                                          dataset_transform=dataset_transform,
                                          download=False)
        except Exception:
            meta_train_dataset = Omniglot(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          class_augmentations=class_augmentations,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        if model_name == 'maml':
            model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvOmniglotReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_Omniglot()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=1, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=28)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss
    elif name == 'miniimagenet':
        transform = []
        if use_augmentations:
            transform.append(transforms.RandomCrop(image_size, padding=8))
            transform.append(transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(Resize(image_size))
        transform.append(ToTensor())
        transform = Compose(transform)
        try:
            meta_train_dataset = MiniImagenet(folder,
                                              transform=transform,
                                              target_transform=Categorical(num_ways),
                                              num_classes_per_task=num_ways,
                                              meta_train=True,
                                              dataset_transform=dataset_transform,
                                              download=False)
        except Exception:
            meta_train_dataset = MiniImagenet(folder,
                                              transform=transform,
                                              target_transform=Categorical(num_ways),
                                              num_classes_per_task=num_ways,
                                              meta_train=True,
                                              dataset_transform=dataset_transform,
                                              download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        if model_name == 'maml':
            model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvMiniImagenetReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_MiniImagenet()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=3, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=84)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss
    elif name == 'tiered_imagenet':
        transform = []
        if use_augmentations:
            transform.append(transforms.RandomCrop(image_size, padding=8))
            transform.append(transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(Resize(image_size))
        transform.append(ToTensor())
        transform = Compose(transform)
        try:
            meta_train_dataset = TieredImagenet(folder,
                                                transform=transform,
                                                target_transform=Categorical(num_ways),
                                                num_classes_per_task=num_ways,
                                                meta_train=True,
                                                dataset_transform=dataset_transform,
                                                download=False)
        except Exception:
            meta_train_dataset = TieredImagenet(folder,
                                                transform=transform,
                                                target_transform=Categorical(num_ways),
                                                num_classes_per_task=num_ways,
                                                meta_train=True,
                                                dataset_transform=dataset_transform,
                                                download=True)
        meta_val_dataset = TieredImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_val=True,
                                          dataset_transform=dataset_transform)
        meta_test_dataset = TieredImagenet(folder,
                                           transform=transform,
                                           target_transform=Categorical(num_ways),
                                           num_classes_per_task=num_ways,
                                           meta_test=True,
                                           dataset_transform=dataset_transform)

        if model_name == 'maml':
            model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvMiniImagenetReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_MiniImagenet()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=3, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=84)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss
    elif name == 'meta_dataset':
        train_set, validation_set, test_set = init_metadataset_data()
        meta_train_dataset = MetaDataset(folder, num_ways=num_ways, num_shots=num_shots, num_shots_test=num_shots_test,
                                         meta_train=True)
        meta_val_dataset = MetaDataset(folder, num_ways=num_ways, num_shots=num_shots, num_shots_test=num_shots_test,
                                       meta_val=True)
        meta_test_dataset = MetaDataset(folder, num_ways=num_ways, num_shots=num_shots, num_shots_test=num_shots_test,
                                        meta_test=True)

        if model_name == 'maml':
            model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvMiniImagenetReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_MiniImagenet()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=3, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=84)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss
    elif name == 'single_meta_dataset':
        train_set, validation_set, test_set = init_metadataset_data(name, sub_dataset_name)
        meta_train_dataset = SingleMetaDataset(folder, source=sub_dataset_name,
                                               num_ways=num_ways, num_shots=num_shots, num_shots_test=num_shots_test,
                                               meta_train=True)
        meta_val_dataset = SingleMetaDataset(folder, source=sub_dataset_name,
                                             num_ways=num_ways, num_shots=num_shots, num_shots_test=num_shots_test,
                                             meta_val=True)
        meta_test_dataset = SingleMetaDataset(folder, source=sub_dataset_name,
                                              num_ways=num_ways, num_shots=num_shots, num_shots_test=num_shots_test,
                                              meta_test=True)

        if model_name == 'maml':
            model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'reptile':
            model = ModelConvMiniImagenetReptile(num_ways, hidden_size=hidden_size)
            loss_function = F.cross_entropy
        elif model_name == 'protonet':
            model = Protonet_MiniImagenet()
            loss_function = prototypical_loss
        elif model_name == 'matching_networks':
            model = MatchingNetwork(keep_prob=0, batch_size=32, num_channels=3, fce=False, num_classes_per_set=num_ways,
                                    num_samples_per_class=num_shots, image_size=84)
            loss_function = torch.nn.NLLLoss
        elif model_name == 'cnaps':
            model = Cnaps()
            loss_function = CNAPsLoss
        elif model_name == 'metaoptnet':
            model = MetaOptNet(name, metaoptnet_embedding, metaoptnet_head,
                               num_ways, num_shots, num_shots_test)
            loss_function = torch.nn.NLLLoss

    # Regression

    elif name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)
        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss
    elif name == 'harmonic':
        transform = ToTensor1D()

        meta_train_dataset = Harmonic(num_shots + num_shots_test,
                                      num_tasks=5000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Harmonic(num_shots + num_shots_test,
                                    num_tasks=5000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Harmonic(num_shots + num_shots_test,
                                     num_tasks=5000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)
        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss
    elif name == 'sinusoid_line':
        transform = ToTensor1D()

        meta_train_dataset = SinusoidAndLine(num_shots + num_shots_test,
                                             num_tasks=1000000,
                                             transform=transform,
                                             target_transform=transform,
                                             dataset_transform=dataset_transform)
        meta_val_dataset = SinusoidAndLine(num_shots + num_shots_test,
                                           num_tasks=1000000,
                                           transform=transform,
                                           target_transform=transform,
                                           dataset_transform=dataset_transform)
        meta_test_dataset = SinusoidAndLine(num_shots + num_shots_test,
                                            num_tasks=1000000,
                                            transform=transform,
                                            target_transform=transform,
                                            dataset_transform=dataset_transform)
        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
