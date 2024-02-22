from src.datasets.omniglot import Omniglot
from src.datasets.miniimagenet import MiniImagenet
from src.datasets.metadataset import MetaDataset
from src.datasets.single_metadataset import SingleMetaDataset
from src.datasets.tiered_miniimagenet import TieredImagenet
from src.datasets.sinusoid import Sinusoid
from src.datasets.harmonic import Harmonic
from src.datasets.sinusoid_line import SinusoidAndLine

__all__ = ['Omniglot', 'MiniImagenet', 'MetaDataset',
           'SingleMetaDataset', 'TieredImagenet', 'Sinusoid', 'Harmonic', 'SinusoidAndLine']
