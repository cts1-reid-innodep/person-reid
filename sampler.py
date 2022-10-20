from random import choices, sample
from typing import Iterator

from torch.utils.data.sampler import Sampler

from dataset import ImageWithIdentityDataset


class RandomIdentitySampler(Sampler):
    """
    Args:
        - data_source (Dataset): dataset to sample from
        - batch_size (int): batch size
        - num_samples (int): number of samples to select for each identity.
    """

    def __init__(
        self,
        data_source: ImageWithIdentityDataset,
        batch_size: int,
        num_samples: int,
    ):
        self.data_source = data_source
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __len__(self):
        item_count = len(self.data_source)
        return item_count + (self.batch_size - item_count % self.batch_size)

    def __iter__(self) -> Iterator[int]:
        identities = self.data_source.identities
        for _ in range(len(self) // self.batch_size):
            pids = sample(identities, k=self.batch_size // self.num_samples)
            for pid in pids:
                indices = self.data_source.get_indices(pid)
                yield from choices(indices, k=self.num_samples)
