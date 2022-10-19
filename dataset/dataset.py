from pathlib import Path
from typing import Callable, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ImageWithIdentityDataset(Dataset):
    def __init__(
        self,
        dataset: list[tuple[Path, int]],
        identities: dict[int, list[int]],
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.__dataset__ = dataset
        self.__identities__ = identities
        self.transform = transform
        self.target_transform = target_transform

    @property
    def num_identities(self) -> int:
        return len(self.__identities__)

    @property
    def identities(self) -> list[int]:
        return list(self.__identities__.keys())

    def get_indices(self, identity: int):
        return self.__identities__[identity]

    def __len__(self):
        return len(self.__dataset__)

    def __getitem__(self, index: int) -> Tuple[Tensor | Image.Image, int]:
        path, pid = self.__dataset__[index]

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            pid = self.target_transform(pid)
        return image, pid
