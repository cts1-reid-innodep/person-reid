from typing import Any, Sequence
from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class Transform:
    __tranform__: Any

    def __init__(self, size: int | Sequence[int] = (256, 128)):
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = [Resize(size), ToTensor(), normalize]
        self.__tranform__ = Compose(transforms)

    def __call__(self, image: Image.Image) -> Tensor:
        return self.__tranform__(image)
