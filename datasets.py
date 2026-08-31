from pathlib import Path
from typing import Any, Callable

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class CatsAndDogs(ImageFolder):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ):
        super().__init__(
            Path(root, "PetImages"),
            transform,
            target_transform,
            is_valid_file=self.__validate_file,
        )

    @staticmethod
    def __validate_file(filepath: str) -> bool:
        path = Path(filepath)
        return path.suffix == ".jpg" and path.stat().st_size > 0


class ChestXRay(ImageFolder):
    def __init__(self, root: str | Path, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, loader: Callable[[str], Any] = default_loader, is_valid_file: Callable[[str], bool] | None = None, allow_empty: bool = False):
        super().__init__(root, transform, target_transform, loader, is_valid_file, allow_empty)