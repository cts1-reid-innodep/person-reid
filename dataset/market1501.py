import re
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import requests
from rich import print
from rich.progress import Progress, track
from rich.prompt import Prompt

# Regex pattern to match person id and camera id
IMAGE_NAME_PATTERN = re.compile(r"([-\d]+)_c(\d)")


def get_image_info(path: Path) -> tuple[int]:
    pid, _ = map(int, IMAGE_NAME_PATTERN.search(path.name).groups())
    return path, pid


def download_dataset(root: Path | str = "data"):
    """
    Download the Market-1501 dataset into `market1501` directory under `root`.
    """

    DOWNLOAD_URL = "https://github.com/maecharlie/person-reid/releases/download/v0.1.0/market1501.zip"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "market1501"
    if dataset_path.exists():
        message = f"Removing [blue]{dataset_path}[/blue]..."
        with Progress() as progress:
            task = progress.add_task(message, total=None)
            shutil.rmtree(dataset_path)
            progress.update(task, total=100, completed=100)

    response = requests.get(DOWNLOAD_URL, stream=True)
    response.raise_for_status()

    size = response.headers.get("content-length")

    archive_path = root / f"market1501.zip"

    if archive_path.exists():
        print(f"Archive [blue]{archive_path}[/blue] exists, skipping download.")
    else:
        try:
            with archive_path.open(mode="wb+") as f:
                if size is None:
                    f.write(response.content)
                    return

                with Progress() as progress:
                    description = "Downloading market1501 dataset ..."
                    task = progress.add_task(description, total=int(size))
                    for chunk in response.iter_content(chunk_size=1048576):
                        progress.advance(task_id=task, advance=len(chunk))
                        f.write(chunk)
        except KeyboardInterrupt:
            archive_path.unlink()
            return

    with ZipFile(str(archive_path.absolute())) as archive:
        description = f"Extracting dataset from {archive_path}"
        for entry in track(archive.filelist, description):
            archive.extract(entry, root)
        archive_path.unlink()


class Market1501:
    def __init__(self, root: str | Path = "data"):
        self.__root__ = Path(root) / "market1501"

        if not self.__root__.exists():
            print(f"[blue]{self.__root__}[/blue] not found.")
            answer = Prompt.ask(
                prompt=f"Download Market-1501 dataset to {self.__root__}",
                choices=["yes", "no"],
                default="yes",
            )
            if answer == "yes":
                self.__root__.parent.mkdir(parents=True, exist_ok=True)
                download_dataset(self.__root__.parent)

        if not self.__root__.exists():
            raise FileNotFoundError(f"{self.__root__} not found.")

        self.train = self.__process__("bounding_box_train")
        self.test = self.__process__("bounding_box_test")
        self.query = self.__process__("query")

    def __process__(self, subdir: str) -> dict:
        path = self.__root__ / subdir
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

        dataset = [
            (image_path, pid)
            for image_path, pid in map(get_image_info, path.glob("*.jpg"))
            if pid != -1
        ]

        identities = defaultdict(list)
        for index, (_, pid) in enumerate(dataset):
            identities[pid].append(index)

        return dict(dataset=dataset, identities=identities)
