from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)


def make_progress():
    sections = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]
    return Progress(*sections)
