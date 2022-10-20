from dataclasses import dataclass
from typing import Callable

from rich.progress import Progress
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import make_progress


@dataclass
class Trainer:
    optimizer: Optimizer
    model: Module
    loss: Callable[..., Tensor]
    dataloader: DataLoader
    epochs: int
    progress: Progress = make_progress()

    def epoch(self):
        description = "Starting epoch"
        dataset_size = len(self.dataloader.sampler)
        epoch_task = self.progress.add_task(description, total=dataset_size)
        for x, labels in self.dataloader:
            self.optimizer.zero_grad()
            self.model.train()

            features: Tensor = self.model(x)
            loss = self.loss(features, labels)

            loss.backward()
            self.optimizer.step()

            description = f"Current epoch (loss: {loss.item():.5f})"
            self.progress.update(
                epoch_task,
                description=description,
                advance=len(x),
            )
        self.progress.remove_task(epoch_task)

    @property
    def state(self):
        model = self.model.state_dict()
        optimizer = self.optimizer.state_dict()
        return dict(model=model, optimizer=optimizer)

    def train(self):
        self.progress.start()
        description = "Starting training"
        train_task = self.progress.add_task(description, total=self.epochs)

        for epoch in range(self.epochs):
            description = f"Training model (epoch: {epoch})"
            self.progress.update(train_task, description=description, advance=1)
            self.epoch()

        self.progress.stop()
