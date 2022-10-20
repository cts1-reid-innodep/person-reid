from dataclasses import dataclass
from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class Trainer:
    optimizer: Optimizer
    model: Module
    loss: Callable[..., Tensor]
    dataloader: DataLoader
    epochs: int

    def epoch(self):
        for x, labels in self.dataloader:
            self.optimizer.zero_grad()
            self.model.train()

            features: Tensor = self.model(x)
            loss = self.loss(features, labels)

            loss.backward()
            self.optimizer.step()

    @property
    def state(self):
        model = self.model.state_dict()
        optimizer = self.optimizer.state_dict()
        return dict(model=model, optimizer=optimizer)

    def train(self):
        for _ in range(self.epochs):
            self.epoch()
