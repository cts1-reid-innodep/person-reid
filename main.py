from pathlib import Path

import rich.traceback
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tranform import Transform

from dataset import ImageWithIdentityDataset, Market1501
from loss import TripletLoss
from model import initialize_model
from sampler import RandomIdentitySampler
from trainer import Trainer
from utils import make_progress

rich.traceback.install()
device = "cuda" if torch.cuda.is_available() else "cpu"
market1501 = Market1501()

train_dataset = ImageWithIdentityDataset(
    dataset=market1501.train["dataset"],
    identities=market1501.train["identities"],
    transform=Transform(),
)


batch_size = 64
num_samples = 8

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=RandomIdentitySampler(train_dataset, batch_size, num_samples),
    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)


model_path = Path("checkpoint.pth")

model = initialize_model("resnet50", num_classes=train_dataset.num_identities)

optimizer = Adam(model.parameters())

model = model.to(device)

if model_path.exists():
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

trainer = Trainer(
    optimizer=optimizer,
    model=model,
    dataloader=train_dataloader,
    loss=TripletLoss(),
    epochs=300,
)
try:
    trainer.train()
except KeyboardInterrupt:
    trainer.progress.stop()
    with make_progress() as progress:
        description = f"Saving model to [blue]{model_path}[/blue]"
        task = progress.add_task(description, total=None)
        torch.save(trainer.state, model_path)
        progress.update(task, total=100, completed=100)
