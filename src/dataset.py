import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import random


def get_celeba_loader(
    data_root: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    max_items: int | None = None
):
    tfm = transforms.Compose([
        transforms.CenterCrop(178),          # CelebA is 178x218-ish; crop to square
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),               # [0,1]
    ])

    ds = datasets.CelebA(
        root=data_root,
        split=split,
        target_type="attr",
        download=True,
        transform=tfm
    )

    if max_items is not None:
        rng = random.Random(123)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        indices = indices[:max_items]
        ds = Subset(ds, indices)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dl


def make_secret_from_batch(batch_imgs: torch.Tensor) -> torch.Tensor:
    """
    No need for separate 'secret' dataset:
    secret = rolled version of cover batch (random other image from same batch).
    """
    return batch_imgs.roll(shifts=1, dims=0)
