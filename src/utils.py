import os
import torch
import torchvision
from datetime import datetime


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def save_image_grid(tensors, path: str, nrow: int = 4):
    """
    tensors: list[Tensor] each (B,3,H,W) in [0,1]
    Will save first nrow images from each tensor stacked vertically.
    """
    imgs = []
    for t in tensors:
        imgs.append(t[:nrow].cpu())
    # stack as (n_sets*nrow, 3, H, W)
    grid = torch.cat(imgs, dim=0)
    torchvision.utils.save_image(grid, path, nrow=nrow)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
