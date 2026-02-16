import torch
import math


@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    x,y in [0,1], returns average PSNR over batch.
    """
    mse = torch.mean((x - y) ** 2, dim=[1,2,3])  # per-sample
    mse = torch.clamp(mse, min=eps)
    psnr_vals = 10.0 * torch.log10(1.0 / mse)
    return float(psnr_vals.mean().item())
