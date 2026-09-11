from __future__ import annotations
import math
from typing import Optional
import torch
import torchvision.transforms.functional as TF
from .constants import CORRUPTION_SPECS

def random_occlusion(
    images: torch.Tensor,
    min_ratio: float = 0.10,
    max_ratio: float = 0.40,
) -> torch.Tensor:
    """Place one random black square on each [0,1] image."""
    if images.ndim != 4:
        raise ValueError("images must have shape [B,C,H,W]")
    if not (0 <= min_ratio <= max_ratio < 1):
        raise ValueError("Need 0 <= min_ratio <= max_ratio < 1")
    out = images.clone()
    b, _, h, w = out.shape
    ratios = torch.empty(b).uniform_(min_ratio, max_ratio)
    for i in range(b):
        side = max(1, int(round(math.sqrt(float(ratios[i]) * h * w))))
        side = min(side, h, w)
        y = int(torch.randint(0, h - side + 1, (1,)).item())
        x = int(torch.randint(0, w - side + 1, (1,)).item())
        out[i, :, y:y+side, x:x+side] = 0.0
    return out

def fixed_ratio_occlusion(
    images: torch.Tensor,
    ratio: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not (0 <= ratio < 1):
        raise ValueError("ratio must be in [0,1)")
    out = images.clone()
    b, _, h, w = out.shape
    side = max(1, int(round(math.sqrt(ratio * h * w))))
    side = min(side, h, w)
    ys = torch.randint(0, h-side+1, (b,), generator=generator)
    xs = torch.randint(0, w-side+1, (b,), generator=generator)
    for i in range(b):
        y, x = int(ys[i]), int(xs[i])
        out[i, :, y:y+side, x:x+side] = 0.0
    return out

def gaussian_noise(
    images: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    noise = torch.randn(
        images.shape, generator=generator, dtype=images.dtype, device="cpu"
    ).to(images.device)
    return (images + sigma * noise).clamp(0.0, 1.0)

def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = max(3, int(round(6 * sigma + 1)))
    if kernel % 2 == 0:
        kernel += 1
    return TF.gaussian_blur(
        images, kernel_size=[kernel, kernel], sigma=[sigma, sigma]
    )

def brightness(images: torch.Tensor, factor: float) -> torch.Tensor:
    return (images * factor).clamp(0.0, 1.0)

def apply_corruption(
    images: torch.Tensor,
    name: str,
    level: int,
    generator: Optional[torch.Generator] = None,
):
    if name not in CORRUPTION_SPECS:
        raise ValueError(f"Unknown corruption: {name}")
    if level not in CORRUPTION_SPECS[name]:
        raise ValueError("level must be 1..5")
    value = float(CORRUPTION_SPECS[name][level])

    if name == "occlusion":
        out = fixed_ratio_occlusion(images, value, generator)
    elif name == "gaussian_noise":
        out = gaussian_noise(images, value, generator)
    elif name == "gaussian_blur":
        out = gaussian_blur(images, value)
    elif name == "brightness":
        out = brightness(images, value)
    else:
        raise AssertionError("unreachable")
    return out, value
