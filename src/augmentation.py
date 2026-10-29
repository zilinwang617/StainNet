from __future__ import annotations

import torch


class RandomFlipRotate:
    """Joint spatial augmentation for paired (x, y) tensors.

    Applies the same random horizontal flip, vertical flip, and 90-degree
    rotation to both tensors so input and label stay aligned.
    Valid for microscopy images, which have no intrinsic orientation.
    """

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-2])
            y = torch.flip(y, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[-2, -1])
        y = torch.rot90(y, k, dims=[-2, -1])
        return x, y
