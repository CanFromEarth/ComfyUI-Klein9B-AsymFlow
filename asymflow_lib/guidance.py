# Copyright (c) 2026 Hansheng Chen
# From: https://github.com/Lakonik/LakonLab

from typing import Optional
import torch


@torch.jit.script
def guidance_jit(
        pos_mean, neg_mean, guidance_scale,
        orthogonal: float = 1.0, parallel_dir: Optional[torch.Tensor] = None):
    bias = (pos_mean - neg_mean) * (guidance_scale - 1)
    if orthogonal:
        dim = list(range(1, pos_mean.dim()))
        if parallel_dir is None:
            parallel_dir = pos_mean
        bias = bias - ((bias * parallel_dir).mean(
            dim=dim, keepdim=True
        ) / (parallel_dir * parallel_dir).mean(
            dim=dim, keepdim=True
        ).clamp(min=1e-6) * parallel_dir).mul(orthogonal)
    return bias
