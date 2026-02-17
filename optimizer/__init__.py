"""
优化器模块

提供优化器和学习率调度器:
- AdamW: Adam with decoupled weight decay
- Lion: EvoLved Sign Momentum optimizer
- WarmupCosineScheduler: Warmup + Cosine Annealing
- CosineAnnealingWarmRestarts: SGDR scheduler
"""

from .adamw import AdamW, AdamW8bit
from .lion import Lion, LionW
from .scheduler import (
    CosineAnnealingWarmRestarts,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    get_scheduler,
)

__all__ = [
    "AdamW",
    "AdamW8bit",
    "Lion",
    "LionW",
    "CosineAnnealingWarmRestarts",
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "get_scheduler",
]
