"""
损失函数模块

提供各种训练任务的损失函数:
- CrossEntropyLoss: 交叉熵损失 (预训练/SFT)
- DPOLoss: 直接偏好优化损失
- GRPOLoss: 组相对偏好优化损失
- PPOLoss: 近端策略优化损失
"""

from .cross_entropy import CrossEntropyLoss
from .dpo_loss import DPOLoss, IPO_Loss
from .grpo_loss import GRPOLoss, GroupRewardNormalizer
from .ppo_loss import PPOLoss, GAE

__all__ = [
    "CrossEntropyLoss",
    "DPOLoss",
    "IPO_Loss",
    "GRPOLoss",
    "GroupRewardNormalizer",
    "PPOLoss",
    "GAE",
]
