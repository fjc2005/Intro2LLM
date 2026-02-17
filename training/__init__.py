"""
训练模块

提供训练器实现:
- Trainer: 训练器基类
- PretrainTrainer: 预训练训练器
- SFTTrainer: 监督微调训练器
- DPOTrainer: DPO 训练器
- GRPOTrainer: GRPO 训练器
- PPOTrainer: PPO 训练器
"""

from .trainer import Trainer
from .pretrain_trainer import PretrainTrainer
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainer
from .grpo_trainer import GRPOTrainer
from .ppo_trainer import PPOTrainer

__all__ = [
    "Trainer",
    "PretrainTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "PPOTrainer",
]
