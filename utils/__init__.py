"""
工具模块

提供训练辅助工具:
- CheckpointManager: 检查点管理
- ModelSerializer: 模型序列化
- EarlyStopping: 早停机制
- ReduceLROnPlateau: 学习率调整
- MixedPrecisionTrainer: 混合精度训练
- FlashAttention: Flash Attention
- WandbLogger: W&B 日志
"""

from .checkpoint import CheckpointManager, ModelSerializer, DistributedCheckpointManager
from .early_stopping import EarlyStopping, ReduceLROnPlateau, TrainingMonitor, MetricsTracker
from .mixed_precision import MixedPrecisionTrainer, FP16Trainer, BF16Trainer, get_mixed_precision_trainer
from .flash_attention import FlashAttention, has_flash_attention, use_flash_attention
from .wandb_utils import (
    init_wandb,
    log_metrics,
    log_hyperparameters,
    log_model_checkpoint,
    watch_model,
    WandbLogger,
    finish_wandb,
)

__all__ = [
    # 检查点
    "CheckpointManager",
    "ModelSerializer",
    "DistributedCheckpointManager",
    # 早停
    "EarlyStopping",
    "ReduceLROnPlateau",
    "TrainingMonitor",
    "MetricsTracker",
    # 混合精度
    "MixedPrecisionTrainer",
    "FP16Trainer",
    "BF16Trainer",
    "get_mixed_precision_trainer",
    # Flash Attention
    "FlashAttention",
    "has_flash_attention",
    "use_flash_attention",
    # W&B
    "init_wandb",
    "log_metrics",
    "log_hyperparameters",
    "log_model_checkpoint",
    "watch_model",
    "WandbLogger",
    "finish_wandb",
]

def TODO(message: str):
    """
    TODO 宏，用于标记需要实现的功能
    """
    print(f"TODO: {message}")