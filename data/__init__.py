"""
数据模块

提供数据集实现:
- BaseDataset: 数据集基类
- PretrainDataset: 预训练数据集
- SFTDataset: 监督微调数据集
- DPODataset: DPO 偏好数据集
"""

from .dataset import BaseDataset, TextDataset
from .pretrain_dataset import PretrainDataset, PackedPretrainDataset
from .sft_dataset import SFTDataset, ConversationDataset
from .dpo_dataset import DPODataset, ConversationalDPODataset

__all__ = [
    "BaseDataset",
    "TextDataset",
    "PretrainDataset",
    "PackedPretrainDataset",
    "SFTDataset",
    "ConversationDataset",
    "DPODataset",
    "ConversationalDPODataset",
]
