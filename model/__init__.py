"""
模型模块

提供大语言模型的完整实现:
- ModelConfig: 模型配置
- LayerNorm, RMSNorm: 归一化层
- TokenEmbedding, RoPE: 嵌入层
- MultiHeadAttention, GroupedQueryAttention: 注意力层
- FeedForward, SwiGLU: 前馈网络
- TransformerBlock: Transformer 块
- CausalLM: 因果语言模型
- LoRA/QLoRA: 高效微调
"""

from .config import ModelConfig
from .norm import LayerNorm, RMSNorm
from .embedding import TokenEmbedding, PositionalEncoding, RoPE
from .attention import MultiHeadAttention, GroupedQueryAttention
from .feedforward import FeedForward, SwiGLU, get_feed_forward
from .transformer_block import TransformerBlock
from .causal_lm import CausalLM, CausalLMOutputWithPast
from .lora import LoRALayer, LinearWithLoRA, get_lora_model, mark_only_lora_as_trainable
from .qlora import QLoRALinear, create_qlora_model

__all__ = [
    # 配置
    "ModelConfig",
    # 归一化
    "LayerNorm",
    "RMSNorm",
    # 嵌入
    "TokenEmbedding",
    "PositionalEncoding",
    "RoPE",
    # 注意力
    "MultiHeadAttention",
    "GroupedQueryAttention",
    # 前馈
    "FeedForward",
    "SwiGLU",
    "get_feed_forward",
    # Transformer
    "TransformerBlock",
    # 语言模型
    "CausalLM",
    "CausalLMOutputWithPast",
    # LoRA
    "LoRALayer",
    "LinearWithLoRA",
    "get_lora_model",
    "mark_only_lora_as_trainable",
    # QLoRA
    "QLoRALinear",
    "create_qlora_model",
]
