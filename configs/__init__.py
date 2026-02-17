"""
配置模块

提供预定义的模型配置:
- tiny_config: Tiny LLM (~10M 参数)
- qwen3_0.5b: Qwen3 0.5B 配置
- lora_config: LoRA/QLoRA 配置
"""

from .tiny_config import TINY_CONFIG, get_tiny_config
from .qwen3_0_5b import QWEN3_0_5B_CONFIG, get_qwen3_0_5b_config
from .lora_config import LoRAConfig, QLoRAConfig, LoRAPresets

__all__ = [
    "TINY_CONFIG",
    "get_tiny_config",
    "QWEN3_0_5B_CONFIG",
    "get_qwen3_0_5b_config",
    "LoRAConfig",
    "QLoRAConfig",
    "LoRAPresets",
]
