"""
Tiny LLM 配置
一个小型语言模型配置，适合教学和快速实验。

参数规模: ~10M
特点:
- 训练快速
- 显存需求低
- 适合验证代码正确性
"""

from model.config import ModelConfig


def get_tiny_config() -> ModelConfig:
    """
    获取 Tiny LLM 配置。

    模型规格:
    - 词表: 10,000
    - 层数: 4
    - 隐藏层: 256
    - 注意力头: 4
    - 参数量: ~10M
    """
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )
    return config


# 配置实例
TINY_CONFIG = get_tiny_config()
