"""
Qwen3 0.5B 配置
模拟 Qwen3 0.5B 模型的配置。

参数规模: ~0.5B (5亿)
参考: Qwen3 技术报告
"""

from model.config import ModelConfig


def get_qwen3_0_5b_config() -> ModelConfig:
    """
    获取 Qwen3 0.5B 配置。

    模型规格:
    - 词表: 151,936 (含多语言 tokens)
    - 层数: 24
    - 隐藏层: 896
    - 注意力头: 14
    - GQA: 2 个 KV 头
    - 参数量: ~0.5B
    """
    config = ModelConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,  # GQA
        max_position_embeddings=32768,  # 支持 32K 上下文
        rope_theta=1000000.0,  # 长上下文 theta
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )
    return config


# 配置实例
QWEN3_0_5B_CONFIG = get_qwen3_0_5b_config()
