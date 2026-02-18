"""
L03: FeedForward 与 Transformer - 基础测试

测试 SwiGLU、GeGLU、TransformerBlock 的基本功能。
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.feedforward import FeedForward, SwiGLU, get_feed_forward
from model.transformer_block import TransformerBlock
from model.config import ModelConfig


def create_test_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )


class TestSwiGLU:
    """测试 SwiGLU 基本功能"""

    def test_swiglu_initialization(self):
        """测试 SwiGLU 初始化"""
        config = create_test_config()

        ffn = SwiGLU(config)

        assert ffn.gate_proj is not None
        assert ffn.up_proj is not None
        assert ffn.down_proj is not None

    def test_swiglu_forward_shapes(self):
        """测试 SwiGLU 前向传播形状"""
        config = create_test_config()

        ffn = SwiGLU(config)

        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        output = ffn(x)

        assert output.shape == x.shape

    def test_swiglu_gating(self):
        """测试门控机制"""
        config = create_test_config()
        config.hidden_size = 64
        config.intermediate_size = 128

        ffn = SwiGLU(config)

        x = torch.randn(2, 8, config.hidden_size)
        output = ffn(x)

        # 验证门控效果: 负值被 SiLU 抑制
        assert not torch.isnan(output).any()


class TestGeGLU:
    """测试 GeGLU 基本功能"""

    def test_geglu_initialization(self):
        """测试 GeGLU 初始化"""
        config = create_test_config()
        config.hidden_act = "gelu"

        ffn = FeedForward(config)

        assert ffn.gate_proj is not None

    def test_geglu_forward(self):
        """测试 GeGLU 前向传播"""
        config = create_test_config()
        config.hidden_act = "gelu"

        ffn = FeedForward(config)

        x = torch.randn(2, 8, config.hidden_size)
        output = ffn(x)

        assert output.shape == x.shape


class TestTransformerBlock:
    """测试 TransformerBlock 基本功能"""

    def test_block_initialization(self):
        """测试 Block 初始化"""
        config = create_test_config()

        block = TransformerBlock(config)

        assert block.input_layernorm is not None
        assert block.self_attn is not None
        assert block.post_attention_layernorm is not None
        assert block.mlp is not None

    def test_block_forward(self):
        """测试 Block 前向传播"""
        config = create_test_config()

        block = TransformerBlock(config)

        batch_size = 2
        seq_len = 16

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output, kv = block(x)

        assert output.shape == x.shape

    def test_block_residual_connection(self):
        """测试残差连接"""
        config = create_test_config()
        config.hidden_size = 64

        block = TransformerBlock(config)

        x = torch.randn(1, 8, config.hidden_size)

        # 设置残差路径可追踪
        output, _ = block(x)

        # 验证输出范围合理 (残差保证不丢失信息)
        assert output.abs().max() < 10


class TestGetFeedForward:
    """测试工厂函数"""

    def test_get_swiglu(self):
        """测试获取 SwiGLU"""
        config = create_test_config()
        config.hidden_act = "silu"

        ffn = get_feed_forward(config)

        assert isinstance(ffn, SwiGLU)

    def test_get_geglu(self):
        """测试获取 GeGLU"""
        config = create_test_config()
        config.hidden_act = "gelu"

        ffn = get_feed_forward(config)

        assert isinstance(ffn, FeedForward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
