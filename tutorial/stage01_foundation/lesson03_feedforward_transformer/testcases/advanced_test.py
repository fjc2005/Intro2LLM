"""
L03: FeedForward 与 Transformer - 进阶测试

测试性能优化、梯度流动、数值稳定性等进阶内容。
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.feedforward import FeedForward, SwiGLU
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


class TestSwiGLUAdvanced:
    """测试 SwiGLU 进阶特性"""

    def test_swiglu_intermediate_size_ratio(self):
        """测试不同 intermediate_size 比例"""
        config = create_test_config()

        # 测试不同的中间层大小
        for ratio in [2, 3, 4]:
            config_test = create_test_config()
            config_test.intermediate_size = config_test.hidden_size * ratio

            ffn = SwiGLU(config_test)
            x = torch.randn(2, 8, config_test.hidden_size)

            output = ffn(x)

            assert output.shape == x.shape

    def test_swiglu_gradient_flow(self):
        """测试梯度流动"""
        config = create_test_config()
        config.hidden_size = 64
        config.intermediate_size = 128

        ffn = SwiGLU(config)

        x = torch.randn(1, 4, config.hidden_size, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()

        # 验证梯度存在
        assert x.grad is not None
        assert ffn.gate_proj.weight.grad is not None


class TestTransformerBlockAdvanced:
    """测试 TransformerBlock 进阶特性"""

    def test_multi_layer_stacking(self):
        """测试多层堆叠"""
        config = create_test_config()
        config.num_hidden_layers = 4
        config.hidden_size = 128

        layers = [TransformerBlock(config) for _ in range(4)]

        x = torch.randn(2, 16, config.hidden_size)

        for layer in layers:
            x, _ = layer(x)

        assert x.shape == (2, 16, config.hidden_size)

    def test_block_with_kv_cache(self):
        """测试带 KV 缓存的 Block"""
        config = create_test_config()

        block = TransformerBlock(config)

        batch_size = 1
        x = torch.randn(batch_size, 8, config.hidden_size)

        # 首次前向
        output1, kv1 = block(x, use_cache=True)

        # 使用缓存
        x2 = torch.randn(batch_size, 1, config.hidden_size)
        output2, kv2 = block(x2, past_key_value=kv1, use_cache=True)

        assert output2.shape[2] == 9  # 8 + 1

    def test_block_deep_network(self):
        """测试深层网络 (12+ 层)"""
        config = create_test_config()
        config.num_hidden_layers = 12
        config.hidden_size = 256

        layers = [TransformerBlock(config) for _ in range(12)]

        x = torch.randn(1, 32, config.hidden_size)

        for layer in layers:
            x, _ = layer(x)

        # 验证深层网络梯度不消失
        assert x.abs().mean() > 0.01


class TestFeedForwardComparison:
    """对比 GeGLU 和 SwiGLU"""

    def test_gelu_vs_silu(self):
        """对比 GELU 和 SiLU 激活函数"""
        x = torch.randn(100, 50)

        # GELU
        geglu = FeedForward(create_test_config())
        geglu.hidden_act = "gelu"
        out_ge = geglu.down_proj(
            torch.nn.functional.gelu(geglu.gate_proj(x)) * geglu.up_proj(x)
        )

        # SiLU
        swiglu = SwiGLU(create_test_config())
        out_si = swiglu.down_proj(
            torch.nn.functional.silu(swiglu.gate_proj(x)) * swiglu.up_proj(x)
        )

        # 输出形状相同
        assert out_ge.shape == out_si.shape


class TestTransformerBlockStability:
    """测试 TransformerBlock 数值稳定性"""

    def test_fp16_stability(self):
        """测试 FP16 数值稳定性"""
        config = create_test_config()
        config.hidden_size = 256

        block = TransformerBlock(config)

        x = torch.randn(4, 32, config.hidden_size, dtype=torch.float16)

        output, _ = block(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_intermediate_size(self):
        """测试大 intermediate_size"""
        config = create_test_config()
        config.hidden_size = 128
        config.intermediate_size = 2048  # 大中间层

        block = TransformerBlock(config)

        x = torch.randn(2, 16, config.hidden_size)
        output, _ = block(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestBlockPreLN:
    """测试 Pre-LN 结构"""

    def test_preln_gradient(self):
        """测试 Pre-LN 的梯度流"""
        config = create_test_config()
        config.hidden_size = 64
        config.num_hidden_layers = 6

        block = TransformerBlock(config)

        x = torch.randn(1, 8, config.hidden_size, requires_grad=True)

        output, _ = block(x)
        loss = output.sum()
        loss.backward()

        # 验证梯度可以流到输入
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
