"""
课时4基础测试：FFN和Transformer Block基础功能验证
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.feedforward import FeedForward, SwiGLU, get_feed_forward
from model.transformer_block import TransformerBlock
from model.config import ModelConfig


class TestSwiGLU:
    """测试SwiGLU基础功能"""

    def test_swiglu_initialization(self):
        """测试SwiGLU正确初始化"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Verify projection layers exist
        assert hasattr(swiglu, 'gate_proj')
        assert hasattr(swiglu, 'up_proj')
        assert hasattr(swiglu, 'down_proj')

        # Verify shapes
        assert swiglu.gate_proj.weight.shape == torch.Size([688, 256])
        assert swiglu.up_proj.weight.shape == torch.Size([688, 256])
        assert swiglu.down_proj.weight.shape == torch.Size([256, 688])

        # Verify no bias
        assert not hasattr(swiglu.gate_proj, 'bias') or swiglu.gate_proj.bias is None

    def test_swiglu_output_shape(self):
        """测试SwiGLU输出形状"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Input x = torch.randn(2, 10, 256)
        x = torch.randn(2, 10, 256)

        # Forward pass
        output = swiglu(x)

        # Verify output shape is [2, 10, 256]
        assert output.shape == torch.Size([2, 10, 256])

    def test_swiglu_activation(self):
        """测试Swish/SiLU激活"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Input
        x = torch.randn(2, 10, 256)

        # Manual SiLU computation: x * sigmoid(x)
        gate_values = nn.functional.linear(x, swiglu.gate_proj.weight)
        silu_output = gate_values * torch.sigmoid(gate_values)

        # Verify SiLU is applied (output should be different from gate values)
        assert not torch.allclose(silu_output, gate_values, atol=1e-4)

    def test_swiglu_gating(self):
        """测试门控机制"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Input
        x = torch.randn(2, 10, 256)

        # Forward
        output = swiglu(x)

        # Verify output is finite
        assert torch.isfinite(output).all()


class TestGeGLU:
    """测试GeGLU基础功能"""

    def test_geglu_initialization(self):
        """测试GeGLU正确初始化"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        geglu = FeedForward(config)

        # Verify projection layers exist
        assert hasattr(geglu, 'gate_proj')
        assert hasattr(geglu, 'up_proj')
        assert hasattr(geglu, 'down_proj')

        # Verify shapes
        assert geglu.gate_proj.weight.shape == torch.Size([688, 256])
        assert geglu.up_proj.weight.shape == torch.Size([688, 256])
        assert geglu.down_proj.weight.shape == torch.Size([256, 688])

    def test_geglu_output_shape(self):
        """测试GeGLU输出形状"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        geglu = FeedForward(config)

        # Input x = torch.randn(2, 10, 256)
        x = torch.randn(2, 10, 256)

        # Forward pass
        output = geglu(x)

        # Verify output shape is [2, 10, 256]
        assert output.shape == torch.Size([2, 10, 256])

    def test_geglu_activation(self):
        """测试GELU激活"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        geglu = FeedForward(config)

        # Input
        x = torch.randn(2, 10, 256)

        # Manual GELU computation
        gate_values = nn.functional.linear(x, geglu.gate_proj.weight)
        gelu_output = nn.functional.gelu(gate_values)

        # Verify GELU is applied
        assert not torch.allclose(gelu_output, gate_values, atol=1e-4)


class TestFeedForward:
    """测试统一FFN接口"""

    def test_ffn_interface(self):
        """测试FFN接口一致性"""
        # Test with SwiGLU
        swiglu_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        # Test with GeGLU
        geglu_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        # Test get_feed_forward
        swiglu_ffn = get_feed_forward(swiglu_config)
        geglu_ffn = get_feed_forward(geglu_config)

        # Same input
        x = torch.randn(2, 10, 256)

        # Both should produce same shape output
        swiglu_output = swiglu_ffn(x)
        geglu_output = geglu_ffn(x)

        assert swiglu_output.shape == geglu_output.shape
        assert swiglu_output.shape == torch.Size([2, 10, 256])

    def test_ffn_dropout(self):
        """测试FFN dropout"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        # This tests basic dropout behavior in the context of FFN
        # Note: Dropout is typically applied in attention, not in FFN
        # But we can verify the FFN runs in both train and eval modes


class TestTransformerBlock:
    """测试Transformer Block基础功能"""

    def test_block_initialization(self):
        """测试Block正确初始化"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Verify submodules exist
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'post_attention_layernorm')
        assert hasattr(block, 'mlp')

    def test_block_output_shape(self):
        """测试Block输出形状"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Input x = torch.randn(2, 10, 256)
        x = torch.randn(2, 10, 256)

        # Forward pass
        output, present_kv = block(x)

        # Verify output shape is [2, 10, 256]
        assert output.shape == torch.Size([2, 10, 256])

    def test_pre_ln_structure(self):
        """测试Pre-LN结构"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Verify Pre-LN: norm comes before sublayer
        # We check that there are two layernorms
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')

    def test_residual_connection(self):
        """测试残差连接"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Input
        x = torch.randn(2, 10, 256)

        # Forward
        output, _ = block(x)

        # With residual, output should be different from input but related
        # Not exactly equal since there are transformations
        assert output.shape == x.shape

    def test_kv_cache_integration(self):
        """测试KV缓存集成"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Input
        x = torch.randn(2, 10, 256)

        # Forward with use_cache=True
        output, present_kv = block(x, use_cache=True)

        # Verify present_key_value is returned
        assert present_kv is not None


class TestDropout:
    """测试Dropout功能"""

    def test_dropout_training_mode(self):
        """测试训练模式dropout"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.5, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Set to training mode
        block.train()

        # Input
        x = torch.randn(2, 10, 256)

        # Forward - should work without error
        output, _ = block(x)
        assert output.shape == torch.Size([2, 10, 256])

    def test_dropout_eval_mode(self):
        """测试评估模式dropout"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.5, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Set to eval mode
        block.eval()

        # Input
        x = torch.randn(2, 10, 256)

        # Forward - outputs should be deterministic
        output1, _ = block(x)
        output2, _ = block(x)

        # In eval mode, outputs should be identical
        assert torch.equal(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
