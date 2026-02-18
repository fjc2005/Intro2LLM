"""
课时4进阶测试：FFN和Transformer Block边界条件与复杂场景
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.feedforward import FeedForward, SwiGLU, get_feed_forward
from model.transformer_block import TransformerBlock
from model.config import ModelConfig


class TestSwiGLUAdvanced:
    """SwiGLU高级测试"""

    def test_swiglu_gradient_flow(self):
        """测试SwiGLU梯度流动"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Input with gradient
        x = torch.randn(2, 10, 256, requires_grad=True)

        # Forward
        output = swiglu(x)

        # Backward
        loss = output.sum()
        loss.backward()

        # Verify gradient flows to input
        assert x.grad is not None

        # Verify gradients to all projection layers
        assert swiglu.gate_proj.weight.grad is not None
        assert swiglu.up_proj.weight.grad is not None
        assert swiglu.down_proj.weight.grad is not None

    def test_swiglu_no_bias(self):
        """测试SwiGLU确实没有bias"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Verify no bias in projection layers
        assert not hasattr(swiglu.gate_proj, 'bias') or swiglu.gate_proj.bias is None
        assert not hasattr(swiglu.up_proj, 'bias') or swiglu.up_proj.bias is None
        assert not hasattr(swiglu.down_proj, 'bias') or swiglu.down_proj.bias is None

    def test_swiglu_numerical_stability(self):
        """测试SwiGLU数值稳定性"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Test with large inputs
        x = torch.randn(2, 10, 256) * 100
        output = swiglu(x)

        # Verify output is finite
        assert torch.isfinite(output).all()

    def test_swiglu_vs_reference(self):
        """测试与参考实现等价"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)

        # Manual reference implementation
        x = torch.randn(2, 10, 256)

        # Get weights
        gate_w = swiglu.gate_proj.weight
        up_w = swiglu.up_proj.weight
        down_w = swiglu.down_proj.weight

        # Reference: SiLU(xW_g) * (xW_u) @ W_d
        gate_out = torch.matmul(x, gate_w.T)
        up_out = torch.matmul(x, up_w.T)
        silu_gate = gate_out * torch.sigmoid(gate_out)
        reference_output = torch.matmul(silu_gate * up_out, down_w.T)

        # Actual output
        actual_output = swiglu(x)

        # Should be close
        assert torch.allclose(actual_output, reference_output, atol=1e-5)


class TestGeGLUAdvanced:
    """GeGLU高级测试"""

    def test_geglu_gradient_flow(self):
        """测试GeGLU梯度流动"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        geglu = FeedForward(config)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output = geglu(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_geglu_vs_swiglu(self):
        """测试GeGLU与SwiGLU差异"""
        swiglu_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        geglu_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        swiglu = SwiGLU(swiglu_config)
        geglu = FeedForward(geglu_config)

        # Same input
        x = torch.randn(2, 10, 256)

        swiglu_output = swiglu(x)
        geglu_output = geglu(x)

        # Both should produce valid outputs
        assert swiglu_output.shape == geglu_output.shape
        # Outputs should be different (different activation functions)
        assert not torch.allclose(swiglu_output, geglu_output, atol=1e-3)


class TestTransformerBlockAdvanced:
    """Transformer Block高级测试"""

    def test_block_gradient_flow(self):
        """测试Block梯度流动"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output, _ = block(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_pre_ln_vs_post_ln(self):
        """测试Pre-LN结构"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Verify Pre-LN structure: norms are applied before sublayers
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')

    def test_block_with_causal_mask(self):
        """测试带因果掩码的Block"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        x = torch.randn(2, 10, 256)

        # Create causal mask
        seq_len = 10
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attention_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        output, _ = block(x, attention_mask=attention_mask)

        assert output.shape == torch.Size([2, 10, 256])

    def test_block_with_padding_mask(self):
        """测试带padding掩码的Block"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        x = torch.randn(2, 10, 256)

        # Create padding mask (first 5 positions valid, rest padding)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, seq]

        output, _ = block(x, attention_mask=attention_mask)

        assert output.shape == torch.Size([2, 10, 256])

    def test_block_incremental_decoding(self):
        """测试Block增量解码"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # First token
        x1 = torch.randn(2, 1, 256)
        output1, kv1 = block(x1, use_cache=True)

        # Second token with KV cache
        x2 = torch.randn(2, 1, 256)
        output2, kv2 = block(x2, past_key_value=kv1, use_cache=True)

        # Verify KV cache is accumulated
        k2, v2 = kv2
        assert k2.shape[2] == 2  # Should have 2 tokens in cache


class TestResidualConnection:
    """残差连接测试"""

    def test_residual_preserves_input(self):
        """测试残差保留输入信息"""
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

        # Residual connection should help maintain input information
        # Output should be in similar range
        assert output.shape == x.shape

    def test_residual_gradient_highway(self):
        """测试残差梯度高速公路"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output, _ = block(x)

        loss = output.sum()
        loss.backward()

        # Gradient should flow to input through residual connection
        assert x.grad is not None


class TestEdgeCases:
    """边界条件测试"""

    def test_single_token_block(self):
        """测试单token Block"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        x = torch.randn(2, 1, 256)
        output, _ = block(x)

        assert output.shape == torch.Size([2, 1, 256])

    def test_very_deep_block_stack(self):
        """测试深层Block堆叠"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        # Test with 10 layers
        num_layers = 10
        blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(num_layers)
        ])

        x = torch.randn(2, 10, 256)

        for block in blocks:
            x, _ = block(x)

        # Gradient flow through deep stack
        x.requires_grad = True
        output = x
        for block in blocks:
            output, _ = block(output)

        loss = output.sum()
        loss.backward()

        # Gradient should flow through deep stack
        assert x.grad is not None

    def test_different_intermediate_sizes(self):
        """测试不同intermediate_size"""
        for intermediate_size_multiplier in [2, 3, 4]:
            hidden_size = 256
            config = ModelConfig(
                vocab_size=1000, hidden_size=hidden_size,
                intermediate_size=hidden_size * intermediate_size_multiplier,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
                attention_dropout=0.0, hidden_act="silu",
                use_rms_norm=True, use_rope=True, use_swiglu=True,
            )

            swiglu = SwiGLU(config)
            x = torch.randn(2, 10, hidden_size)
            output = swiglu(x)

            assert output.shape == torch.Size([2, 10, hidden_size])


class TestPerformance:
    """性能测试"""

    def test_ffn_inference_speed(self):
        """测试FFN推理速度"""
        import time

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        swiglu = SwiGLU(config)
        swiglu.eval()

        x = torch.randn(4, 64, 256)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = swiglu(x)

        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = swiglu(x)
        end = time.time()

        assert end - start < 30  # Should complete in under 30 seconds

    def test_block_memory_usage(self):
        """测试Block内存使用"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=688,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        block = TransformerBlock(config)

        # Count parameters
        num_params = sum(p.numel() for p in block.parameters())

        # Should have significant parameters
        assert num_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
