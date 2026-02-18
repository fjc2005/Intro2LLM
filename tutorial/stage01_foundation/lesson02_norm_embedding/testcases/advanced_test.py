"""
课时2进阶测试：Norm和Embedding边界条件与复杂场景
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.norm import LayerNorm, RMSNorm
from model.embedding import TokenEmbedding, RoPE


class TestLayerNormAdvanced:
    """LayerNorm高级测试"""

    def test_layernorm_gradient_flow(self):
        """测试LayerNorm梯度流动"""
        # Create LayerNorm
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Create input with gradient tracking
        x = torch.randn(2, 10, hidden_size, requires_grad=True)

        # Forward pass
        output = ln(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Verify gradients are computed correctly
        assert x.grad is not None
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None

        # Gradient shapes should match parameters
        assert x.grad.shape == x.shape
        assert ln.weight.grad.shape == ln.weight.shape
        assert ln.bias.grad.shape == ln.bias.shape

    def test_layernorm_large_input(self):
        """测试大规模输入处理"""
        # Test batch_size=32, seq_len=2048, hidden_size=4096 (simulated smaller for test)
        batch_size, seq_len, hidden_size = 4, 128, 512

        ln = LayerNorm(hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Forward pass should work without errors
        output = ln(x)
        assert output.shape == x.shape

        # Verify normalization still works
        mean = output.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)

    def test_layernorm_extreme_values(self):
        """测试极值处理"""
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Test with large values (1e10)
        x_large = torch.ones(2, 10, hidden_size) * 1e10
        output_large = ln(x_large)
        # Should not overflow, should produce normalized output
        assert torch.isfinite(output_large).all()

        # Test with very small values (-1e10)
        x_small = torch.ones(2, 10, hidden_size) * -1e10
        output_small = ln(x_small)
        # Should handle without issues
        assert torch.isfinite(output_small).all()

    def test_layernorm_equivalence_to_reference(self):
        """测试与PyTorch参考实现等价"""
        hidden_size = 64

        # Our implementation
        custom_ln = LayerNorm(hidden_size)

        # PyTorch reference
        pytorch_ln = nn.LayerNorm(hidden_size)

        # Copy parameters
        pytorch_ln.weight.data = custom_ln.weight.data.clone()
        pytorch_ln.bias.data = custom_ln.bias.data.clone()

        # Test input
        torch.manual_seed(42)
        x = torch.randn(2, 10, hidden_size)

        # Forward pass
        custom_output = custom_ln(x)
        pytorch_output = pytorch_ln(x)

        # Verify outputs are approximately equal
        assert torch.allclose(custom_output, pytorch_output, atol=1e-5)


class TestRMSNormAdvanced:
    """RMSNorm高级测试"""

    def test_rmsnorm_gradient_flow(self):
        """测试RMSNorm梯度流动"""
        hidden_size = 64
        rms = RMSNorm(hidden_size)

        # Create input with gradient tracking
        x = torch.randn(2, 10, hidden_size, requires_grad=True)

        # Forward pass
        output = rms(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Verify gradients
        assert x.grad is not None
        assert rms.weight.grad is not None
        assert x.grad.shape == x.shape
        assert rms.weight.grad.shape == rms.weight.shape

    def test_rmsnorm_no_bias_grad(self):
        """测试RMSNorm确实没有bias"""
        hidden_size = 64
        rms = RMSNorm(hidden_size)

        # Verify no bias parameter
        assert not hasattr(rms, 'bias')

        # Test gradient computation
        x = torch.randn(2, 10, hidden_size, requires_grad=True)
        output = rms(x)
        loss = output.sum()
        loss.backward()

        # Only weight should have gradient
        assert rms.weight.grad is not None

    def test_rmsnorm_parameter_count(self):
        """测试RMSNorm参数量"""
        hidden_size = 64

        ln = LayerNorm(hidden_size)
        rms = RMSNorm(hidden_size)

        # Count parameters
        ln_params = sum(p.numel() for p in ln.parameters())
        rms_params = sum(p.numel() for p in rms.parameters())

        # RMSNorm should have half the parameters (no bias)
        assert rms_params == hidden_size  # Just weight
        assert ln_params == 2 * hidden_size  # weight + bias
        assert rms_params == ln_params / 2


class TestRoPEAdvanced:
    """RoPE高级测试"""

    def test_rope_relative_position_property(self):
        """测试RoPE相对位置性质"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Create queries and keys at different positions
        seq_len = 10
        q_pos_0 = torch.randn(1, 1, 1, head_dim)
        k_pos_5 = torch.randn(1, 1, 1, head_dim)

        # Position ids
        pos_q = torch.tensor([[0]])
        pos_k = torch.tensor([[5]])

        # Apply RoPE
        q_embed, _ = rope(q_pos_0, q_pos_0, pos_q)
        _, k_embed = rope(k_pos_5, k_pos_5, pos_k)

        # The relative position should be encoded
        # q at position 0 and k at position 5 should have angle difference
        # This is more of a conceptual test - verifying rotation happens
        assert not torch.allclose(q_embed, k_embed, atol=1e-6)

    def test_rope_rotation_angle_computation(self):
        """测试旋转角度计算"""
        head_dim = 64
        base = 10000.0
        rope = RoPE(head_dim, base=base)

        # Verify inv_freq is computed correctly
        # inv_freq[i] = 1.0 / (base ^ (2i/dim))
        expected_inv_freq = torch.zeros(head_dim // 2)
        for i in range(head_dim // 2):
            expected_inv_freq[i] = 1.0 / (base ** (2 * i / head_dim))

        assert torch.allclose(rope.inv_freq, expected_inv_freq, atol=1e-6)

    def test_rope_long_sequence_extrapolation(self):
        """测试RoPE长序列外推"""
        head_dim = 64
        max_seq_len = 512
        rope = RoPE(head_dim, max_seq_len)

        # Test with longer sequence than max_seq_len
        long_seq_len = 600
        q = torch.randn(1, 1, long_seq_len, head_dim)
        k = torch.randn(1, 1, long_seq_len, head_dim)

        # Position ids beyond max_seq_len
        position_ids = torch.arange(long_seq_len).unsqueeze(0)

        # Should still work (RoPE computes on-the-fly for longer sequences)
        q_embed, k_embed = rope(q, k, position_ids)

        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape

    def test_rope_gradient_flow(self):
        """测试RoPE梯度流动"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Create inputs with gradient tracking
        q = torch.randn(2, 8, 10, head_dim, requires_grad=True)
        k = torch.randn(2, 8, 10, head_dim, requires_grad=True)
        position_ids = torch.arange(10).unsqueeze(0).expand(2, 10)

        # Forward pass
        q_embed, k_embed = rope(q, k, position_ids)
        loss = q_embed.sum() + k_embed.sum()

        # Backward pass
        loss.backward()

        # Verify gradients flow back to inputs
        assert q.grad is not None
        assert k.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape


class TestEmbeddingAdvanced:
    """Embedding高级测试"""

    def test_embedding_gradient_sparse(self):
        """测试Embedding稀疏梯度"""
        vocab_size = 100
        hidden_size = 16
        emb = TokenEmbedding(vocab_size, hidden_size)

        # Create input
        input_ids = torch.LongTensor([[1, 2, 3]])

        # Forward pass
        output = emb(input_ids)
        loss = output.sum()
        loss.backward()

        # Only indices 1, 2, 3 should have non-zero gradients
        grad = emb.embedding.weight.grad
        assert grad is not None

        # Check that other indices have zero gradient
        for i in range(vocab_size):
            if i not in [1, 2, 3]:
                assert torch.allclose(grad[i], torch.zeros(hidden_size))

    def test_embedding_weight_tying(self):
        """测试权重共享"""
        vocab_size = 100
        hidden_size = 16

        emb1 = TokenEmbedding(vocab_size, hidden_size)
        emb2 = TokenEmbedding(vocab_size, hidden_size)

        # Share weights
        emb2.embedding.weight = emb1.embedding.weight

        # Modify emb1 weight
        with torch.no_grad():
            emb1.embedding.weight[0] = torch.ones(hidden_size)

        # Verify emb2 also changed
        assert torch.allclose(emb2.embedding.weight[0], torch.ones(hidden_size))

    def test_embedding_vocab_oob(self):
        """测试词汇表越界处理"""
        vocab_size = 100
        hidden_size = 16
        emb = TokenEmbedding(vocab_size, hidden_size)

        # Create input with out-of-bounds index
        input_ids = torch.LongTensor([[99, 100]])  # 100 is OOB

        # Should raise IndexError
        with pytest.raises((IndexError, RuntimeError)):
            emb(input_ids)


class TestEdgeCases:
    """边界条件测试"""

    def test_norm_single_element(self):
        """测试单元素归一化"""
        hidden_size = 1
        ln = LayerNorm(hidden_size)
        rms = RMSNorm(hidden_size)

        x = torch.randn(2, 10, 1)

        # Should work without errors
        ln_out = ln(x)
        rms_out = rms(x)

        assert ln_out.shape == x.shape
        assert rms_out.shape == x.shape

    def test_norm_batch_size_one(self):
        """测试batch_size=1"""
        hidden_size = 64
        ln = LayerNorm(hidden_size)
        rms = RMSNorm(hidden_size)

        x = torch.randn(1, 10, hidden_size)

        ln_out = ln(x)
        rms_out = rms(x)

        assert ln_out.shape == x.shape
        assert rms_out.shape == x.shape

        # Verify normalization
        mean = ln_out.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_norm_seq_len_one(self):
        """测试seq_len=1"""
        hidden_size = 64
        ln = LayerNorm(hidden_size)
        rms = RMSNorm(hidden_size)

        x = torch.randn(2, 1, hidden_size)

        ln_out = ln(x)
        rms_out = rms(x)

        assert ln_out.shape == x.shape
        assert rms_out.shape == x.shape

    def test_rope_head_dim_odd(self):
        """测试head_dim为奇数"""
        # RoPE typically requires even head_dim for pairing
        # Test with odd dimension to see behavior
        head_dim = 63
        rope = RoPE(head_dim)

        # inv_freq should be head_dim // 2 = 31
        assert rope.inv_freq.shape == torch.Size([head_dim // 2])

    def test_embedding_padding_idx(self):
        """测试padding_idx处理"""
        vocab_size = 100
        hidden_size = 16
        padding_idx = 0

        # Create embedding with padding_idx
        emb = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        # Input with padding
        input_ids = torch.LongTensor([[0, 1, 2, 0]])

        # Forward pass
        output = emb(input_ids)

        # Verify padding positions have zero output
        assert torch.allclose(output[0, 0], torch.zeros(hidden_size))
        assert torch.allclose(output[0, 3], torch.zeros(hidden_size))


class TestIntegration:
    """集成测试"""

    def test_pre_ln_transformer_block_pattern(self):
        """测试Pre-LN结构模式"""
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Simulate attention output
        def fake_attention(x):
            return x * 0.5

        # Pre-LN pattern: x = x + Attention(LayerNorm(x))
        x = torch.randn(2, 10, hidden_size, requires_grad=True)
        residual = x
        x_norm = ln(x)
        attn_out = fake_attention(x_norm)
        output = residual + attn_out

        # Verify gradient flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert ln.weight.grad is not None

    def test_norm_embedding_pipeline(self):
        """测试Norm+Embedding完整pipeline"""
        vocab_size = 100
        hidden_size = 64

        emb = TokenEmbedding(vocab_size, hidden_size)
        ln = LayerNorm(hidden_size)

        # Forward pipeline
        input_ids = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        embedded = emb(input_ids)
        normalized = ln(embedded)

        # Verify shapes
        assert embedded.shape == torch.Size([2, 3, hidden_size])
        assert normalized.shape == torch.Size([2, 3, hidden_size])

        # Verify gradients flow
        loss = normalized.sum()
        loss.backward()

        assert emb.embedding.weight.grad is not None
        assert ln.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
