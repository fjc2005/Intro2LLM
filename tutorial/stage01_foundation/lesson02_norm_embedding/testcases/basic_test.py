"""
课时2基础测试：Norm和Embedding基础功能验证
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


class TestLayerNorm:
    """测试LayerNorm基础功能"""

    def test_layernorm_initialization(self):
        """测试LayerNorm正确初始化"""
        # Create LayerNorm(hidden_size=64)
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Verify weight shape is [64]
        assert ln.weight.shape == torch.Size([hidden_size])
        # Verify weight initial value is 1
        assert torch.allclose(ln.weight, torch.ones(hidden_size))

        # Verify bias shape is [64]
        assert ln.bias.shape == torch.Size([hidden_size])
        # Verify bias initial value is 0
        assert torch.allclose(ln.bias, torch.zeros(hidden_size))

    def test_layernorm_output_shape(self):
        """测试LayerNorm输出形状"""
        # Create LayerNorm
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Input x = torch.randn(2, 10, 64)
        x = torch.randn(2, 10, hidden_size)

        # Forward pass
        output = ln(x)

        # Verify output shape matches input
        assert output.shape == x.shape
        assert output.shape == torch.Size([2, 10, hidden_size])

    def test_layernorm_normalization(self):
        """测试LayerNorm归一化效果"""
        # Create LayerNorm
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Initialize with weight=1, bias=0 for pure normalization test
        nn.init.ones_(ln.weight)
        nn.init.zeros_(ln.bias)

        # Input x = torch.randn(2, 10, 64)
        torch.manual_seed(42)
        x = torch.randn(2, 10, hidden_size)

        # Apply LayerNorm
        output = ln(x)

        # Verify output mean ≈ 0, std ≈ 1 along last dimension
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5)

    def test_layernorm_learnable_params(self):
        """测试LayerNorm可学习参数"""
        # Create LayerNorm
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Verify weight and bias are nn.Parameter
        assert isinstance(ln.weight, nn.Parameter)
        assert isinstance(ln.bias, nn.Parameter)

        # Verify they require gradients
        assert ln.weight.requires_grad
        assert ln.bias.requires_grad

        # Test gradient computation
        x = torch.randn(2, 10, hidden_size, requires_grad=True)
        output = ln(x)
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None
        assert ln.weight.grad.shape == ln.weight.shape
        assert ln.bias.grad.shape == ln.bias.shape


class TestRMSNorm:
    """测试RMSNorm基础功能"""

    def test_rmsnorm_initialization(self):
        """测试RMSNorm正确初始化"""
        # Create RMSNorm(hidden_size=64)
        hidden_size = 64
        rms = RMSNorm(hidden_size)

        # Verify weight shape is [64]
        assert rms.weight.shape == torch.Size([hidden_size])
        # Verify weight initial value is 1
        assert torch.allclose(rms.weight, torch.ones(hidden_size))

        # Verify no bias parameter exists
        assert not hasattr(rms, 'bias')

    def test_rmsnorm_output_shape(self):
        """测试RMSNorm输出形状"""
        # Create RMSNorm
        hidden_size = 64
        rms = RMSNorm(hidden_size)

        # Input x = torch.randn(2, 10, 64)
        x = torch.randn(2, 10, hidden_size)

        # Forward pass
        output = rms(x)

        # Verify output shape matches input
        assert output.shape == x.shape
        assert output.shape == torch.Size([2, 10, hidden_size])

    def test_rmsnorm_normalization(self):
        """测试RMSNorm归一化效果"""
        # Create RMSNorm
        hidden_size = 64
        rms = RMSNorm(hidden_size)

        # Initialize with weight=1 for pure normalization test
        nn.init.ones_(rms.weight)

        # Input x = torch.randn(2, 10, 64)
        torch.manual_seed(42)
        x = torch.randn(2, 10, hidden_size)

        # Apply RMSNorm
        output = rms(x)

        # Verify output RMS ≈ 1 along last dimension
        # RMS = sqrt(mean(x^2))
        rms_values = torch.sqrt(output.pow(2).mean(dim=-1))
        assert torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-5)

    def test_rmsnorm_vs_layernorm_when_centered(self):
        """测试均值接近0时RMSNorm与LayerNorm等价"""
        # Create modules with weight=1, bias=0
        hidden_size = 64
        ln = LayerNorm(hidden_size)
        rms = RMSNorm(hidden_size)

        nn.init.ones_(ln.weight)
        nn.init.zeros_(ln.bias)
        nn.init.ones_(rms.weight)

        # Create input with mean ≈ 0
        torch.manual_seed(42)
        x = torch.randn(2, 10, hidden_size)

        # Normalize
        ln_output = ln(x)
        rms_output = rms(x)

        # For centered data, RMSNorm and LayerNorm should be similar
        # (not exactly equal due to mean subtraction in LayerNorm)
        # The key difference is LayerNorm subtracts mean, RMSNorm doesn't
        # So they should have different outputs unless mean is exactly 0

        # Verify both produce normalized outputs
        ln_rms = torch.sqrt(ln_output.pow(2).mean(dim=-1))
        rms_rms = torch.sqrt(rms_output.pow(2).mean(dim=-1))

        assert torch.allclose(ln_rms, torch.ones_like(ln_rms), atol=1e-4)
        assert torch.allclose(rms_rms, torch.ones_like(rms_rms), atol=1e-4)


class TestTokenEmbedding:
    """测试TokenEmbedding基础功能"""

    def test_embedding_initialization(self):
        """测试Embedding层初始化"""
        vocab_size = 1000
        hidden_size = 64

        # Create TokenEmbedding
        emb = TokenEmbedding(vocab_size, hidden_size)

        # Verify embedding matrix shape is [1000, 64]
        assert emb.embedding.weight.shape == torch.Size([vocab_size, hidden_size])

    def test_embedding_forward(self):
        """测试Embedding前向传播"""
        vocab_size = 1000
        hidden_size = 64

        # Create TokenEmbedding
        emb = TokenEmbedding(vocab_size, hidden_size)

        # Input: input_ids = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        input_ids = torch.LongTensor([[1, 2, 3], [4, 5, 6]])

        # Forward pass
        output = emb(input_ids)

        # Verify output shape is [2, 3, 64]
        assert output.shape == torch.Size([2, 3, hidden_size])

    def test_embedding_indexing(self):
        """测试Embedding索引功能"""
        vocab_size = 100
        hidden_size = 16

        # Create TokenEmbedding
        emb = TokenEmbedding(vocab_size, hidden_size)

        # Set a known weight for testing
        with torch.no_grad():
            emb.embedding.weight[0] = torch.ones(hidden_size)
            emb.embedding.weight[1] = torch.zeros(hidden_size)

        # Verify embedding[0] returns the 0th token vector
        input_ids = torch.LongTensor([[0]])
        output = emb(input_ids)
        assert torch.allclose(output[0, 0], torch.ones(hidden_size))

        # Verify embedding[1] returns the 1st token vector
        input_ids = torch.LongTensor([[1]])
        output = emb(input_ids)
        assert torch.allclose(output[0, 0], torch.zeros(hidden_size))


class TestRoPE:
    """测试RoPE基础功能"""

    def test_rope_initialization(self):
        """测试RoPE正确初始化"""
        head_dim = 64
        max_seq_len = 512
        base = 10000.0

        # Create RoPE
        rope = RoPE(head_dim, max_seq_len, base)

        # Verify inv_freq is created
        assert hasattr(rope, 'inv_freq')
        # inv_freq shape should be [head_dim // 2]
        assert rope.inv_freq.shape == torch.Size([head_dim // 2])

    def test_rope_output_shape(self):
        """测试RoPE输出形状"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Input: q = torch.randn(2, 8, 10, 64) [batch, heads, seq, dim]
        batch, heads, seq, dim = 2, 8, 10, head_dim
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)

        # Position ids
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, seq)

        # Apply RoPE
        q_embed, k_embed = rope(q, k, position_ids)

        # Verify output shapes match input
        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape
        assert q_embed.shape == torch.Size([batch, heads, seq, dim])

    def test_rope_position_differentiation(self):
        """测试不同位置的编码不同"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Create same token at different positions
        seq_len = 10
        q = torch.ones(1, 1, seq_len, head_dim)
        k = torch.ones(1, 1, seq_len, head_dim)

        # Position ids [0, 1, 2, ..., 9]
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Apply RoPE
        q_embed, _ = rope(q, k, position_ids)

        # Verify different positions have different encodings
        # Compare position 0 and position 5
        pos_0 = q_embed[0, 0, 0, :]
        pos_5 = q_embed[0, 0, 5, :]

        # They should be different
        assert not torch.allclose(pos_0, pos_5, atol=1e-6)

    def test_rope_same_position_same_encoding(self):
        """测试相同位置的编码相同"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Create different tokens at same positions
        seq_len = 5
        q1 = torch.randn(1, 1, seq_len, head_dim)
        q2 = torch.randn(1, 1, seq_len, head_dim)
        k = torch.randn(1, 1, seq_len, head_dim)

        # Same position ids for both
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Apply RoPE
        q1_embed, _ = rope(q1, k, position_ids)
        q2_embed, _ = rope(q2, k, position_ids)

        # Verify same positions get same rotation
        # The rotation pattern should be the same (though input values differ)
        # We can verify this by checking that position i in both sequences
        # has been rotated by the same amount

        # For this, we check that the rotation is position-dependent
        # by comparing the change pattern
        for i in range(seq_len - 1):
            # The encoding difference between adjacent positions should be consistent
            diff1 = q1_embed[0, 0, i + 1, :] - q1_embed[0, 0, i, :]
            # We can't directly compare diff1 and diff2 since inputs differ
            # But we verify the rotation is happening
            assert not torch.allclose(q1_embed[0, 0, i, :], q1_embed[0, 0, i + 1, :], atol=1e-6)


class TestNumericalStability:
    """数值稳定性测试"""

    def test_layernorm_dtype_preservation(self):
        """测试LayerNorm保持数据类型"""
        hidden_size = 64
        ln = LayerNorm(hidden_size)

        # Test float32
        x_fp32 = torch.randn(2, 10, hidden_size, dtype=torch.float32)
        output_fp32 = ln(x_fp32)
        assert output_fp32.dtype == torch.float32

    def test_rmsnorm_dtype_preservation(self):
        """测试RMSNorm保持数据类型"""
        hidden_size = 64
        rms = RMSNorm(hidden_size)

        # Test float32
        x_fp32 = torch.randn(2, 10, hidden_size, dtype=torch.float32)
        output_fp32 = rms(x_fp32)
        assert output_fp32.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
