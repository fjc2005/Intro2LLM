"""
L01: Tokenizer 与 Embedding - 进阶测试

测试 RoPE 的相对位置特性、数值稳定性、性能优化等进阶内容。
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.embedding import TokenEmbedding, RoPE, PositionalEncoding


class TestRoPEAdvanced:
    """测试 RoPE 进阶特性"""

    def test_rope_long_sequence(self):
        """测试 RoPE 处理长序列"""
        dim = 128
        seq_len = 4096  # 长序列

        rope = RoPE(dim, max_position_embeddings=seq_len)

        q = torch.randn(1, 1, seq_len, dim)
        k = torch.randn(1, 1, seq_len, dim)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # 数值稳定性测试
        q_rot, k_rot = rope(q, k, position_ids)

        # 输出不应该包含 NaN 或 Inf
        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()
        assert not torch.isnan(k_rot).any()
        assert not torch.isinf(k_rot).any()

    def test_rope_different_bases(self):
        """测试不同 base 参数的影响"""
        bases = [10000.0, 50000.0, 100000.0]
        dim = 64
        seq_len = 128

        rope1 = RoPE(dim, base=bases[0])
        rope2 = RoPE(dim, base=bases[1])

        q = torch.randn(1, 1, seq_len, dim // 2)
        k = torch.randn(1, 1, seq_len, dim // 2)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        q_rot1, _ = rope1(q, k, position_ids)
        q_rot2, _ = rope2(q, k, position_ids)

        # 不同 base 应该产生不同的旋转
        assert not torch.allclose(q_rot1, q_rot2)

    def test_rope_energy_conservation(self):
        """测试 RoPE 能量守恒 (旋转不改变向量长度)"""
        dim = 64
        head_dim = dim

        rope = RoPE(dim)

        q = torch.randn(2, 4, 16, head_dim)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)

        # 计算旋转前的范数
        q_norm_before = torch.norm(q, dim=-1)

        # 旋转
        q_rot, _ = rope(q, q, position_ids)

        # 计算旋转后的范数
        q_norm_after = torch.norm(q_rot, dim=-1)

        # 范数应该保持不变 (允许数值误差)
        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5)

    def test_rope_batch_processing(self):
        """测试批量处理"""
        dim = 64
        batch_size = 8
        seq_len = 32

        rope = RoPE(dim)

        # 不同 batch 有不同的 position_ids
        q = torch.randn(batch_size, 2, seq_len, dim // 2)
        k = torch.randn(batch_size, 2, seq_len, dim // 2)

        # 使用相同的 position_ids
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        q_rot, k_rot = rope(q, k, position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestPositionalEncoding:
    """测试 Sinusoidal 位置编码"""

    def test_positional_encoding_initialization(self):
        """测试位置编码初始化"""
        d_model = 128
        max_len = 512

        pe = PositionalEncoding(d_model, max_len)

        # 检查缓冲区形状
        assert pe.pe.shape == (max_len, d_model)

    def test_positional_encoding_forward(self):
        """测试位置编码前向传播"""
        d_model = 128
        batch_size = 4
        seq_len = 32

        pe = PositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        assert output.shape == x.shape

    def test_positional_encoding_additive(self):
        """测试位置编码是加法形式"""
        d_model = 64
        seq_len = 16

        pe = PositionalEncoding(d_model)
        x = torch.randn(1, seq_len, d_model)

        output = pe(x)

        # 输出应该是原始输入 + 位置编码
        # pe[:seq_len] 应该是输出 - 输入
        assert torch.allclose(output, x + pe.pe[:seq_len], atol=1e-5)


class TestTokenEmbeddingAdvanced:
    """测试 TokenEmbedding 进阶特性"""

    def test_embedding_padding(self):
        """测试 padding token 处理"""
        vocab_size = 100
        hidden_size = 32

        embed = TokenEmbedding(vocab_size, hidden_size)

        # 测试 padding token (通常 ID 为 0)
        input_ids = torch.tensor([[0, 1, 2]])
        output = embed(input_ids)

        # 应该能正常处理
        assert output.shape == (1, 3, hidden_size)

    def test_embedding_weight_tying(self):
        """测试权重共享 (embedding 与 lm_head)"""
        vocab_size = 1000
        hidden_size = 128

        embed = TokenEmbedding(vocab_size, hidden_size)
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        # 共享权重
        lm_head.weight = embed.embedding.weight

        # 验证权重共享
        assert lm_head.weight is embed.embedding.weight

    def test_embedding_large_vocab(self):
        """测试大词表"""
        vocab_size = 50000
        hidden_size = 256

        embed = TokenEmbedding(vocab_size, hidden_size)

        input_ids = torch.randint(0, vocab_size, (4, 100))
        output = embed(input_ids)

        assert output.shape == (4, 100, hidden_size)


class TestRoPENumericalStability:
    """测试 RoPE 数值稳定性"""

    def test_rope_fp16_stability(self):
        """测试 FP16 下的数值稳定性"""
        dim = 64

        rope = RoPE(dim)

        # FP16 输入
        q = torch.randn(1, 1, 128, dim // 2, dtype=torch.float16)
        k = torch.randn(1, 1, 128, dim // 2, dtype=torch.float16)
        position_ids = torch.arange(128).unsqueeze(0)

        q_rot, k_rot = rope(q, k, position_ids)

        # 不应该溢出
        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()

    def test_rope_bf16_stability(self):
        """测试 BF16 下的数值稳定性"""
        dim = 64

        rope = RoPE(dim)

        # BF16 输入
        q = torch.randn(1, 1, 256, dim // 2, dtype=torch.bfloat16)
        k = torch.randn(1, 1, 256, dim // 2, dtype=torch.bfloat16)
        position_ids = torch.arange(256).unsqueeze(0)

        q_rot, k_rot = rope(q, k, position_ids)

        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()


class TestRoPEMathematicalProperties:
    """测试 RoPE 数学性质"""

    def test_rope_2d_rotation(self):
        """测试二维旋转"""
        # 对于 2D 向量，旋转应该精确
        x = torch.tensor([[1.0, 0.0]])  # x 轴单位向量
        angle = torch.pi / 2  # 90 度

        # 旋转矩阵: [cos, -sin; sin, cos]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        expected = torch.tensor([[0.0, 1.0]])  # 旋转后应该是 y 轴

        # 应用旋转
        x_rot = x * cos + RoPE.rotate_half(x) * sin

        assert torch.allclose(x_rot, expected, atol=1e-5)

    def test_rope_invariance_to_absolute_position(self):
        """测试绝对位置的可加性 (用于理解 RoPE)"""
        dim = 32
        rope = RoPE(dim)

        q = torch.randn(1, 1, 1, dim // 2)
        k = torch.randn(1, 1, 1, dim // 2)

        # 不同位置的旋转
        q_rot_0, k_rot_0 = rope(q, k, torch.tensor([[5]]))
        q_rot_1, k_rot_1 = rope(q, k, torch.tensor([[6]]))

        # 验证旋转是位置相关的
        assert not torch.allclose(q_rot_0, q_rot_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
