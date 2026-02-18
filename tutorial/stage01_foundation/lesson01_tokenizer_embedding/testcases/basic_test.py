"""
L01: Tokenizer 与 Embedding - 基础测试

测试 TokenEmbedding 和 RoPE 的核心功能正确性。
"""

import torch
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.embedding import TokenEmbedding, RoPE


class TestTokenEmbedding:
    """测试 TokenEmbedding 类的基本功能"""

    def test_embedding_initialization(self):
        """测试嵌入层初始化"""
        vocab_size = 1000
        hidden_size = 256

        embed = TokenEmbedding(vocab_size, hidden_size)

        # 检查嵌入矩阵形状
        assert embed.embedding.weight.shape == (vocab_size, hidden_size)

    def test_embedding_forward(self):
        """测试嵌入层前向传播"""
        vocab_size = 1000
        hidden_size = 256
        batch_size = 4
        seq_len = 16

        embed = TokenEmbedding(vocab_size, hidden_size)

        # 创建随机 token IDs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 前向传播
        output = embed(input_ids)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_embedding_gradient(self):
        """测试嵌入层梯度计算"""
        vocab_size = 100
        hidden_size = 32

        embed = TokenEmbedding(vocab_size, hidden_size)
        input_ids = torch.tensor([[1, 2, 3]])

        output = embed(input_ids)
        loss = output.sum()
        loss.backward()

        # 检查梯度存在
        assert embed.embedding.weight.grad is not None


class TestRoPE:
    """测试 RoPE 类的基本功能"""

    def test_rope_initialization(self):
        """测试 RoPE 初始化"""
        dim = 64

        rope = RoPE(dim)

        # 检查 inv_freq 形状
        assert rope.inv_freq.shape == (dim // 2,)

    def test_rotate_half(self):
        """测试 rotate_half 函数"""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        result = RoPE.rotate_half(x)

        # 验证结果: [-x2, x1, -x4, x3]
        expected = torch.tensor([[-3.0, 4.0, -1.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_rope_forward(self):
        """测试 RoPE 前向传播"""
        dim = 64
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = dim // num_heads

        rope = RoPE(dim)

        # 创建 Q 和 K
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # 前向传播
        q_rot, k_rot = rope(q, k, position_ids)

        # 检查输出形状不变
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_relative_position(self):
        """测试 RoPE 的相对位置特性"""
        dim = 32
        head_dim = dim

        rope = RoPE(dim)

        # 创建单个头的 Q 和 K
        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        # 位置 0 和位置 1 的旋转应该不同
        q_rot_0, k_rot_0 = rope(q, q, torch.tensor([[0]]))
        q_rot_1, k_rot_1 = rope(q, q, torch.tensor([[1]]))

        # 旋转后的向量应该不同
        assert not torch.allclose(q_rot_0, q_rot_1)


class TestRoPEIntegration:
    """测试 RoPE 与注意力模块的集成"""

    def test_rope_with_attention_shapes(self):
        """测试 RoPE 在注意力场景下的形状"""
        dim = 64
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = dim // num_heads

        rope = RoPE(dim)

        # 模拟注意力输入
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # 应用 RoPE
        q_rot, k_rot = rope(q, k, position_ids)

        # 验证形状
        assert q_rot.shape == (batch_size, num_heads, seq_len, head_dim)
        assert k_rot.shape == (batch_size, num_heads, seq_len, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
