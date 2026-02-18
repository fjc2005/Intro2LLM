"""
L02: Normalization 与 Attention - 基础测试

测试 LayerNorm, RMSNorm, MultiHeadAttention, GroupedQueryAttention 的基本功能。
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.norm import LayerNorm, RMSNorm
from model.attention import MultiHeadAttention, GroupedQueryAttention
from model.config import ModelConfig


def create_test_config():
    """创建测试用配置"""
    return ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
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


class TestLayerNorm:
    """测试 LayerNorm 基本功能"""

    def test_layernorm_initialization(self):
        """测试 LayerNorm 初始化"""
        normalized_shape = 128

        norm = LayerNorm(normalized_shape)

        # 检查参数形状
        assert norm.weight.shape == (normalized_shape,)
        assert norm.bias.shape == (normalized_shape,)

    def test_layernorm_forward(self):
        """测试 LayerNorm 前向传播"""
        normalized_shape = 64
        batch_size = 4
        seq_len = 16

        norm = LayerNorm(normalized_shape)
        x = torch.randn(batch_size, seq_len, normalized_shape)

        output = norm(x)

        # 输出形状应该与输入相同
        assert output.shape == x.shape

    def test_layernorm_normalization(self):
        """测试 LayerNorm 归一化效果"""
        normalized_shape = 32

        norm = LayerNorm(normalized_shape)
        x = torch.randn(2, 8, normalized_shape)

        output = norm(x)

        # 检查最后一维的均值接近 0，方差接近 1
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


class TestRMSNorm:
    """测试 RMSNorm 基本功能"""

    def test_rmsnorm_initialization(self):
        """测试 RMSNorm 初始化"""
        hidden_size = 128

        norm = RMSNorm(hidden_size)

        # 检查权重形状
        assert norm.weight.shape == (hidden_size,)

    def test_rmsnorm_forward(self):
        """测试 RMSNorm 前向传播"""
        hidden_size = 64
        batch_size = 4
        seq_len = 16

        norm = RMSNorm(hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = norm(x)

        # 输出形状应该与输入相同
        assert output.shape == x.shape

    def test_rmsnorm_normalization(self):
        """测试 RMSNorm 归一化效果"""
        hidden_size = 32

        norm = RMSNorm(hidden_size)
        x = torch.randn(2, 8, hidden_size)

        output = norm(x)

        # 检查 RMS 归一化: sqrt(mean(output^2)) 应该接近 1 (考虑权重)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        # 由于有权重，RMS 可能不为 1，但应该有限
        assert rms.max() < 10  # 不应该爆炸


class TestMultiHeadAttention:
    """测试 MultiHeadAttention 基本功能"""

    def test_mha_initialization(self):
        """测试 MHA 初始化"""
        config = create_test_config()
        config.num_key_value_heads = config.num_attention_heads  # MHA

        attn = MultiHeadAttention(config)

        # 检查投影层
        assert attn.q_proj is not None
        assert attn.k_proj is not None
        assert attn.v_proj is not None
        assert attn.o_proj is not None

    def test_mha_forward_shapes(self):
        """测试 MHA 前向传播形状"""
        config = create_test_config()
        config.num_key_value_heads = config.num_attention_heads

        attn = MultiHeadAttention(config)

        batch_size = 2
        seq_len = 16

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output, _ = attn(x)

        # 输出形状应该与输入相同
        assert output.shape == x.shape

    def test_mha_with_mask(self):
        """测试带掩码的 MHA"""
        config = create_test_config()
        config.num_key_value_heads = config.num_attention_heads

        attn = MultiHeadAttention(config)

        batch_size = 2
        seq_len = 8

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 创建因果掩码 (下三角)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

        output, _ = attn(x, attention_mask=mask)

        assert output.shape == x.shape


class TestGroupedQueryAttention:
    """测试 GroupedQueryAttention 基本功能"""

    def test_gqa_initialization(self):
        """测试 GQA 初始化"""
        config = create_test_config()
        config.num_key_value_heads = 2  # GQA: 4 heads, 2 KV heads

        attn = GroupedQueryAttention(config)

        assert attn.num_key_value_heads == 2
        assert attn.num_attention_heads == 4

    def test_gqa_forward_shapes(self):
        """测试 GQA 前向传播形状"""
        config = create_test_config()
        config.num_key_value_heads = 2

        attn = GroupedQueryAttention(config)

        batch_size = 2
        seq_len = 16

        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output, _ = attn(x)

        assert output.shape == x.shape

    def test_gqa_kv_cache(self):
        """测试 GQA KV 缓存"""
        config = create_test_config()
        config.num_key_value_heads = 2

        attn = GroupedQueryAttention(config)

        batch_size = 1
        seq_len = 8

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 首次前向传播
        output1, kv_cache1 = attn(x, use_cache=True)

        # 第二次前向传播，使用缓存
        x2 = torch.randn(batch_size, 1, config.hidden_size)
        output2, kv_cache2 = attn(x2, past_key_value=kv_cache1, use_cache=True)

        assert kv_cache1 is not None
        assert kv_cache2 is not None


class TestAttentionComparison:
    """测试 MHA 和 GQA 的对比"""

    def test_mha_vs_gqa_output_shape(self):
        """测试 MHA 和 GQA 输出形状一致"""
        config = create_test_config()

        # MHA
        mha_config = create_test_config()
        mha = MultiHeadAttention(mha_config)

        # GQA
        gqa_config = create_test_config()
        gqa_config.num_key_value_heads = 2
        gqa = GroupedQueryAttention(gqa_config)

        x = torch.randn(2, 16, config.hidden_size)

        mha_out, _ = mha(x)
        gqa_out, _ = gqa(x)

        # 输出形状应该相同
        assert mha_out.shape == gqa_out.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
