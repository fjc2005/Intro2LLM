"""
L02: Normalization 与 Attention - 进阶测试

测试性能优化、边界情况、GQA 与 MHA 对比等进阶内容。
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


class TestRMSNormAdvanced:
    """测试 RMSNorm 进阶特性"""

    def test_rmsnorm_eps_effect(self):
        """测试不同 eps 值的影响"""
        hidden_size = 64

        x = torch.randn(2, 8, hidden_size)

        # 较小的 eps
        norm_small_eps = RMSNorm(hidden_size, eps=1e-8)
        out_small = norm_small_eps(x)

        # 较大的 eps
        norm_large_eps = RMSNorm(hidden_size, eps=1e-2)
        out_large = norm_large_eps(x)

        # 较大的 eps 应该产生更大的输出 (因为除数更大)
        assert out_large.abs().mean() > out_small.abs().mean()

    def test_rmsnorm_weight_effect(self):
        """测试可学习权重的效果"""
        hidden_size = 32

        # 初始化为全 1
        norm1 = RMSNorm(hidden_size)
        norm1.weight.data = torch.ones(hidden_size)

        # 初始化为全 2
        norm2 = RMSNorm(hidden_size)
        norm2.weight.data = torch.ones(hidden_size) * 2

        x = torch.randn(2, 8, hidden_size)

        out1 = norm1(x)
        out2 = norm2(x)

        # 权重为 2 时，输出应该是权重为 1 时的 2 倍
        assert torch.allclose(out2, out1 * 2, atol=1e-5)

    def test_rmsnorm_fp16_stability(self):
        """测试 FP16 数值稳定性"""
        hidden_size = 128

        norm = RMSNorm(hidden_size)

        # FP16 输入
        x = torch.randn(4, 32, hidden_size, dtype=torch.float16)

        output = norm(x)

        # 不应该溢出
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestAttentionAdvanced:
    """测试注意力进阶特性"""

    def test_gqa_different_group_sizes(self):
        """测试不同 GQA 分组大小"""
        config = create_test_config()
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 测试不同的 KV 头数
        for num_kv in [1, 2, 4]:
            config_test = create_test_config()
            config_test.num_key_value_heads = num_kv

            attn = GroupedQueryAttention(config_test)
            output, _ = attn(x)

            assert output.shape == x.shape

    def test_attention_large_sequence(self):
        """测试长序列处理"""
        config = create_test_config()
        config.hidden_size = 128

        attn = GroupedQueryAttention(config)

        batch_size = 1
        seq_len = 1024  # 长序列

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 不应该 OOM 或变慢太多
        output, _ = attn(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_attention_sliding_window(self):
        """测试滑动窗口注意力"""
        config = create_test_config()
        config.hidden_size = 128

        attn = GroupedQueryAttention(config)

        batch_size = 1
        seq_len = 64

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 创建滑动窗口掩码 (只关注 window_size 范围内的 token)
        window_size = 16
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=-window_size)
        mask = mask.unsqueeze(0).unsqueeze(0)

        output, _ = attn(x, attention_mask=mask)

        assert output.shape == x.shape


class TestAttentionMasking:
    """测试注意力掩码"""

    def test_causal_mask(self):
        """测试因果掩码"""
        config = create_test_config()
        config.num_key_value_heads = 2

        attn = GroupedQueryAttention(config)

        batch_size = 1
        seq_len = 8

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 因果掩码: 下三角为 0 (允许关注)，上三角为 -inf (禁止关注)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0)
        mask = mask.unsqueeze(0).unsqueeze(0)

        output, _ = attn(x, attention_mask=mask)

        # 由于使用了因果掩码，不同时序的输出应该不同
        # 验证输出与不使用掩码时不同
        output_no_mask, _ = attn(x)

        assert not torch.allclose(output, output_no_mask)

    def test_padding_mask(self):
        """测试 padding 掩码"""
        config = create_test_config()
        config.num_key_value_heads = 2

        attn = GroupedQueryAttention(config)

        batch_size = 2
        seq_len = 8

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 假设第二个样本的有效长度是 5
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 0, 0, 0]])

        # 扩展为 4D 掩码
        mask = (attention_mask.unsqueeze(1).unsqueeze(2) == 0)
        mask = mask.to(dtype=torch.float32) * float('-inf')

        output, _ = attn(x, attention_mask=mask)

        assert output.shape == x.shape


class TestAttentionKVCache:
    """测试 KV 缓存"""

    def test_kv_cache_shapes(self):
        """测试 KV 缓存形状"""
        config = create_test_config()
        config.num_key_value_heads = 2

        attn = GroupedQueryAttention(config)

        batch_size = 1
        seq_len = 8

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 首次前向
        output1, kv1 = attn(x, use_cache=True)

        # 验证缓存形状
        # key 和 value 形状: [batch, num_kv_heads, cache_len, head_dim]
        assert kv1[0].shape[0] == batch_size
        assert kv1[0].shape[1] == config.num_key_value_heads

    def test_kv_cache_autoregressive(self):
        """测试自回归生成中的 KV 缓存"""
        config = create_test_config()
        config.num_key_value_heads = 2

        attn = GroupedQueryAttention(config)

        batch_size = 1
        hidden_size = config.hidden_size
        head_dim = config.head_dim

        # 初始序列
        x = torch.randn(batch_size, 5, hidden_size)

        # 首次前向
        _, kv = attn(x, use_cache=True)

        # 逐步生成新 token
        for i in range(3):
            x_next = torch.randn(batch_size, 1, hidden_size)
            _, kv = attn(x_next, past_key_value=kv, use_cache=True)

        # 验证最终缓存长度
        # 应该是 5 + 3 = 8
        assert kv[0].shape[2] == 8


class TestNormComparison:
    """对比 LayerNorm 和 RMSNorm"""

    def test_layernorm_vs_rmsnorm(self):
        """对比 LayerNorm 和 RMSNorm 的输出"""
        hidden_size = 64
        batch_size = 2
        seq_len = 16

        x = torch.randn(batch_size, seq_len, hidden_size)

        # LayerNorm
        ln = LayerNorm(hidden_size)
        ln_out = ln(x)

        # RMSNorm
        rms = RMSNorm(hidden_size)
        rms_out = rms(x)

        # 两者都应该有相同的形状
        assert ln_out.shape == rms_out.shape == x.shape

        # 但输出不同 (因为 LayerNorm 有偏置，RMSNorm 没有)


class TestAttentionNumericalStability:
    """测试数值稳定性"""

    def test_attention_softmax_stability(self):
        """测试 Softmax 数值稳定性"""
        config = create_test_config()

        attn = GroupedQueryAttention(config)

        batch_size = 1
        seq_len = 32

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # 添加大的 logits 值，测试 softmax 稳定性
        # 注意: 实际测试时注意力在内部计算，这里只是确保不崩溃

        output, _ = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_large_batch(self):
        """测试大批量处理"""
        config = create_test_config()
        config.hidden_size = 256

        attn = GroupedQueryAttention(config)

        batch_size = 32  # 大批量
        seq_len = 64

        x = torch.randn(batch_size, seq_len, config.hidden_size)

        output, _ = attn(x)

        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
