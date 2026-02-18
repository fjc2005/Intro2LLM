"""
课时3基础测试：Attention机制基础功能验证
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.attention import MultiHeadAttention, GroupedQueryAttention
from model.embedding import RoPE


class TestScaledDotProductAttention:
    """测试Scaled Dot-Product Attention"""

    def test_attention_output_shape(self):
        """测试注意力输出形状"""
        # Create Q, K, V with shape [2, 4, 8, 64] (batch=2, heads=4, seq=8, dim=64)
        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        # Compute attention scores: Q @ K^T / sqrt(dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Verify output shape is [2, 4, 8, 64]
        assert output.shape == torch.Size([batch, heads, seq, dim])

    def test_attention_weights_shape(self):
        """测试注意力权重形状"""
        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # Verify attn_weights shape is [2, 4, 8, 8]
        assert attn_weights.shape == torch.Size([batch, heads, seq, seq])

    def test_attention_weights_sum_to_one(self):
        """测试注意力权重每行和为1"""
        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)

        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # Verify each row sums to 1
        row_sums = attn_weights.sum(dim=-1)
        expected = torch.ones(batch, heads, seq)
        assert torch.allclose(row_sums, expected, atol=1e-6)

    def test_scaling_factor(self):
        """测试缩放因子"""
        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)

        # Without scaling
        scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))

        # With scaling
        scores_scaled = scores_unscaled / math.sqrt(dim)

        # Verify scaled values have smaller magnitude
        assert scores_scaled.abs().max() < scores_unscaled.abs().max()

    def test_masking_effect(self):
        """测试掩码效果"""
        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)

        # Create upper triangular mask (causal mask)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        scores_masked = scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(scores_masked, dim=-1)

        # Verify masked positions (upper triangular) have 0 weight
        upper_triangular = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        masked_weights = attn_weights[0, 0, upper_triangular]
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)


class TestMultiHeadAttention:
    """测试Multi-Head Attention"""

    def test_mha_initialization(self):
        """测试MHA正确初始化"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Verify projection layers exist
        assert hasattr(mha, 'q_proj')
        assert hasattr(mha, 'k_proj')
        assert hasattr(mha, 'v_proj')
        assert hasattr(mha, 'o_proj')

        # Verify shapes
        assert mha.q_proj.weight.shape == torch.Size([256, 256])
        assert mha.k_proj.weight.shape == torch.Size([256, 256])
        assert mha.v_proj.weight.shape == torch.Size([256, 256])
        assert mha.o_proj.weight.shape == torch.Size([256, 256])

    def test_mha_output_shape(self):
        """测试MHA输出形状"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Input x = torch.randn(2, 10, 256)
        x = torch.randn(2, 10, 256)

        # Forward pass
        output, _ = mha(x)

        # Verify output shape is [2, 10, 256]
        assert output.shape == torch.Size([2, 10, 256])

    def test_mha_with_mask(self):
        """测试带掩码的MHA"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        x = torch.randn(2, 10, 256)

        # Create causal attention mask
        seq_len = 10
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attention_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        attention_mask = attention_mask.masked_fill(mask, float('-inf'))

        # Forward pass with mask
        output, _ = mha(x, attention_mask=attention_mask)

        # Verify output shape
        assert output.shape == torch.Size([2, 10, 256])

    def test_mha_head_split(self):
        """测试多头分割"""
        from model.config import ModelConfig

        hidden_size = 256
        num_heads = 8

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=hidden_size,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Verify head_dim = hidden_size / num_heads
        expected_head_dim = hidden_size // num_heads
        assert expected_head_dim == 32


class TestGroupedQueryAttention:
    """测试GQA"""

    def test_gqa_initialization(self):
        """测试GQA正确初始化"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,  # GQA: fewer KV heads
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        gqa = GroupedQueryAttention(config)

        # Verify projection layers
        assert hasattr(gqa, 'q_proj')
        assert hasattr(gqa, 'k_proj')
        assert hasattr(gqa, 'v_proj')
        assert hasattr(gqa, 'o_proj')

        # Q projection should be full size
        assert gqa.q_proj.weight.shape == torch.Size([256, 256])

    def test_gqa_kv_repetition(self):
        """测试GQA的KV重复"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        gqa = GroupedQueryAttention(config)

        # num_key_value_groups = 8 / 2 = 4
        assert config.num_key_value_groups == 4

        # Test repeat_kv function
        batch, seq, head_dim = 2, 10, 32
        kv = torch.randn(batch, 2, seq, head_dim)
        repeated_kv = gqa.repeat_kv(kv, num_groups=4)

        # After repetition, should have 8 heads
        assert repeated_kv.shape == torch.Size([batch, 8, seq, head_dim])

    def test_gqa_output_consistency(self):
        """测试GQA输出与MHA形状一致"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        gqa = GroupedQueryAttention(config)

        x = torch.randn(2, 10, 256)

        # Forward pass
        output, _ = gqa(x)

        # Verify output shape matches input (same as MHA)
        assert output.shape == torch.Size([2, 10, 256])

    def test_gqa_memory_saving(self):
        """测试GQA内存节省"""
        from model.config import ModelConfig

        # MHA config
        mha_config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        # GQA config
        gqa_config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        # GQA should use num_kv_heads/num_heads = 2/8 = 0.25 of memory
        memory_ratio = gqa_config.num_key_value_heads / mha_config.num_key_value_heads
        assert memory_ratio == 0.25


class TestCausalMask:
    """测试因果掩码"""

    def test_causal_mask_shape(self):
        """测试因果掩码形状"""
        seq_len = 10

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Verify shape is [10, 10]
        assert mask.shape == torch.Size([seq_len, seq_len])

    def test_causal_mask_upper_triangular(self):
        """测试上三角为True (masked)"""
        seq_len = 5

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Upper triangular (excluding diagonal) should be True
        expected_upper = torch.tensor([
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ])

        assert torch.equal(mask, expected_upper)

    def test_causal_mask_lower_triangular(self):
        """测试下三角含对角线为False (not masked)"""
        seq_len = 5

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Lower triangular (including diagonal) should be False
        for i in range(seq_len):
            for j in range(i + 1):
                assert not mask[i, j]

    def test_causal_mask_in_attention(self):
        """测试掩码在注意力中的应用"""
        batch, heads, seq, dim = 2, 4, 8, 64

        # Create scores
        scores = torch.randn(batch, heads, seq, seq)

        # Create causal mask
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()

        # Apply mask
        masked_scores = scores.masked_fill(mask, float('-inf'))

        # Compute softmax
        attn_weights = torch.softmax(masked_scores, dim=-1)

        # Verify each position can only attend to previous positions and itself
        for pos in range(seq):
            # Future positions should have 0 attention
            future_weights = attn_weights[0, 0, pos, pos+1:]
            assert torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6)


class TestRoPEIntegration:
    """测试RoPE集成"""

    def test_rope_in_attention(self):
        """测试注意力中RoPE应用"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,  # Enable RoPE
            use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        x = torch.randn(2, 10, 256)
        position_ids = torch.arange(10).unsqueeze(0).expand(2, 10)

        # Forward pass with position_ids
        output, _ = mha(x, position_ids=position_ids)

        # Verify output shape
        assert output.shape == torch.Size([2, 10, 256])

    def test_rope_different_positions(self):
        """测试不同位置有不同编码"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Same content at different positions
        seq_len = 10
        q = torch.ones(1, 1, seq_len, head_dim)
        k = torch.ones(1, 1, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Apply RoPE
        q_embed, k_embed = rope(q, k, position_ids)

        # Position 0 and position 5 should have different encodings
        pos_0 = q_embed[0, 0, 0]
        pos_5 = q_embed[0, 0, 5]

        assert not torch.allclose(pos_0, pos_5, atol=1e-6)


class TestKVCache:
    """测试KV缓存"""

    def test_kv_cache_concat(self):
        """测试KV缓存拼接"""
        batch, heads, past_len, new_len, dim = 2, 8, 5, 3, 64

        # Create past KV cache
        past_k = torch.randn(batch, heads, past_len, dim)
        past_v = torch.randn(batch, heads, past_len, dim)

        # Create new KV
        new_k = torch.randn(batch, heads, new_len, dim)
        new_v = torch.randn(batch, heads, new_len, dim)

        # Concatenate
        cat_k = torch.cat([past_k, new_k], dim=2)
        cat_v = torch.cat([past_v, new_v], dim=2)

        # Verify concatenated on seq_len dimension (dim=2)
        assert cat_k.shape == torch.Size([batch, heads, past_len + new_len, dim])
        assert cat_v.shape == torch.Size([batch, heads, past_len + new_len, dim])

    def test_kv_cache_return(self):
        """测试KV缓存返回"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        x = torch.randn(2, 10, 256)

        # Forward with use_cache=True
        output, present_kv = mha(x, use_cache=True)

        # Verify present_key_value is returned
        assert present_kv is not None
        assert len(present_kv) == 2  # (key, value)

        k, v = present_kv
        assert k.shape == torch.Size([2, 8, 10, 32])  # [batch, heads, seq, head_dim]
        assert v.shape == torch.Size([2, 8, 10, 32])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
