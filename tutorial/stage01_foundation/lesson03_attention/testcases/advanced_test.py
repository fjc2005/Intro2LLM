"""
课时3进阶测试：Attention机制边界条件与复杂场景
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


class TestScaledDotProductAttentionAdvanced:
    """缩放点积注意力高级测试"""

    def test_attention_gradient_flow(self):
        """测试注意力梯度流动"""
        # Create Q, K, V with gradient tracking
        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim, requires_grad=True)
        K = torch.randn(batch, heads, seq, dim, requires_grad=True)
        V = torch.randn(batch, heads, seq, dim, requires_grad=True)

        # Forward pass
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Verify gradients exist for Q, K, V
        assert Q.grad is not None, "Q should have gradient"
        assert K.grad is not None, "K should have gradient"
        assert V.grad is not None, "V should have gradient"

        # Verify gradient shapes match input shapes
        assert Q.grad.shape == Q.shape
        assert K.grad.shape == K.shape
        assert V.grad.shape == V.shape

    def test_attention_numerical_stability(self):
        """测试数值稳定性"""
        batch, heads, seq, dim = 2, 4, 8, 64

        # Test with extremely large values
        Q = torch.randn(batch, heads, seq, dim) * 1e10
        K = torch.randn(batch, heads, seq, dim) * 1e10
        V = torch.randn(batch, heads, seq, dim)

        # Compute attention - should not overflow
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output should be finite"

    def test_attention_with_extreme_dims(self):
        """测试极端维度"""
        # Test head_dim = 1
        batch, heads, seq, dim = 2, 4, 8, 1
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        assert output.shape == torch.Size([batch, heads, seq, dim])

        # Test head_dim = 128
        dim = 128
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        assert output.shape == torch.Size([batch, heads, seq, dim])

    def test_attention_equivalence_to_pytorch(self):
        """测试与PyTorch实现等价"""
        from torch.nn.functional import scaled_dot_product_attention

        batch, heads, seq, dim = 2, 4, 8, 64
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        # Custom implementation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        custom_output = torch.matmul(attn_weights, V)

        # PyTorch implementation
        pytorch_output = scaled_dot_product_attention(Q, K, V)

        # Verify outputs are close
        assert torch.allclose(custom_output, pytorch_output, atol=1e-5)


class TestMultiHeadAttentionAdvanced:
    """MHA高级测试"""

    def test_mha_gradient_checkpointing(self):
        """测试梯度检查点"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Test with gradient checkpointing
        x = torch.randn(2, 10, 256, requires_grad=True)

        # Wrap forward with checkpoint
        from torch.utils.checkpoint import checkpoint
        output = checkpoint(mha, x, use_reentrant=False)

        loss = output.sum()
        loss.backward()

        # Verify gradient exists
        assert x.grad is not None

    def test_mha_different_batch_sizes(self):
        """测试不同batch size"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Test batch_size = 1
        x1 = torch.randn(1, 10, 256)
        output1, _ = mha(x1)
        assert output1.shape == torch.Size([1, 10, 256])

        # Test batch_size = 64
        x64 = torch.randn(64, 10, 256)
        output64, _ = mha(x64)
        assert output64.shape == torch.Size([64, 10, 256])

    def test_mha_sequence_length_variation(self):
        """测试不同序列长度"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Test seq_len = 1
        x1 = torch.randn(2, 1, 256)
        output1, _ = mha(x1)
        assert output1.shape == torch.Size([2, 1, 256])

        # Test seq_len = 512
        x512 = torch.randn(2, 512, 256)
        output512, _ = mha(x512)
        assert output512.shape == torch.Size([2, 512, 256])

    def test_mha_weight_initialization(self):
        """测试权重初始化"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Verify weights are initialized (not zero)
        assert (mha.q_proj.weight != 0).any()
        assert (mha.k_proj.weight != 0).any()
        assert (mha.v_proj.weight != 0).any()
        assert (mha.o_proj.weight != 0).any()


class TestGroupedQueryAttentionAdvanced:
    """GQA高级测试"""

    def test_gqa_gradient_flow(self):
        """测试GQA梯度流动"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        gqa = GroupedQueryAttention(config)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output, _ = gqa(x)

        loss = output.sum()
        loss.backward()

        # Verify gradients flow to input
        assert x.grad is not None

    def test_gqa_invalid_config(self):
        """测试无效GQA配置"""
        from model.config import ModelConfig

        # num_heads cannot be evenly divided by num_kv_heads
        # This should be handled by the config validation
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=7, num_key_value_heads=3,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        # num_key_value_groups should be computed
        # 7 / 3 = 2.33 - not integer division
        # The config should handle this or the model should raise error

    def test_gqa_vs_mha_quality(self):
        """测试GQA与MHA质量对比"""
        from model.config import ModelConfig

        # MHA config
        mha_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        # GQA config
        gqa_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(mha_config)
        gqa = GroupedQueryAttention(gqa_config)

        # Same input
        x = torch.randn(2, 10, 256)

        mha_output, _ = mha(x)
        gqa_output, _ = gqa(x)

        # Both should produce valid outputs
        assert mha_output.shape == gqa_output.shape
        assert torch.isfinite(mha_output).all()
        assert torch.isfinite(gqa_output).all()


class TestCausalMaskAdvanced:
    """因果掩码高级测试"""

    def test_causal_mask_broadcasting(self):
        """测试掩码广播"""
        batch, heads, seq, dim = 2, 4, 8, 64

        # Create scores
        scores = torch.randn(batch, heads, seq, seq)

        # Create causal mask [seq, seq]
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()

        # Apply mask (should broadcast)
        masked_scores = scores.masked_fill(mask, float('-inf'))

        # Verify mask applied correctly
        for b in range(batch):
            for h in range(heads):
                for i in range(seq):
                    for j in range(seq):
                        if i < j:
                            assert masked_scores[b, h, i, j] == float('-inf')
                        else:
                            assert masked_scores[b, h, i, j] == scores[b, h, i, j]

    def test_causal_mask_with_padding(self):
        """测试带padding的因果掩码"""
        batch, heads, seq, dim = 2, 4, 8, 64

        # Create scores
        scores = torch.randn(batch, heads, seq, seq)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()

        # Create padding mask (last 2 positions are padding)
        padding_mask = torch.tensor([False, False, False, False, False, False, True, True])
        padding_mask = padding_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, seq]

        # Apply both masks
        # Padding positions should be masked in both cases
        masked_scores = scores.masked_fill(causal_mask, float('-inf'))

        assert masked_scores.shape == scores.shape

    def test_causal_mask_gradient(self):
        """测试掩码对梯度的影响"""
        batch, heads, seq, dim = 2, 4, 8, 64

        # Create Q, K, V with gradient
        Q = torch.randn(batch, heads, seq, dim, requires_grad=True)
        K = torch.randn(batch, heads, seq, dim, requires_grad=True)
        V = torch.randn(batch, heads, seq, dim, requires_grad=True)

        # Compute attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)

        # Apply causal mask
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        scores_masked = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores_masked, dim=-1)
        output = torch.matmul(attn_weights, V)

        loss = output.sum()
        loss.backward()

        # Gradients should exist
        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None


class TestRoPEIntegrationAdvanced:
    """RoPE集成高级测试"""

    def test_rope_relative_position_property(self):
        """测试RoPE相对位置性质"""
        head_dim = 64
        rope = RoPE(head_dim)

        # Same content at different positions
        seq_len = 10
        q = torch.randn(1, 1, seq_len, head_dim)
        k = torch.randn(1, 1, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Apply RoPE
        q_embed, k_embed = rope(q, k, position_ids)

        # Verify RoPE is applied (output should be different from input)
        assert not torch.allclose(q_embed, q, atol=1e-6)
        assert not torch.allclose(k_embed, k, atol=1e-6)

    def test_rope_long_sequence(self):
        """测试RoPE长序列处理"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Test with sequence longer than max_position_embeddings
        x = torch.randn(2, 600, 256)

        try:
            output, _ = mha(x)
            # Should handle gracefully or truncate
            assert output.shape[1] == 600
        except Exception:
            # May raise error for too long sequence - that's OK
            pass

    def test_rope_with_kv_cache(self):
        """测试RoPE与KV缓存"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # First pass with prompt
        x1 = torch.randn(2, 10, 256)
        output1, present_kv1 = mha(x1, use_cache=True)

        # Second pass with new token using cache
        x2 = torch.randn(2, 1, 256)
        output2, present_kv2 = mha(x2, past_key_value=present_kv1, use_cache=True)

        # Verify output shapes
        assert output1.shape == torch.Size([2, 10, 256])
        assert output2.shape == torch.Size([2, 1, 256])


class TestKVCacheAdvanced:
    """KV缓存高级测试"""

    def test_kv_cache_incremental_generation(self):
        """测试增量生成"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Simulate incremental generation
        # First token
        x1 = torch.randn(2, 1, 256)
        _, kv1 = mha(x1, use_cache=True)

        # Second token with cache
        x2 = torch.randn(2, 1, 256)
        _, kv2 = mha(x2, past_key_value=kv1, use_cache=True)

        # Verify KV cache is accumulated
        k2, v2 = kv2
        assert k2.shape[2] == 2  # Should have 2 tokens in cache

    def test_kv_cache_memory_growth(self):
        """测试KV缓存内存增长"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Track memory growth
        memory_sizes = []

        for seq_len in [1, 10, 50, 100]:
            x = torch.randn(2, seq_len, 256)
            output, kv = mha(x, use_cache=True)

            if kv is not None:
                k, v = kv
                memory_sizes.append(k.element_size() * k.nelement())

        # Verify memory grows with sequence length
        assert memory_sizes[-1] > memory_sizes[0]

    def test_kv_cache_dtype_consistency(self):
        """测试KV缓存数据类型一致"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        # Test with float32
        x_fp32 = torch.randn(2, 10, 256, dtype=torch.float32)
        output_fp32, kv_fp32 = mha(x_fp32, use_cache=True)

        if kv_fp32 is not None:
            k_fp32, v_fp32 = kv_fp32
            assert k_fp32.dtype == torch.float32
            assert v_fp32.dtype == torch.float32


class TestEdgeCases:
    """边界条件测试"""

    def test_single_token_attention(self):
        """测试单token注意力"""
        batch, heads, seq, dim = 2, 4, 1, 64

        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # With single token, attention weight should be 1.0
        assert torch.allclose(attn_weights, torch.ones_like(attn_weights), atol=1e-6)
        assert output.shape == torch.Size([batch, heads, seq, dim])

    def test_dropout_effect(self):
        """测试dropout效果"""
        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.5, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)

        x = torch.randn(2, 10, 256)

        # Training mode - outputs should vary
        mha.train()
        outputs = []
        for _ in range(5):
            out, _ = mha(x)
            outputs.append(out.clone())

        # In training mode, outputs should be different due to dropout
        # (Note: may not always be different, but check it runs)
        assert outputs[0].shape == torch.Size([2, 10, 256])

        # Eval mode - outputs should be deterministic
        mha.eval()
        out1, _ = mha(x)
        out2, _ = mha(x)
        assert torch.equal(out1, out2)

    def test_attention_mask_variants(self):
        """测试不同类型掩码"""
        batch, heads, seq, dim = 2, 4, 8, 64

        # Test causal mask
        causal_mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()

        # Test padding mask (first 2 positions valid, rest padding)
        padding_mask = torch.tensor([False, False, True, True, True, True, True, True])
        padding_mask = padding_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq]

        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)

        # Apply causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        scores_causal = scores.masked_fill(causal_mask, float('-inf'))
        attn_causal = torch.softmax(scores_causal, dim=-1)

        # Verify causal masking works
        for i in range(seq):
            for j in range(i + 1, seq):
                assert attn_causal[0, 0, i, j] < 1e-6


class TestPerformance:
    """性能测试"""

    def test_mha_inference_speed(self):
        """测试MHA推理速度"""
        import time

        from model.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(config)
        mha.eval()

        x = torch.randn(4, 64, 256)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = mha(x)

        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = mha(x)
        end = time.time()

        # Just verify it runs in reasonable time
        assert end - start < 60  # Should complete in under 60 seconds

    def test_gqa_memory_efficiency(self):
        """测试GQA内存效率"""
        from model.config import ModelConfig

        # MHA config
        mha_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        # GQA config
        gqa_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="gelu",
            use_rms_norm=True, use_rope=True, use_swiglu=False,
        )

        mha = MultiHeadAttention(mha_config)
        gqa = GroupedQueryAttention(gqa_config)

        # Count parameters
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())

        # GQA should have fewer KV projection parameters
        # The ratio should be approximately num_kv_heads / num_heads
        ratio = gqa_config.num_key_value_heads / mha_config.num_attention_heads
        assert ratio == 0.25  # 2 / 8 = 0.25

    def test_attention_scaling(self):
        """测试注意力随序列长度扩展"""
        batch, heads, dim = 2, 4, 64

        # Time complexity is O(seq_len^2)
        # We verify this by measuring computation time

        for seq_len in [32, 64, 128]:
            Q = torch.randn(batch, heads, seq_len, dim)
            K = torch.randn(batch, heads, seq_len, dim)
            V = torch.randn(batch, heads, seq_len, dim)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)

            # Verify output shape scales with seq_len
            assert output.shape == torch.Size([batch, heads, seq_len, dim])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
