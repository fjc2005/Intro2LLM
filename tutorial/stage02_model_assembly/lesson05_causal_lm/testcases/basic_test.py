"""
课时5基础测试：Causal LM基础功能验证
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.causal_lm import CausalLM, CausalLMOutputWithPast
from model.config import ModelConfig


class TestCausalLM:
    """测试Causal LM基础功能"""

    def test_model_initialization(self):
        """测试模型正确初始化"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Verify submodules exist
        assert hasattr(model, 'embed_tokens')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'lm_head')

    def test_forward_output_shape(self):
        """测试前向传播输出形状"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Input
        input_ids = torch.randint(0, 1000, (2, 10))

        # Forward
        output = model(input_ids)

        # Verify logits shape
        assert output.logits.shape == torch.Size([2, 10, 1000])

    def test_lm_head_projection(self):
        """测试LM Head投影"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Verify lm_head weight shape
        assert model.lm_head.weight.shape == torch.Size([1000, 256])

    def test_weight_tying(self):
        """测试权重共享"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
            tie_word_embeddings=True,
        )

        model = CausalLM(config)

        # Verify weight tying (lm_head shares weights with embed_tokens)
        # Check that lm_head.weight is the same object as embed_tokens.embedding.weight
        assert model.lm_head.weight is model.get_input_embeddings().weight


class TestKVCache:
    """测试KV缓存基础功能"""

    def test_kv_cache_initialization(self):
        """测试KV缓存正确初始化"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Forward with cache
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids, use_cache=True)

        # Verify past_key_values is returned
        assert output.past_key_values is not None

    def test_kv_cache_shape(self):
        """测试KV缓存形状"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids, use_cache=True)

        # Verify each layer returns KV cache
        past_kv = output.past_key_values

        # Should have num_hidden_layers KV pairs
        assert len(past_kv) == 2

        # Each should be a tuple of (key, value)
        for layer_kv in past_kv:
            k, v = layer_kv
            # Shape: [batch, num_heads, seq_len, head_dim]
            assert k.shape[0] == 2  # batch
            assert k.shape[1] == 4  # num_heads
            assert k.shape[2] == 10  # seq_len

    def test_kv_cache_accumulation(self):
        """测试KV缓存累积"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # First pass
        input_ids_1 = torch.randint(0, 1000, (2, 5))
        output_1 = model(input_ids_1, use_cache=True)
        past_kv_1 = output_1.past_key_values

        # Second pass with cache
        input_ids_2 = torch.randint(0, 1000, (2, 1))
        output_2 = model(input_ids_2, past_key_values=past_kv_1, use_cache=True)
        past_kv_2 = output_2.past_key_values

        # Verify cache is accumulated
        k_2, v_2 = past_kv_2[0]
        assert k_2.shape[2] == 6  # 5 + 1


class TestGeneration:
    """测试生成功能"""

    def test_generate_output_shape(self):
        """测试生成输出形状"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        # Input
        input_ids = torch.randint(0, 1000, (2, 5))

        # Generate
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=10)

        # Verify output shape
        assert output_ids.shape[0] == 2  # batch
        assert output_ids.shape[1] == 15  # 5 + 10

    def test_greedy_generation(self):
        """测试贪心生成"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        # Same input should produce same output with temperature=0
        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            output1 = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
            output2 = model.generate(input_ids, max_new_tokens=5, temperature=0.0)

        # Should be identical with greedy
        assert torch.equal(output1, output2)

    def test_temperature_effect(self):
        """测试温度效果"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            # High temperature = more random
            output_high = model.generate(input_ids, max_new_tokens=5, temperature=2.0)
            # Low temperature = more deterministic
            output_low = model.generate(input_ids, max_new_tokens=5, temperature=0.1)

        # Both should produce valid outputs
        assert output_high.shape[1] == 10
        assert output_low.shape[1] == 10

    def test_top_k_filtering(self):
        """测试Top-k过滤"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=5, top_k=10)

        assert output.shape[1] == 10

    def test_eos_stopping(self):
        """测试EOS停止"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=20, eos_token_id=2, pad_token_id=0
            )

        # Should generate up to max_new_tokens or stop at EOS
        assert output.shape[1] <= 25  # 5 + 20


class TestRepetitionPenalty:
    """测试重复惩罚"""

    def test_repetition_penalty_applied(self):
        """测试重复惩罚应用"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        # Create input that will definitely repeat
        input_ids = torch.tensor([[1, 2, 3]])

        with torch.no_grad():
            output_no_penalty = model.generate(
                input_ids, max_new_tokens=5, repetition_penalty=1.0
            )
            output_with_penalty = model.generate(
                input_ids, max_new_tokens=5, repetition_penalty=1.5
            )

        # Both should generate
        assert output_no_penalty.shape[1] == 8
        assert output_with_penalty.shape[1] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
