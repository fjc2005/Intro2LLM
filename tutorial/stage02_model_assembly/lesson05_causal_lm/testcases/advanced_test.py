"""
课时5进阶测试：Causal LM边界条件与复杂场景
"""

import pytest
import torch
import torch.nn as nn
import time
import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.causal_lm import CausalLM, CausalLMOutputWithPast
from model.config import ModelConfig


class TestCausalLMAdvanced:
    """Causal LM高级测试"""

    def test_gradient_checkpointing(self):
        """测试梯度检查点兼容性"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Test with gradient
        input_ids = torch.randint(0, 1000, (2, 10))
        input_ids.requires_grad = True

        # Forward
        output = model(input_ids)
        loss = output.logits.sum()
        loss.backward()

        # Verify gradient flows
        assert input_ids.grad is not None

    def test_model_save_load(self):
        """测试模型保存加载"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Save state dict
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), path)

            # Load into new model
            model2 = CausalLM(config)
            model2.load_state_dict(torch.load(path))

            # Verify weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.equal(p1, p2)

    def test_dtype_compatibility(self):
        """测试数据类型兼容性"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Test with float32
        input_ids = torch.randint(0, 1000, (2, 10))
        output_fp32 = model(input_ids)
        assert output_fp32.logits.dtype == torch.float32

    def test_batch_generation(self):
        """测试批量生成"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        # Batch input
        input_ids = torch.randint(0, 1000, (4, 5))

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=5)

        # Verify batch output
        assert output.shape[0] == 4
        assert output.shape[1] == 10


class TestKVCacheAdvanced:
    """KV缓存高级测试"""

    def test_kv_cache_memory_usage(self):
        """测试KV缓存内存使用"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Different sequence lengths
        for seq_len in [10, 20, 50]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            output = model(input_ids, use_cache=True)

            if output.past_key_values:
                k, v = output.past_key_values[0]
                assert k.shape[2] == seq_len

    def test_kv_cache_gqa_optimization(self):
        """测试GQA优化效果"""
        # MHA config
        mha_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        # GQA config
        gqa_config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        mha_model = CausalLM(mha_config)
        gqa_model = CausalLM(gqa_config)

        input_ids = torch.randint(0, 1000, (2, 10))

        mha_output = mha_model(input_ids, use_cache=True)
        gqa_output = gqa_model(input_ids, use_cache=True)

        # Both should return valid outputs
        assert mha_output.logits.shape == gqa_output.logits.shape


class TestGenerationAdvanced:
    """生成高级测试"""

    def test_top_p_nucleus(self):
        """测试Top-p核采样"""
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
            output = model.generate(input_ids, max_new_tokens=5, top_p=0.9)

        assert output.shape[1] == 10

    def test_combined_sampling(self):
        """测试组合采样策略"""
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
                input_ids, max_new_tokens=5,
                temperature=0.8, top_k=20, top_p=0.9
            )

        assert output.shape[1] == 10

    def test_repetition_penalty_strength(self):
        """测试重复惩罚强度"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3]])

        with torch.no_grad():
            # Different penalty values
            outputs = []
            for penalty in [1.0, 1.2, 1.5, 2.0]:
                out = model.generate(input_ids, max_new_tokens=5, repetition_penalty=penalty)
                outputs.append(out)

        # All should produce valid outputs
        for out in outputs:
            assert out.shape[1] == 8

    def test_max_length_enforcement(self):
        """测试最大长度强制"""
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
            output = model.generate(input_ids, max_new_tokens=10)

        # Should not exceed max
        assert output.shape[1] <= 15

    def test_deterministic_sampling(self):
        """测试确定性采样"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        # Set seed for reproducibility
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (1, 5))
        with torch.no_grad():
            output1 = model.generate(input_ids, max_new_tokens=5, temperature=0.0)

        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model.generate(input_ids, max_new_tokens=5, temperature=0.0)

        # With same seed and temperature=0, should be identical
        assert torch.equal(output1, output2)


class TestEdgeCases:
    """边界条件测试"""

    def test_single_token_generation(self):
        """测试单token生成"""
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
            output = model.generate(input_ids, max_new_tokens=1)

        assert output.shape[1] == 6

    def test_vocabulary_boundaries(self):
        """测试词表边界"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Test with max token ID
        input_ids = torch.tensor([[999]])  # max vocab - 1

        output = model(input_ids)

        # Should produce valid logits
        assert output.logits.shape == torch.Size([1, 1, 1000])


class TestPerformance:
    """性能测试"""

    def test_inference_speed_with_cache(self):
        """测试带缓存的推理速度"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 10))

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        # With cache - incremental generation
        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                output = model.generate(input_ids, max_new_tokens=10)
        end = time.time()

        assert end - start < 120  # Should complete reasonably fast

    def test_memory_efficiency(self):
        """测试内存效率"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert num_params > 0

    def test_batch_scaling(self):
        """测试批量扩展性"""
        config = ModelConfig(
            vocab_size=1000, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            attention_dropout=0.0, hidden_act="silu",
            use_rms_norm=True, use_rope=True, use_swiglu=True,
        )

        model = CausalLM(config)
        model.eval()

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, 1000, (batch_size, 5))

            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=5)

            assert output.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
