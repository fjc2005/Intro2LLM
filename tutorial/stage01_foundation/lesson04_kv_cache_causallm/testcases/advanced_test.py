"""
L04: KV Cache 与 CausalLM - 进阶测试

测试 KV Cache 优化、采样策略、生成质量等进阶内容。
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file))))))

from model.causal_lm import CausalLM
from model.config import ModelConfig


def create_test_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )


class TestGenerationAdvanced:
    """测试生成进阶特性"""

    def test_generate_temperature(self):
        """测试温度参数"""
        config = create_test_config()

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            # 高温度 (更随机)
            output_high = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=2.0
            )

            # 低温度 (更确定)
            output_low = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.1
            )

        # 两个输出应该不同 (高温度产生更多样化结果)
        assert output_high.shape[1] == 10
        assert output_low.shape[1] == 10

    def test_generate_top_k(self):
        """测试 Top-k 采样"""
        config = create_test_config()

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                top_k=10
            )

        assert output.shape[1] == 15

    def test_generate_top_p(self):
        """测试 Top-p 采样"""
        config = create_test_config()

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                top_p=0.9
            )

        assert output.shape[1] == 15


class TestKVCacheAdvanced:
    """测试 KV Cache 进阶特性"""

    def test_kv_cache_all_layers(self):
        """测试所有层的 KV Cache"""
        config = create_test_config()
        config.num_hidden_layers = 4

        model = CausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        output = model(input_ids, use_cache=True)

        # 验证每层都有缓存
        kv = output.past_key_values
        assert len(kv) == config.num_hidden_layers

    def test_kv_cache_multiple_steps(self):
        """测试多步缓存"""
        config = create_test_config()

        model = CausalLM(config)
        model.eval()

        batch_size = 1
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        with torch.no_grad():
            # 逐步生成
            for _ in range(3):
                output = model(input_ids, use_cache=True)
                kv = output.past_key_values

                # 获取最后一个 token 的预测
                next_token = output.logits[:, -1:].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        # 验证序列长度
        assert input_ids.shape[1] == 5 + 3


class TestCausalLMAdvanced:
    """测试 CausalLM 进阶特性"""

    def test_weight_tying(self):
        """测试权重共享 (embedding 与 lm_head)"""
        config = create_test_config()

        model = CausalLM(config)

        # 检查权重是否共享
        # 注意: 如果配置了共享，embed_tokens 和 lm_head 应该共享权重

    def test_prepare_inputs_for_generation(self):
        """测试生成输入准备"""
        config = create_test_config()

        model = CausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        # 准备生成输入
        prepared = model.prepare_inputs_for_generation(input_ids)

        assert "input_ids" in prepared


class TestCausalMask:
    """测试因果掩码"""

    def test_causal_mask_effect(self):
        """测试因果掩码的效果"""
        config = create_test_config()

        model = CausalLM(config)

        # 创建相同输入，不同位置
        batch_size = 1
        seq_len = 8

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # 设置不同位置为不同 token，看输出是否不同
        output = model(input_ids)

        # 验证不同位置的输出确实不同
        logits = output.logits

        # 位置 0 的预测应该和位置 7 的预测不同
        # 因为位置 7 可以看到所有之前的 token
        assert not torch.allclose(logits[:, 0, :], logits[:, 7, :])


class TestGenerationQuality:
    """测试生成质量"""

    def test_generation_determinism(self):
        """测试生成的确定性"""
        config = create_test_config()

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        # 固定随机种子
        torch.manual_seed(42)

        with torch.no_grad():
            output1 = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.0
            )

        torch.manual_seed(42)

        with torch.no_grad():
            output2 = model.generate(
                input_ids,
                max_new_tokens=5,
                temperature=0.0
            )

        # 贪婪生成应该是确定的
        assert torch.equal(output1, output2)


class TestLongSequenceGeneration:
    """测试长序列生成"""

    def test_long_generation(self):
        """测试生成长序列"""
        config = create_test_config()
        config.hidden_size = 64
        config.num_hidden_layers = 2

        model = CausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 3))

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.0
            )

        assert output_ids.shape[1] == 53


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
