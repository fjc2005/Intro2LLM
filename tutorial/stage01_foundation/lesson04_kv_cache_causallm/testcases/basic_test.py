"""
L04: KV Cache 与 CausalLM - 基础测试

测试 CausalLM、KV Cache、生成功能的基本正确性。
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.causal_lm import CausalLM, CausalLMOutputWithPast
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


class TestCausalLM:
    """测试 CausalLM 基本功能"""

    def test_causal_lm_initialization(self):
        """测试 CausalLM 初始化"""
        config = create_test_config()

        model = CausalLM(config)

        assert model.embed_tokens is not None
        assert len(model.layers) == config.num_hidden_layers
        assert model.norm is not None
        assert model.lm_head is not None

    def test_causal_lm_forward(self):
        """测试 CausalLM 前向传播"""
        config = create_test_config()

        model = CausalLM(config)

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        # 检查输出形状
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_causal_lm_with_labels(self):
        """测试带标签的前向传播"""
        config = create_test_config()

        model = CausalLM(config)

        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids, labels=labels)

        assert output.loss is not None


class TestCausalLMOutput:
    """测试 CausalLM 输出容器"""

    def test_output_container(self):
        """测试输出容器"""
        logits = torch.randn(2, 10, 1000)
        loss = torch.tensor(1.5)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None
        )

        assert output.loss == loss
        assert output.logits.shape == logits.shape


class TestKVCache:
    """测试 KV Cache"""

    def test_kv_cache_basic(self):
        """测试基本的 KV Cache"""
        config = create_test_config()

        model = CausalLM(config)

        batch_size = 1
        seq_len = 8

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # 首次前向，启用缓存
        output1 = model(input_ids, use_cache=True)
        kv1 = output1.past_key_values

        # 第二次前向
        next_token = torch.randint(0, config.vocab_size, (batch_size, 1))
        output2 = model(next_token, past_key_values=kv1, use_cache=True)

        assert output2.logits.shape == (batch_size, 1, config.vocab_size)

    def test_kv_cache_length(self):
        """测试缓存长度增加"""
        config = create_test_config()

        model = CausalLM(config)

        batch_size = 1

        # 初始序列
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))
        output = model(input_ids, use_cache=True)
        kv = output.past_key_values

        # 验证缓存长度
        cache_len = kv[0][0].shape[2]  # key 的序列长度
        assert cache_len == 5


class TestGeneration:
    """测试文本生成"""

    def test_generate_greedy(self):
        """测试贪婪生成"""
        config = create_test_config()

        model = CausalLM(config)
        model.eval()

        batch_size = 1
        prompt_len = 5

        input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.0  # 贪婪
            )

        # 验证生成长度
        assert output_ids.shape[1] == prompt_len + 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
