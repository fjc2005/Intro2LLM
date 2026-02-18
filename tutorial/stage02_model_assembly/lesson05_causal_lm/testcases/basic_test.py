"""
课时5基础测试：Causal LM基础功能验证
"""

import pytest
import torch
import torch.nn as nn


class TestCausalLM:
    """测试Causal LM基础功能"""

    def test_model_initialization(self):
        """测试模型正确初始化"""
        # TODO: 使用tiny_config创建CausalLM
        # 验证包含: embed_tokens, layers, norm, lm_head
        pass

    def test_forward_output_shape(self):
        """测试前向传播输出形状"""
        # TODO: 输入input_ids = torch.randint(0, 1000, (2, 10))
        # 验证logits形状为[2, 10, vocab_size]
        pass

    def test_lm_head_projection(self):
        """测试LM Head投影"""
        # TODO: 验证hidden_size -> vocab_size的投影
        pass

    def test_weight_tying(self):
        """测试权重共享"""
        # TODO: 当tie_word_embeddings=True时
        # 验证lm_head.weight与embed_tokens.weight相同
        pass


class TestKVCache:
    """测试KV缓存基础功能"""

    def test_kv_cache_initialization(self):
        """测试KV缓存正确初始化"""
        # TODO: use_cache=True时验证返回past_key_values
        pass

    def test_kv_cache_shape(self):
        """测试KV缓存形状"""
        # TODO: 验证每层缓存包含(key, value)元组
        # 验证形状为[batch, num_heads, seq_len, head_dim]
        pass

    def test_kv_cache_accumulation(self):
        """测试KV缓存累积"""
        # TODO: 第一次前向seq_len=5，第二次seq_len=1
        # 验证缓存长度从5变为6
        pass


class TestGeneration:
    """测试生成功能"""

    def test_generate_output_shape(self):
        """测试生成输出形状"""
        # TODO: 输入seq_len=5，max_new_tokens=10
        # 验证输出seq_len=15
        pass

    def test_greedy_generation(self):
        """测试贪心生成"""
        # TODO: temperature=0等效于argmax
        # 验证确定性的输出
        pass

    def test_temperature_effect(self):
        """测试温度效果"""
        # TODO: 对比temperature=0.1和2.0的分布
        # 低温更尖锐，高温更平缓
        pass

    def test_top_k_filtering(self):
        """测试Top-k过滤"""
        # TODO: 设置top_k=10
        # 验证只有top 10 logits非-inf
        pass

    def test_eos_stopping(self):
        """测试EOS停止"""
        # TODO: 设置eos_token_id
        # 验证生成EOS后停止
        pass


class TestRepetitionPenalty:
    """测试重复惩罚"""

    def test_repetition_penalty_applied(self):
        """测试重复惩罚应用"""
        # TODO: 已生成token的logits被修改
        pass

    def test_repetition_penalty_direction(self):
        """测试重复惩罚方向"""
        # TODO: 正logits减小，负logits增大(绝对值)
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
