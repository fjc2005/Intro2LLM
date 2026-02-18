"""
课时1基础测试：Config和Tokenizer基础功能验证
"""

import pytest
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple


class TestModelConfig:
    """测试ModelConfig配置类"""

    def test_config_initialization(self):
        """测试配置对象能正确初始化"""
        # TODO: 创建ModelConfig实例
        # config = ModelConfig(vocab_size=32000, hidden_size=4096, ...)
        # 验证各属性正确设置
        pass

    def test_head_dim_calculation(self):
        """测试head_dim自动计算"""
        # TODO: 验证 head_dim = hidden_size / num_attention_heads
        # hidden_size=4096, num_attention_heads=32 → head_dim=128
        pass

    def test_gqa_configuration(self):
        """测试GQA配置"""
        # TODO: 创建GQA配置
        # num_attention_heads=32, num_key_value_heads=8
        # 验证 num_attention_heads % num_key_value_heads == 0
        pass

    def test_config_serialization(self):
        """测试配置序列化与反序列化"""
        # TODO: config.to_dict() 然后 ModelConfig.from_dict()
        # 验证恢复后的配置与原配置相同
        pass


class TestBPETokenizer:
    """测试BPE分词器基础功能"""

    def test_tokenizer_initialization(self):
        """测试分词器能正确初始化"""
        # TODO: 创建BPETokenizer实例
        # 验证基础词表包含字节字符或特殊token
        pass

    def test_simple_encode_decode(self):
        """测试简单文本的编解码"""
        # TODO: 训练一个简单的BPE词表
        # 对"hello world"进行编码再解码
        # 验证解码结果与原始文本一致
        pass

    def test_vocab_size_limit(self):
        """测试词表大小限制"""
        # TODO: 使用vocab_size=100训练
        # 验证最终词表大小不超过100
        pass

    def test_special_tokens(self):
        """测试特殊token处理"""
        # TODO: 验证<pad>, <s>, </s>等特殊token存在
        # 验证它们的ID分配
        pass

    def test_oov_handling(self):
        """测试未登录词处理"""
        # TODO: 训练时不包含某些字符
        # 验证编码时能通过子词分解处理OOV
        pass


class TestByteLevelTokenizer:
    """测试字节级分词器"""

    def test_byte_encoding(self):
        """测试字节级编码"""
        # TODO: 验证任何Unicode文本都能被编码
        # 包括emoji、中文、特殊符号等
        pass

    def test_byte_decode_consistency(self):
        """测试字节编解码一致性"""
        # TODO: 对随机文本编码再解码
        # 验证结果与原始文本一致
        pass

    def test_no_unk_token(self):
        """测试字节级分词不会出现UNK"""
        # TODO: 验证所有字节值(0-255)都在基础词表中
        # 因此不会出现unknown token
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
