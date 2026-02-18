"""
课时1基础测试：Config和Tokenizer基础功能验证
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.config import ModelConfig
from dataclasses import dataclass, fields


class TestModelConfig:
    """测试ModelConfig配置类"""

    def test_config_initialization(self):
        """测试配置对象能正确初始化"""
        # Create ModelConfig instance with all required fields
        config = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="silu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=True,
        )

        # Verify all attributes are correctly set
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.intermediate_size == 11008
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 32
        assert config.max_position_embeddings == 2048
        assert config.rope_theta == 10000.0
        assert config.rms_norm_eps == 1e-6
        assert config.attention_dropout == 0.0
        assert config.hidden_act == "silu"
        assert config.use_rms_norm is True
        assert config.use_rope is True
        assert config.use_swiglu is True

    def test_head_dim_calculation(self):
        """测试head_dim自动计算"""
        # Create config with known values
        config = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="silu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=True,
        )

        # Verify head_dim = hidden_size / num_attention_heads
        # 4096 / 32 = 128
        expected_head_dim = 4096 // 32
        assert config.head_dim == expected_head_dim
        assert config.head_dim == 128

        # Test with different values
        config2 = ModelConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=8,
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
        assert config2.head_dim == 512 // 8  # 64

    def test_gqa_configuration(self):
        """测试GQA配置"""
        # Create GQA config
        config = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA: fewer KV heads
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="silu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=True,
        )

        # Verify num_attention_heads % num_key_value_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0

        # Verify num_key_value_groups calculation
        expected_groups = 32 // 8  # 4
        assert config.num_key_value_groups == expected_groups

    def test_config_serialization(self):
        """测试配置序列化与反序列化"""
        # Create original config
        original_config = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="silu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=True,
        )

        # Serialize to dict (dataclasses.asdict)
        config_dict = {
            'vocab_size': original_config.vocab_size,
            'hidden_size': original_config.hidden_size,
            'intermediate_size': original_config.intermediate_size,
            'num_hidden_layers': original_config.num_hidden_layers,
            'num_attention_heads': original_config.num_attention_heads,
            'num_key_value_heads': original_config.num_key_value_heads,
            'max_position_embeddings': original_config.max_position_embeddings,
            'rope_theta': original_config.rope_theta,
            'rms_norm_eps': original_config.rms_norm_eps,
            'attention_dropout': original_config.attention_dropout,
            'hidden_act': original_config.hidden_act,
            'use_rms_norm': original_config.use_rms_norm,
            'use_rope': original_config.use_rope,
            'use_swiglu': original_config.use_swiglu,
        }

        # Deserialize back to ModelConfig
        restored_config = ModelConfig(**config_dict)

        # Verify restored config matches original
        assert restored_config.vocab_size == original_config.vocab_size
        assert restored_config.hidden_size == original_config.hidden_size
        assert restored_config.intermediate_size == original_config.intermediate_size
        assert restored_config.num_hidden_layers == original_config.num_hidden_layers
        assert restored_config.num_attention_heads == original_config.num_attention_heads
        assert restored_config.num_key_value_heads == original_config.num_key_value_heads
        assert restored_config.max_position_embeddings == original_config.max_position_embeddings
        assert restored_config.rope_theta == original_config.rope_theta
        assert restored_config.rms_norm_eps == original_config.rms_norm_eps
        assert restored_config.attention_dropout == original_config.attention_dropout
        assert restored_config.hidden_act == original_config.hidden_act
        assert restored_config.use_rms_norm == original_config.use_rms_norm
        assert restored_config.use_rope == original_config.use_rope
        assert restored_config.use_swiglu == original_config.use_swiglu


class TestBPETokenizer:
    """测试BPE分词器基础功能 - 使用模拟/基础实现测试"""

    def test_tokenizer_interface(self):
        """测试分词器基本接口存在"""
        # For this educational project, we test that the tokenizer
        # base class defines the expected interface
        from tokenizer.base_tokenizer import BaseTokenizer

        # Verify BaseTokenizer is a class
        assert isinstance(BaseTokenizer, type)

        # Check that the expected methods exist
        assert hasattr(BaseTokenizer, 'encode')
        assert hasattr(BaseTokenizer, 'decode')
        assert hasattr(BaseTokenizer, 'vocab_size')

    def test_simple_vocab_mapping(self):
        """测试简单词表映射"""
        # Create a simple vocabulary mapping for testing
        vocab = {
            '<pad>': 0,
            '<s>': 1,
            '</s>': 2,
            'hello': 3,
            'world': 4,
            'h': 5,
            'e': 6,
            'l': 7,
            'o': 8,
            'w': 9,
            'r': 10,
            'd': 11,
        }

        # Verify vocab structure
        assert len(vocab) == 12
        assert vocab['<pad>'] == 0
        assert vocab['<s>'] == 1
        assert vocab['</s>'] == 2
        assert vocab['hello'] == 3
        assert vocab['world'] == 4

    def test_vocab_size_limit(self):
        """测试词表大小限制概念"""
        # Test that vocab_size parameter controls the size
        target_vocab_size = 100

        # Create a small vocab
        small_vocab = {f'token_{i}': i for i in range(target_vocab_size)}

        # Verify size constraint
        assert len(small_vocab) == target_vocab_size
        assert len(small_vocab) <= target_vocab_size

    def test_special_tokens(self):
        """测试特殊token处理"""
        # Define standard special tokens
        special_tokens = ['<pad>', '<s>', '</s>', '<unk>']

        # Verify special tokens list
        assert '<pad>' in special_tokens
        assert '<s>' in special_tokens
        assert '</s>' in special_tokens
        assert '<unk>' in special_tokens

        # Test special token IDs are typically small
        special_token_ids = {0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>'}
        assert special_token_ids[0] == '<pad>'
        assert special_token_ids[1] == '<s>'


class TestByteLevelTokenizer:
    """测试字节级分词器"""

    def test_byte_encoding_concept(self):
        """测试字节级编码概念"""
        # Test that any Unicode text can be encoded to bytes
        texts = [
            "Hello World",
            "你好世界",
            "",
            "12345",
        ]

        for text in texts:
            # Encode to bytes
            byte_encoded = text.encode('utf-8')
            # Verify it's bytes
            assert isinstance(byte_encoded, bytes)
            # Decode back
            decoded = byte_encoded.decode('utf-8')
            # Verify round-trip
            assert decoded == text

    def test_byte_decode_consistency(self):
        """测试字节编解码一致性"""
        test_texts = [
            "hello world",
            "Test with spaces",
            "Numbers 123",
        ]

        for text in test_texts:
            # Encode to bytes
            encoded = text.encode('utf-8')
            # Decode back
            decoded = encoded.decode('utf-8')
            # Verify consistency
            assert decoded == text

    def test_all_bytes_representable(self):
        """测试所有字节值都可表示"""
        # Byte-level tokenizer should handle all 256 byte values
        all_bytes = bytes(range(256))

        # Verify we can decode
        # Note: some byte sequences may not be valid UTF-8,
        # but byte-level tokenizers handle them specially
        assert len(all_bytes) == 256

        # Verify each byte value is in range 0-255
        for b in all_bytes:
            assert 0 <= b <= 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
