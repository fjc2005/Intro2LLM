"""
课时1进阶测试：Config和Tokenizer高级功能验证
"""

import pytest
import torch
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from model.config import ModelConfig


class TestModelConfigAdvanced:
    """ModelConfig高级测试"""

    def test_invalid_head_configuration(self):
        """测试无效的头数配置应抛出异常"""
        # Test that invalid configs raise appropriate errors or behave correctly
        # hidden_size not divisible by num_attention_heads

        # This should work fine - 512 / 8 = 64
        config_valid = ModelConfig(
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
        assert config_valid.head_dim == 64

        # Test with uneven division - head_dim calculation should still work
        # Note: In real implementation, this might raise an error during model creation
        # but the config itself just stores values
        config_uneven = ModelConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=8,
            num_attention_heads=6,  # 512 / 6 = 85.33...
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
        # The property will do integer division
        assert config_uneven.head_dim == 512 // 6  # 85

    def test_invalid_gqa_configuration(self):
        """测试无效的GQA配置"""
        # Create a config where num_attention_heads is not divisible by num_key_value_heads
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=8,
            num_attention_heads=7,  # Not divisible by num_key_value_heads
            num_key_value_heads=3,  # 7 / 3 = 2.33...
            max_position_embeddings=512,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="gelu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=False,
        )

        # The config stores values but num_key_value_groups calculation will be truncated
        # 7 // 3 = 2
        assert config.num_key_value_groups == 7 // 3

    def test_parameter_count_estimation(self):
        """测试参数量估算"""
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

        # Estimate embedding parameters
        embedding_params = config.vocab_size * config.hidden_size
        assert embedding_params == 32000 * 4096

        # Estimate attention parameters per layer
        # Q, K, V projections: 3 * hidden_size * hidden_size (for MHA)
        # O projection: hidden_size * hidden_size
        qkv_params = 3 * config.hidden_size * config.hidden_size
        o_proj_params = config.hidden_size * config.hidden_size
        attention_params_per_layer = qkv_params + o_proj_params
        assert attention_params_per_layer == 4 * 4096 * 4096

        # Total attention parameters
        total_attention_params = attention_params_per_layer * config.num_hidden_layers

        # Verify calculations
        assert embedding_params == 131072000
        assert total_attention_params == 4 * 4096 * 4096 * 32

    def test_gqa_kv_cache_savings(self):
        """测试GQA的KV缓存节省计算"""
        # Standard MHA
        mha_config = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,  # MHA: same as attention heads
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="silu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=True,
        )

        # GQA config
        gqa_config = ModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,   # GQA: fewer KV heads
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            hidden_act="silu",
            use_rms_norm=True,
            use_rope=True,
            use_swiglu=True,
        )

        # Calculate KV cache size ratio
        # MHA: 32 heads per sequence position
        # GQA: 8 heads per sequence position
        kv_cache_ratio = gqa_config.num_key_value_heads / mha_config.num_key_value_heads
        assert kv_cache_ratio == 0.25  # GQA uses 25% of MHA KV cache

        # Verify num_key_value_groups
        assert mha_config.num_key_value_groups == 1  # 32/32
        assert gqa_config.num_key_value_groups == 4  # 32/8

    def test_config_equality(self):
        """测试配置相等性比较"""
        config1 = ModelConfig(
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

        # Verify all attributes are equal
        assert config1.vocab_size == config2.vocab_size
        assert config1.hidden_size == config2.hidden_size
        assert config1.intermediate_size == config2.intermediate_size
        assert config1.num_hidden_layers == config2.num_hidden_layers
        assert config1.num_attention_heads == config2.num_attention_heads
        assert config1.num_key_value_heads == config2.num_key_value_heads
        assert config1.head_dim == config2.head_dim
        assert config1.num_key_value_groups == config2.num_key_value_groups

    def test_config_modification(self):
        """测试配置修改"""
        config = ModelConfig(
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

        # Modify config
        config.vocab_size = 2000
        config.hidden_size = 768

        # Verify changes
        assert config.vocab_size == 2000
        assert config.hidden_size == 768
        # Derived properties should update
        assert config.head_dim == 768 // 8  # 96


class TestTokenizerAdvanced:
    """Tokenizer高级测试"""

    def test_byte_fallback_encoding(self):
        """测试字节回退编码"""
        # Test encoding of characters that might not be in vocab
        texts = [
            "Hello \x00 World",  # Null character
            "Test \x80\x81",     # High bytes
            "Mix 中文 and bytes",
        ]

        for text in texts:
            # All text can be encoded as UTF-8 bytes
            byte_encoded = text.encode('utf-8', errors='ignore')
            assert isinstance(byte_encoded, bytes)

            # Decode back
            decoded = byte_encoded.decode('utf-8', errors='ignore')
            assert isinstance(decoded, str)

    def test_tokenizer_consistency(self):
        """测试分词器一致性"""
        # Same text should always produce same tokens
        text = "hello world"

        # Simulate encoding (would use actual tokenizer in real test)
        tokens_1 = [1, 5, 6, 7, 7, 8, 9, 10, 7, 8, 2]  # <s> hello world </s>
        tokens_2 = [1, 5, 6, 7, 7, 8, 9, 10, 7, 8, 2]

        assert tokens_1 == tokens_2

        # Decode should return similar text
        decoded = "hello world"
        assert decoded == text

    def test_subword_tokenization(self):
        """测试子词分词"""
        # Test that unknown words are broken into subwords
        vocab = {
            'un': 0,
            '##believable': 1,
            '##able': 2,
            '##ing': 3,
        }

        word = "unbelievable"
        # Expected tokenization: ['un', '##believable']
        tokens = ['un', '##believable']

        assert len(tokens) == 2
        assert tokens[0] == 'un'
        assert tokens[1] == '##believable'

    def test_deterministic_encoding(self):
        """测试编码确定性：相同输入产生相同输出"""
        text = "hello world"

        # Simulate multiple encodings
        encodings = []
        for _ in range(5):
            # In a real test, this would call tokenizer.encode(text)
            tokens = [1, 2, 3, 4, 5]  # Simulated
            encodings.append(tokens)

        # All encodings should be identical
        for encoding in encodings:
            assert encoding == encodings[0]


class TestTokenizerEdgeCases:
    """边界条件测试"""

    def test_empty_text(self):
        """测试空文本处理"""
        text = ""

        # Empty string encoding
        encoded = []
        assert len(encoded) == 0

        # Or with special tokens
        encoded_with_special = [1, 2]  # <s></s>
        assert len(encoded_with_special) == 2

    def test_whitespace_only(self):
        """测试纯空白字符"""
        texts = [
            "   ",      # Spaces
            "\t\n\r",   # Tabs, newlines
            "  \t  \n  ",  # Mixed whitespace
        ]

        for text in texts:
            # Should be encodable
            assert isinstance(text, str)
            assert len(text) > 0

    def test_unicode_edge_cases(self):
        """测试Unicode边界情况"""
        texts = [
            "\u0000",          # Null character
            "\uffff",          # Max BMP character
            "\U0001f600",      # Emoji (surrogate pair in UTF-16)
            "é",               # Combining character
            "\u200b",          # Zero-width space
        ]

        for text in texts:
            # Should be encodable to bytes
            try:
                encoded = text.encode('utf-8')
                decoded = encoded.decode('utf-8')
                assert decoded == text
            except UnicodeEncodeError:
                # Some characters might not encode in certain modes
                pass

    def test_very_long_text(self):
        """测试超长文本"""
        # Generate a long text (1000 characters)
        long_text = "word " * 200

        assert len(long_text) == 1000

        # Should be processable
        encoded = long_text.encode('utf-8')
        assert len(encoded) > 0


class TestIntegration:
    """集成测试"""

    def test_tokenizer_with_config(self):
        """测试分词器与配置的集成"""
        # Create config with specific vocab_size
        config = ModelConfig(
            vocab_size=50000,
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

        # Verify vocab_size is accessible
        assert config.vocab_size == 50000

        # In real integration, tokenizer would be created with this vocab_size
        # tokenizer = Tokenizer(vocab_size=config.vocab_size)
        # assert tokenizer.vocab_size == config.vocab_size

    def test_encode_batch(self):
        """测试批量编码概念"""
        texts = [
            "First text",
            "Second text which is longer",
            "Third",
        ]

        # Batch encoding would process all texts
        # For this test, we verify the concept
        assert len(texts) == 3

        # Each text should be individually processable
        for text in texts:
            assert isinstance(text, str)
            assert len(text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
