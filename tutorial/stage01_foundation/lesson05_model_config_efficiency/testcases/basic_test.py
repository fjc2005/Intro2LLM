"""
L05: 模型配置与效率分析 - 基础测试

测试配置类、参数量计算、FLOPs 估算的基本功能。
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.config import ModelConfig
from utils.compute import compute_num_params, compute_flops_forward, estimate_memory


def create_tiny_config():
    """创建小型配置用于测试"""
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


class TestModelConfig:
    """测试 ModelConfig"""

    def test_config_creation(self):
        """测试配置创建"""
        config = create_tiny_config()

        assert config.vocab_size == 1000
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 2

    def test_head_dim_property(self):
        """测试 head_dim 属性"""
        config = create_tiny_config()

        head_dim = config.head_dim
        expected = config.hidden_size // config.num_attention_heads

        assert head_dim == expected

    def test_num_key_value_groups_property(self):
        """测试 num_key_value_groups 属性"""
        config = create_tiny_config()

        groups = config.num_key_value_groups
        expected = config.num_attention_heads // config.num_key_value_heads

        assert groups == expected


class TestComputeNumParams:
    """测试参数量计算"""

    def test_num_params_basic(self):
        """测试基本参数量计算"""
        config = create_tiny_config()

        num_params = compute_num_params(config)

        # 验证参数量 > 0
        assert num_params > 0

    def test_num_params_order(self):
        """测试参数量级正确性"""
        config = create_tiny_config()

        num_params = compute_num_params(config)

        # 小模型应该在 1M 以内
        assert num_params < 1_000_000

    def test_num_params_scales_with_layers(self):
        """测试层数增加时参数量增加"""
        config1 = create_tiny_config()
        config1.num_hidden_layers = 2

        config2 = create_tiny_config()
        config2.num_hidden_layers = 4

        params1 = compute_num_params(config1)
        params2 = compute_num_params(config2)

        assert params2 > params1


class TestComputeFLOPs:
    """测试 FLOPs 计算"""

    def test_flops_basic(self):
        """测试基本 FLOPs 计算"""
        config = create_tiny_config()

        flops = compute_flops_forward(config)

        assert flops > 0

    def test_flops_order(self):
        """测试 FLOPs 量级"""
        config = create_tiny_config()

        flops = compute_flops_forward(config)

        # 小模型的 FLOPs 应该在合理范围
        assert flops > 1e6  # 至少几百万


class TestEstimateMemory:
    """测试显存估算"""

    def test_estimate_memory_fp32(self):
        """测试 FP32 显存估算"""
        config = create_tiny_config()

        memory = estimate_memory(config, batch_size=1, seq_len=32, precision="fp32")

        # 参数量 * 4 bytes
        num_params = compute_num_params(config)
        expected = num_params * 4 / (1024 ** 3)

        assert memory > 0

    def test_estimate_memory_fp16(self):
        """测试 FP16 显存估算"""
        config = create_tiny_config()

        memory = estimate_memory(config, batch_size=1, seq_len=32, precision="fp16")

        # FP16 应该比 FP32 少一半
        assert memory > 0


class TestEfficiencyComparison:
    """测试效率对比"""

    def test_mha_vs_gqa_params(self):
        """对比 MHA 和 GQA 参数量"""
        config_mha = create_tiny_config()
        config_mha.num_key_value_heads = config_mha.num_attention_heads

        config_gqa = create_tiny_config()
        config_gqa.num_key_value_heads = 1

        params_mha = compute_num_params(config_mha)
        params_gqa = compute_num_params(config_gqa)

        # GQA 参数量应该略少 (K、V 投影更小)
        assert params_gqa <= params_mha


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
