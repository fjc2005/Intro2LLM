"""
L05: 模型配置与效率分析 - 进阶测试

测试显存估算、KV Cache 显存、性能对比等进阶内容。
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.config import ModelConfig
from utils.compute import compute_num_params, compute_flops_forward, estimate_memory, estimate_kv_cache_memory


def create_tiny_config():
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


class TestComputeAdvanced:
    """测试进阶计算"""

    def test_large_model_params(self):
        """测试大模型参数量"""
        # 模拟 7B 模型配置
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

        num_params = compute_num_params(config)

        # 7B 模型应该约 7B 参数
        assert 6e9 < num_params < 8e9

    def test_flops_scales_with_layers(self):
        """测试 FLOPs 随层数增加"""
        config1 = create_tiny_config()
        config1.num_hidden_layers = 1

        config2 = create_tiny_config()
        config2.num_hidden_layers = 8

        flops1 = compute_flops_forward(config1)
        flops2 = compute_flops_forward(config2)

        assert flops2 > flops1 * 7


class TestMemoryAdvanced:
    """测试显存估算进阶"""

    def test_kv_cache_memory(self):
        """测试 KV Cache 显存"""
        config = create_tiny_config()

        batch_size = 4
        seq_len = 128

        memory = estimate_kv_cache_memory(config, batch_size, seq_len)

        assert memory > 0

    def test_training_vs_inference_memory(self):
        """测试训练 vs 推理显存"""
        config = create_tiny_config()

        # 推理只需要模型参数 + KV Cache
        mem_inference = estimate_memory(config, batch_size=1, seq_len=64, precision="fp16")

        # 训练需要额外空间 (梯度、优化器状态)
        # 简化估算: 训练显存 ≈ 4x 推理显存 (梯度 + Adam 状态)
        assert mem_inference > 0

    def test_batch_size_scaling(self):
        """测试 batch size 对显存的影响"""
        config = create_tiny_config()

        mem_bs1 = estimate_memory(config, batch_size=1, seq_len=64, precision="fp16")
        mem_bs4 = estimate_memory(config, batch_size=4, seq_len=64, precision="fp16")

        # 更大的 batch 应该需要更多显存 (主要是 KV Cache)
        # 注意: 纯模型参数不变，所以差异主要来自 KV Cache


class TestEfficiencyAdvanced:
    """测试效率进阶对比"""

    def test_gqa_kv_cache_savings(self):
        """测试 GQA 节省 KV Cache"""
        # MHA 配置
        config_mha = create_tiny_config()
        config_mha.num_key_value_heads = config_mha.num_attention_heads

        # GQA 配置
        config_gqa = create_tiny_config()
        config_gqa.num_key_value_heads = 1

        # MHA KV Cache
        kv_mha = estimate_kv_cache_memory(config_mha, batch_size=1, seq_len=1024)
        kv_gqa = estimate_kv_cache_memory(config_gqa, batch_size=1, seq_len=1024)

        # GQA 应该显著减少 KV Cache
        assert kv_gqa < kv_mha

    def test_sequence_length_scaling(self):
        """测试序列长度对显存的影响"""
        config = create_tiny_config()

        mem_128 = estimate_memory(config, batch_size=1, seq_len=128, precision="fp16")
        mem_512 = estimate_memory(config, batch_size=1, seq_len=512, precision="fp16")
        mem_2048 = estimate_memory(config, batch_size=1, seq_len=2048, precision="fp16")

        # 更长的序列需要更多显存
        assert mem_512 > mem_128
        assert mem_2048 > mem_512


class TestRealWorldScenarios:
    """测试实际场景"""

    def test_7b_model_memory(self):
        """测试 7B 模型显存需求"""
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

        # FP16 推理显存
        memory = estimate_memory(config, batch_size=1, seq_len=2048, precision="fp16")

        # 7B 模型 FP16 约需 14GB
        # 加上 KV Cache，应该约 16-20GB
        assert 10 < memory < 30  # GB

    def test_tiny_model_fits_cpu(self):
        """测试小模型是否可以在 CPU 上运行"""
        config = create_tiny_config()

        memory = estimate_memory(config, batch_size=1, seq_len=128, precision="fp32")

        # 小模型应该可以在 CPU 内存中放下
        # 约几 MB
        assert memory < 1  # GB


class TestPrecisionComparison:
    """测试不同精度对比"""

    def test_fp32_vs_fp16(self):
        """测试 FP32 vs FP16"""
        config = create_tiny_config()

        mem_fp32 = estimate_memory(config, batch_size=1, seq_len=64, precision="fp32")
        mem_fp16 = estimate_memory(config, batch_size=1, seq_len=64, precision="fp16")

        # FP16 应该约为 FP32 的一半
        assert abs(mem_fp32 / mem_fp16 - 2) < 0.1

    def test_bf16_vs_fp16(self):
        """测试 BF16 vs FP16"""
        config = create_tiny_config()

        mem_fp16 = estimate_memory(config, batch_size=1, seq_len=64, precision="fp16")
        mem_bf16 = estimate_memory(config, batch_size=1, seq_len=64, precision="bf16")

        # BF16 和 FP16 显存相近
        assert abs(mem_fp16 - mem_bf16) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
