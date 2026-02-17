"""
计算与数据工程模块测试
测试 FLOPs 计算、显存估算和数据处理功能。
"""

import pytest
from utils.compute import (
    estimate_train_flops,
    estimate_inference_flops,
    estimate_train_memory,
    estimate_inference_memory,
    estimate_training_time,
    print_memory_breakdown,
)


def test_estimate_train_flops():
    """测试训练 FLOPs 估算"""
    # 7B 模型配置
    model_params = 7_000_000_000  # 7B
    batch_size = 4
    seq_len = 4096
    num_layers = 32
    hidden_size = 4096
    num_heads = 32
    intermediate_size = 11008

    flops = estimate_train_flops(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
    )

    # 验证 FLOPs 是正数
    assert flops > 0

    # 验证 FLOPs 大小合理 (7B 模型约 42B FLOPs/token, batch=4 -> ~168B)
    # 允许一些误差范围
    expected_flops_per_token = 6 * model_params  # 经验公式
    assert flops / batch_size > expected_flops_per_token * 0.8
    assert flops / batch_size < expected_flops_per_token * 1.5

    print(f"✓ 训练 FLOPs: {flops / 1e12:.2f} TFLOPs (batch={batch_size})")


def test_estimate_train_flops_gqa():
    """测试使用 GQA 时的 FLOPs 估算"""
    model_params = 7_000_000_000
    batch_size = 4
    seq_len = 4096
    num_layers = 32
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 8  # GQA: 8 个 KV 头
    intermediate_size = 11008

    # 不使用 GQA
    flops_standard = estimate_train_flops(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_heads,  # 标准 MHA
        intermediate_size=intermediate_size,
    )

    # 使用 GQA
    flops_gqa = estimate_train_flops(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        use_gqa=True,
    )

    # GQA 应该减少 FLOPs
    assert flops_gqa < flops_standard

    print(f"✓ MHA FLOPs: {flops_standard / 1e12:.2f} TFLOPs")
    print(f"✓ GQA FLOPs: {flops_gqa / 1e12:.2f} TFLOPs (减少 {100 * (1 - flops_gqa / flops_standard):.1f}%)")


def test_estimate_inference_flops():
    """测试推理 FLOPs 估算"""
    model_params = 7_000_000_000
    batch_size = 1
    seq_len = 4096

    # 带 KV Cache
    flops_with_cache = estimate_inference_flops(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        use_kv_cache=True,
    )

    # 不带 KV Cache
    flops_without_cache = estimate_inference_flops(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        use_kv_cache=False,
    )

    # 带 KV Cache 应该减少 FLOPs
    assert flops_with_cache < flops_without_cache

    print(f"✓ 推理 FLOPs (with KV Cache): {flops_with_cache / 1e12:.2f} TFLOPs")
    print(f"✓ 推理 FLOPs (no KV Cache): {flops_without_cache / 1e12:.2f} TFLOPs")


def test_estimate_train_memory():
    """测试训练显存估算"""
    model_params = 7_000_000_000  # 7B
    batch_size = 4
    seq_len = 4096
    hidden_size = 4096
    num_layers = 32

    # FP16 + Adam
    memory_fp16 = estimate_train_memory(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="fp16",
        optimizer="adamw",
    )

    # 转换为 GB
    memory_gb = memory_fp16 / (1024 ** 3)

    # 7B 模型 FP16 训练约需 140-200GB 显存
    assert memory_gb > 100
    assert memory_gb < 300

    print(f"✓ 训练显存 (FP16 + AdamW): {memory_gb:.1f} GB")


def test_estimate_train_memory_optimizer():
    """测试不同优化器的显存差异"""
    model_params = 7_000_000_000
    batch_size = 4
    seq_len = 4096
    hidden_size = 4096
    num_layers = 32

    # AdamW
    memory_adam = estimate_train_memory(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="fp16",
        optimizer="adamw",
    )

    # Lion (只需要一个状态)
    memory_lion = estimate_train_memory(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="fp16",
        optimizer="lion",
    )

    # Lion 应该比 Adam 节省约一半优化器显存
    assert memory_lion < memory_adam
    saving_ratio = 1 - memory_lion / memory_adam
    assert saving_ratio > 0.3  # 至少节省 30%

    print(f"✓ AdamW 显存: {memory_adam / (1024**3):.1f} GB")
    print(f"✓ Lion 显存: {memory_lion / (1024**3):.1f} GB (节省 {saving_ratio*100:.1f}%)")


def test_estimate_train_memory_gradient_checkpointing():
    """测试梯度检查点对显存的影响"""
    model_params = 7_000_000_000
    batch_size = 4
    seq_len = 4096
    hidden_size = 4096
    num_layers = 32

    # 不使用 gradient checkpointing
    memory_no_gc = estimate_train_memory(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="fp16",
        optimizer="adamw",
        use_gradient_checkpointing=False,
    )

    # 使用 gradient checkpointing
    memory_with_gc = estimate_train_memory(
        model_params=model_params,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="fp16",
        optimizer="adamw",
        use_gradient_checkpointing=True,
    )

    # Gradient checkpointing 应该减少显存
    assert memory_with_gc < memory_no_gc

    print(f"✓ 无 GC 显存: {memory_no_gc / (1024**3):.1f} GB")
    print(f"✓ 有 GC 显存: {memory_with_gc / (1024**3):.1f} GB")


def test_estimate_inference_memory():
    """测试推理显存估算"""
    model_params = 7_000_000_000
    batch_size = 1
    max_seq_len = 4096
    hidden_size = 4096
    num_layers = 32

    # FP16 推理
    memory_fp16 = estimate_inference_memory(
        model_params=model_params,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="fp16",
        use_kv_cache=True,
    )

    # Int8 推理
    memory_int8 = estimate_inference_memory(
        model_params=model_params,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        precision="int8",
        use_kv_cache=True,
    )

    # Int8 应该显著减少显存
    assert memory_int8 < memory_fp16

    print(f"✓ FP16 推理显存: {memory_fp16 / (1024**3):.1f} GB")
    print(f"✓ Int8 推理显存: {memory_int8 / (1024**3):.1f} GB")


def test_estimate_training_time():
    """测试训练时间估算"""
    # 7B 模型，1T tokens，8x H100
    flops_per_token = 6 * 7_000_000_000  # ~42B
    batch_size = 16
    num_tokens = 1_000_000_000_000  # 1T
    num_gpus = 8

    time_hours = estimate_training_time(
        flops_per_token=flops_per_token,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_gpus=num_gpus,
        gpu_tflops=312,  # H100
        utilization=0.5,
    )

    # 估算应该在合理范围内 (约 1-2 小时)
    assert time_hours > 0
    assert time_hours < 10

    print(f"✓ 估算训练时间: {time_hours:.1f} 小时")


def test_print_memory_breakdown():
    """测试显存分解报告"""
    print_memory_breakdown(
        model_params=7_000_000_000,
        batch_size=4,
        seq_len=4096,
        hidden_size=4096,
        num_layers=32,
        precision="fp16",
        optimizer="adamw",
    )
    print("✓ 显存分解报告测试通过")
