"""
训练模块测试
测试训练各组件的正确性。
"""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from model.config import ModelConfig
from model.causal_lm import CausalLM
from optimizer.adamw import AdamW
from optimizer.scheduler import WarmupCosineScheduler
from loss.cross_entropy import CrossEntropyLoss
from loss.dpo_loss import DPOLoss


def test_adamw():
    """测试 AdamW 优化器。"""
    model = torch.nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # 模拟训练步骤
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)

    loss = torch.nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✓ AdamW 测试通过")


def test_scheduler():
    """测试学习率调度器。"""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
    )

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # 验证学习率变化
    assert lrs[0] < 1.0  # warmup 阶段递增
    assert lrs[9] == 1.0  # warmup 结束达到峰值
    assert lrs[-1] < lrs[50]  # 余弦衰减

    print("✓ WarmupCosineScheduler 测试通过")


def test_cross_entropy_loss():
    """测试交叉熵损失。"""
    loss_fn = CrossEntropyLoss()

    batch_size = 2
    seq_len = 10
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, labels)

    # 验证损失是标量
    assert loss.shape == torch.Size([])
    assert loss.item() > 0

    print("✓ CrossEntropyLoss 测试通过")


def test_dpo_loss():
    """测试 DPO 损失。"""
    dpo_loss = DPOLoss(beta=0.1)

    batch_size = 4

    # 模拟 log probabilities
    policy_chosen_logps = torch.randn(batch_size) * -1.0  # 负数（对数概率）
    policy_rejected_logps = torch.randn(batch_size) * -2.0
    reference_chosen_logps = torch.randn(batch_size) * -1.5
    reference_rejected_logps = torch.randn(batch_size) * -1.5

    result = dpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )

    # 验证输出
    assert "loss" in result
    assert "chosen_rewards" in result
    assert "rejected_rewards" in result
    assert "reward_margin" in result

    print("✓ DPOLoss 测试通过")


def test_training_pipeline():
    """测试完整训练流程。"""
    # 创建小型模型
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    model = CausalLM(config)

    # 创建模拟数据
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # 模拟训练步骤
    model.train()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✓ 训练流程测试通过")


if __name__ == "__main__":
    print("运行训练模块测试...\n")

    test_adamw()
    test_scheduler()
    test_cross_entropy_loss()
    test_dpo_loss()
    test_training_pipeline()

    print("\n✅ 所有测试通过！")
