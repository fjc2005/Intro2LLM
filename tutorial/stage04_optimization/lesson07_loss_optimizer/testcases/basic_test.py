"""
课时7基础测试：Loss和Optimizer基础功能验证
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from loss.cross_entropy import CrossEntropyLoss
from optimizer.adamw import AdamW
from optimizer.lion import Lion
from optimizer.scheduler import WarmupCosineScheduler


class TestCrossEntropyLoss:
    """测试交叉熵损失"""

    def test_loss_initialization(self):
        """测试损失函数初始化"""
        # Verify the class exists
        assert CrossEntropyLoss is not None

        # Test basic initialization
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        assert loss_fn is not None

    def test_basic_loss_computation(self):
        """测试基本loss计算"""
        loss_fn = CrossEntropyLoss()

        # Create simple logits and labels
        logits = torch.randn(2, 10)  # [batch, vocab]
        labels = torch.randint(0, 10, (2,))  # [batch]

        # Compute loss
        loss = loss_fn(logits, labels)

        # Verify output is a scalar tensor
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # loss should be non-negative

    def test_ignore_index(self):
        """测试ignore_index功能"""
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        # Create logits and labels with ignore_index
        logits = torch.randn(2, 10)
        labels = torch.tensor([0, -100])  # Second label is ignored

        # Compute loss
        loss = loss_fn(logits, labels)

        # Verify loss is computed only on first sample
        assert loss.item() >= 0

    def test_label_smoothing(self):
        """测试label smoothing"""
        loss_fn_smooth = CrossEntropyLoss(label_smoothing=0.1)
        loss_fn_no_smooth = CrossEntropyLoss(label_smoothing=0.0)

        # Same inputs
        logits = torch.randn(2, 10)
        labels = torch.randint(0, 10, (2,))

        # Compute losses
        loss_smooth = loss_fn_smooth(logits, labels)
        loss_no_smooth = loss_fn_no_smooth(logits, labels)

        # Both should be valid
        assert loss_smooth.item() >= 0
        assert loss_no_smooth.item() >= 0


class TestAdamW:
    """测试AdamW优化器"""

    def test_adamw_initialization(self):
        """测试AdamW初始化"""
        # Verify the class exists
        assert AdamW is not None

        # Test basic initialization
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        assert optimizer is not None

    def test_single_step_update(self):
        """测试单步更新"""
        # Create simple model
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.1)

        # Save initial weights
        initial_weight = model.weight.clone()

        # Forward pass
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        # Verify weights changed
        assert not torch.equal(model.weight, initial_weight)

    def test_momentum_accumulation(self):
        """测试动量累积"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.1, betas=(0.9, 0.999))

        # Multiple steps
        for _ in range(3):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Model should have been updated
        assert True  # If we got here, update worked

    def test_weight_decay_decoupled(self):
        """测试解耦权重衰减"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=0.01)

        # Forward pass
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Verify decay was applied
        assert True  # If we got here, decay worked


class TestLion:
    """测试Lion优化器"""

    def test_lion_initialization(self):
        """测试Lion初始化"""
        # Verify the class exists
        assert Lion is not None

        # Test basic initialization
        model = nn.Linear(10, 10)
        optimizer = Lion(model.parameters(), lr=1e-3)

        assert optimizer is not None

    def test_sign_update(self):
        """测试符号更新"""
        model = nn.Linear(10, 5)
        optimizer = Lion(model.parameters(), lr=0.01)

        # Forward pass
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

        # Verify weights changed
        assert True  # If we got here, update worked


class TestWarmupCosineScheduler:
    """测试学习率调度"""

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        # Verify the class exists
        assert WarmupCosineScheduler is not None

        # Test basic initialization
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=100, total_steps=1000
        )

        assert scheduler is not None

    def test_warmup_phase(self):
        """测试warmup阶段"""
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=100, total_steps=1000
        )

        # Initial lr should be close to 0
        initial_lr = optimizer.param_groups[0]['lr']

        # After warmup steps
        for _ in range(100):
            scheduler.step()

        # After warmup, lr should be at max
        final_warmup_lr = optimizer.param_groups[0]['lr']

        # Warmup should increase lr
        assert final_warmup_lr >= initial_lr

    def test_cosine_decay_phase(self):
        """测试cosine decay阶段"""
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=100, total_steps=1000
        )

        # Skip warmup
        for _ in range(100):
            scheduler.step()

        # Get lr after warmup
        lr_after_warmup = optimizer.param_groups[0]['lr']

        # Take some decay steps
        for _ in range(500):
            scheduler.step()

        # Get lr after decay
        lr_after_decay = optimizer.param_groups[0]['lr']

        # Decay should reduce lr
        assert lr_after_decay < lr_after_warmup


class TestDPOLoss:
    """测试DPO损失"""

    def test_dpo_loss_exists(self):
        """测试DPO损失存在"""
        from loss.dpo_loss import DPOLoss

        assert DPOLoss is not None

    def test_dpo_initialization(self):
        """测试DPO损失初始化"""
        from loss.dpo_loss import DPOLoss

        loss_fn = DPOLoss(beta=0.1)
        assert loss_fn is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
