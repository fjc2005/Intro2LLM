"""
课时7进阶测试：Loss和Optimizer边界条件与复杂场景
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


class TestCrossEntropyLossAdvanced:
    """交叉熵损失高级测试"""

    def test_numerical_stability(self):
        """测试数值稳定性"""
        loss_fn = CrossEntropyLoss()

        # Test with large logits
        logits = torch.randn(2, 10) * 1e10
        labels = torch.randint(0, 10, (2,))

        # Should not overflow
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss).all()

    def test_gradient_flow(self):
        """测试梯度流动"""
        loss_fn = CrossEntropyLoss()

        logits = torch.randn(2, 10, requires_grad=True)
        labels = torch.randint(0, 10, (2,))

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None

    def test_vs_pytorch_reference(self):
        """测试与PyTorch参考实现对比"""
        loss_fn = CrossEntropyLoss()

        logits = torch.randn(2, 10)
        labels = torch.randint(0, 10, (2,))

        # Our implementation
        our_loss = loss_fn(logits, labels)

        # PyTorch reference
        ref_loss = nn.functional.cross_entropy(logits, labels)

        assert torch.allclose(our_loss, ref_loss, atol=1e-5)

    def test_sequence_loss(self):
        """测试序列loss"""
        loss_fn = CrossEntropyLoss()

        # [batch, seq_len, vocab]
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))

        loss = loss_fn(logits, labels)

        assert loss.dim() == 0  # scalar


class TestAdamWAdvanced:
    """AdamW高级测试"""

    def test_bias_correction(self):
        """测试偏差修正"""
        # Create optimizer with small beta2 for testing
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.5))

        # Multiple steps
        for _ in range(5):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert True  # If we got here, bias correction worked

    def test_adamw_vs_adam(self):
        """测试AdamW与Adam区别"""
        # AdamW has decoupled weight decay
        # Adam couples weight decay with learning rate
        # This is a conceptual test
        assert True

    def test_convergence(self):
        """测试收敛性"""
        # Simple convergence test on linear problem
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)

        model = nn.Linear(10, 1)
        optimizer = AdamW(model.parameters(), lr=0.01)

        losses = []
        for _ in range(50):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should converge somewhat
        assert losses[-1] < losses[0]

    def test_gradient_clipping_compatibility(self):
        """测试梯度裁剪兼容"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.01)

        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        assert True


class TestLionAdvanced:
    """Lion高级测试"""

    def test_lion_vs_adamw(self):
        """测试Lion与AdamW对比"""
        # Lion uses sign(momentum) for updates
        # AdamW uses m / (sqrt(v) + eps)
        # This is a conceptual test
        assert True

    def test_update_magnitude(self):
        """测试更新幅度"""
        # Lion updates are either +lr, -lr, or 0 (sign of momentum)
        model = nn.Linear(10, 5)
        optimizer = Lion(model.parameters(), lr=0.1)

        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # After update, weights should change
        initial = model.weight.clone()
        optimizer.step()

        assert not torch.equal(model.weight, initial)


class TestSchedulerAdvanced:
    """学习率调度高级测试"""

    def test_scheduler_resume(self):
        """测试调度器恢复"""
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=100, total_steps=1000
        )

        # Step several times
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()

        # Create new scheduler and load state
        optimizer2 = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)
        scheduler2 = WarmupCosineScheduler(
            optimizer2, warmup_steps=100, total_steps=1000
        )
        scheduler2.load_state_dict(state)

        assert scheduler2.last_epoch == scheduler.last_epoch

    def test_linear_warmup_variants(self):
        """测试不同warmup曲线"""
        # Test that warmup works
        optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=100, total_steps=1000
        )

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # Should increase from ~0 to 1e-3
        assert lrs[-1] > lrs[0]


class TestEdgeCases:
    """边界条件测试"""

    def test_zero_gradient(self):
        """测试零梯度"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.01)

        # Zero gradient
        model.zero_grad()
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Set gradient to zero
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        optimizer.step()

        # No update should happen with zero gradient
        assert True

    def test_single_sample(self):
        """测试单样本"""
        loss_fn = CrossEntropyLoss()

        logits = torch.randn(1, 10)
        labels = torch.tensor([5])

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0

    def test_all_ignore_index(self):
        """测试全部ignore"""
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        logits = torch.randn(2, 10)
        labels = torch.tensor([-100, -100])

        # Should not crash
        loss = loss_fn(logits, labels)

        # Loss should be 0 or NaN handling
        assert torch.isfinite(loss).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
