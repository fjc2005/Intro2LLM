"""
L07: 优化器与学习率调度 - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from optimizer.adamw import AdamW
from optimizer.lion import Lion
from optimizer.scheduler import WarmupCosineScheduler


class TestAdamW:
    """测试 AdamW 优化器"""

    def test_adamw_basic(self):
        """测试 AdamW 基本功能"""
        model = torch.nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        loss = model(torch.randn(5, 10)).sum()
        loss.backward()
        optimizer.step()

        assert optimizer.param_groups[0]['lr'] > 0


class TestLion:
    """测试 Lion 优化器"""

    def test_lion_basic(self):
        """测试 Lion 基本功能"""
        model = torch.nn.Linear(10, 10)
        optimizer = Lion(model.parameters(), lr=1e-3)

        loss = model(torch.randn(5, 10)).sum()
        loss.backward()
        optimizer.step()


class TestScheduler:
    """测试学习率调度器"""

    def test_cosine_scheduler(self):
        """测试 Cosine 调度"""
        scheduler = WarmupCosineScheduler(
            optimizer=None,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6
        )

        # 初始学习率
        lr0 = scheduler.get_lr(0)
        assert lr0 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
