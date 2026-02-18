"""
L07: 优化器与学习率调度 - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from optimizer.scheduler import WarmupCosineScheduler


class TestSchedulerAdvanced:
    """测试调度器进阶特性"""

    def test_warmup(self):
        """测试预热阶段"""
        scheduler = WarmupCosineScheduler(
            optimizer=None,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-6
        )

        # 预热阶段学习率应该增加
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        assert lr_50 > lr_0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
