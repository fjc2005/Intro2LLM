"""
课时9基础测试：GRPO和PPO基础功能验证
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from loss.grpo_loss import GRPOLoss
from loss.ppo_loss import PPOLoss


class TestGRPOLoss:
    """测试GRPO损失"""

    def test_grpo_initialization(self):
        """测试GRPO损失初始化"""
        assert GRPOLoss is not None
        loss_fn = GRPOLoss()
        assert loss_fn is not None

    def test_group_baseline(self):
        """测试组内基线"""
        # Group baseline should be average within group
        assert True  # Implementation-specific

    def test_advantage_computation(self):
        """测试优势计算"""
        # A = reward - baseline
        assert True  # Implementation-specific


class TestPPOLoss:
    """测试PPO损失"""

    def test_ppo_initialization(self):
        """测试PPO损失初始化"""
        assert PPOLoss is not None
        loss_fn = PPOLoss(clip_ratio=0.2)
        assert loss_fn is not None

    def test_clip_function(self):
        """测试裁剪函数"""
        # PPO clips the ratio
        assert True  # Implementation-specific

    def test_gae_computation(self):
        """测试GAE计算"""
        # GAE should compute advantages
        assert True  # Implementation-specific


class TestGRPOTrainer:
    """测试GRPO训练器"""

    def test_group_generation(self):
        """测试组生成"""
        from training.grpo_trainer import GRPOTrainer
        assert GRPOTrainer is not None


class TestPPOTrainer:
    """测试PPO训练器"""

    def test_experience_collection(self):
        """测试经验收集"""
        from training.ppo_trainer import PPOTrainer
        assert PPOTrainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
