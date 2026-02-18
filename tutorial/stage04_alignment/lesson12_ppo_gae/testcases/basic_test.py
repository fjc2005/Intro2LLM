"""
L12: PPO 与 GAE - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.ppo_loss import compute_gae


class TestPPO:
    """测试 PPO 损失"""

    def test_gae_basic(self):
        """测试 GAE 计算"""
        rewards = torch.randn(2, 10)
        values = torch.randn(2, 10)
        dones = torch.zeros(2, 10)

        advantages = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

        assert advantages.shape == rewards.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
