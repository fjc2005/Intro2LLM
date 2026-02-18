"""
L12: PPO 与 GAE - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.ppo_loss import compute_gae


class TestPPOAdvanced:
    """测试 PPO 进阶"""

    def test_gae_discount(self):
        """测试折扣因子"""
        rewards = torch.tensor([[1.0, 0.0, 0.0]])
        values = torch.tensor([[0.9, 0.8, 0.7]])
        dones = torch.tensor([[False, True, True]])

        advantages = compute_gae(rewards, values, dones, gamma=0.9, lam=0.95)

        assert not torch.isnan(advantages).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
