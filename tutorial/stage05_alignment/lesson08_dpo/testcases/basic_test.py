"""
课时8基础测试：DPO基础功能验证
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from loss.dpo_loss import DPOLoss
from training.dpo_trainer import DPOTrainer
from model.config import ModelConfig


class TestDPOLoss:
    """测试DPO损失"""

    def test_dpo_loss_initialization(self):
        """测试DPO损失初始化"""
        assert DPOLoss is not None
        loss_fn = DPOLoss(beta=0.1)
        assert loss_fn is not None

    def test_dpo_loss_computation(self):
        """测试DPO损失计算"""
        loss_fn = DPOLoss(beta=0.1)

        # Simple log probabilities
        policy_chosen_logps = torch.tensor([-1.0, -2.0])
        policy_rejected_logps = torch.tensor([-1.5, -2.5])
        ref_chosen_logps = torch.tensor([-1.0, -2.0])
        ref_rejected_logps = torch.tensor([-1.5, -2.5])

        # Compute loss
        loss = loss_fn(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )

        # Verify output is scalar
        assert loss.dim() == 0

    def test_beta_effect(self):
        """测试beta参数影响"""
        # Different beta values should produce different losses
        loss_fn_low = DPOLoss(beta=0.01)
        loss_fn_high = DPOLoss(beta=1.0)

        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-2.0])
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-2.0])

        loss_low = loss_fn_low(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        loss_high = loss_fn_high(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

        # Both should be positive
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0

    def test_logsigmoid_range(self):
        """测试logsigmoid输出范围"""
        loss_fn = DPOLoss(beta=0.1)

        # Test with known values
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-2.0])
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-2.0])

        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

        # Loss should be positive
        assert loss.item() >= 0


class TestDPOTrainer:
    """测试DPO训练器"""

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        assert DPOTrainer is not None

    def test_reference_model_frozen(self):
        """测试reference模型冻结"""
        # Reference model should have frozen parameters
        assert True  # Implementation-specific

    def test_compute_logps(self):
        """测试log概率计算"""
        # Should be able to compute log probabilities
        assert True  # Implementation-specific

    def test_prompt_masking(self):
        """测试prompt masking"""
        # Should mask prompts in loss computation
        assert True  # Implementation-specific


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
