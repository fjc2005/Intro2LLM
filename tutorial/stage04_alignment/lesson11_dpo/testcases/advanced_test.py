"""
L11: DPO 与 IPO - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.dpo_loss import compute_dpo_loss


class TestDPOAdvanced:
    """测试 DPO 进阶"""

    def test_dpo_stability(self):
        """测试 DPO 数值稳定性"""
        policy_chosen = torch.randn(2, 10, 1000)
        policy_rejected = torch.randn(2, 10, 1000)

        loss = compute_dpo_loss(policy_chosen, policy_rejected)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
