"""
L11: DPO 与 IPO - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.dpo_loss import compute_dpo_loss


class TestDPOLoss:
    """测试 DPO 损失"""

    def test_dpo_basic(self):
        """测试 DPO 损失计算"""
        # 模拟 logits
        # chosen: 偏好响应
        # rejected: 不偏好响应
        policy_chosen = torch.randn(2, 10, 1000)
        policy_rejected = torch.randn(2, 10, 1000)

        loss = compute_dpo_loss(policy_chosen, policy_rejected)

        assert loss > 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
