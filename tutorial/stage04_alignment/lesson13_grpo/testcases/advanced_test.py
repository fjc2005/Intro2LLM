"""
L13: GRPO - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.grpo_loss import compute_grpo_loss


class TestGRPOAdvanced:
    """测试 GRPO 进阶"""

    def test_grpo_stability(self):
        """测试 GRPO 数值稳定性"""
        logits = torch.randn(2, 4, 10, 1000)
        group_ratios = torch.tensor([1.0, 0.0, 0.0, 0.0])

        loss = compute_grpo_loss(logits, group_ratios)

        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
