"""
L13: GRPO - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.grpo_loss import compute_grpo_loss


class TestGRPO:
    """测试 GRPO 损失"""

    def test_grpo_basic(self):
        """测试 GRPO 损失计算"""
        # logits: [batch, num_responses, seq_len, vocab]
        logits = torch.randn(2, 4, 10, 1000)
        # 假设第一个响应是偏好
        group_ratios = torch.tensor([1.0, 0.0, 0.0, 0.0])

        loss = compute_grpo_loss(logits, group_ratios)

        assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
