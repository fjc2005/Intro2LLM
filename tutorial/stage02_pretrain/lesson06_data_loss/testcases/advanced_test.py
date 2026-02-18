"""
L06: 数据处理与损失函数 - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.cross_entropy import compute_cross_entropy_loss


class TestCrossEntropyAdvanced:
    """测试 Cross Entropy 进阶特性"""

    def test_loss_fp16(self):
        """测试 FP16 损失计算"""
        logits = torch.randn(2, 8, 1000, dtype=torch.float16)
        labels = torch.randint(0, 1000, (2, 8))

        loss = compute_cross_entropy_loss(logits, labels)

        assert not torch.isnan(loss)

    def test_loss_large_vocab(self):
        """测试大词表"""
        logits = torch.randn(2, 8, 50000)
        labels = torch.randint(0, 50000, (2, 8))

        loss = compute_cross_entropy_loss(logits, labels)

        assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
