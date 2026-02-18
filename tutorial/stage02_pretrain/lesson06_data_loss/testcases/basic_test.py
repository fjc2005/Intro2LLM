"""
L06: 数据处理与损失函数 - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from loss.cross_entropy import compute_cross_entropy_loss
from data.filtering import filter_duplicates, filter_low_quality


class TestCrossEntropyLoss:
    """测试 Cross Entropy 损失"""

    def test_loss_basic(self):
        """测试基本损失计算"""
        # logits: [batch, seq, vocab]
        logits = torch.randn(2, 8, 1000)
        # labels: [batch, seq]
        labels = torch.randint(0, 1000, (2, 8))

        loss = compute_cross_entropy_loss(logits, labels)

        assert loss > 0
        assert not torch.isnan(loss)

    def test_loss_with_mask(self):
        """测试带掩码的损失"""
        logits = torch.randn(2, 8, 1000)
        labels = torch.randint(0, 1000, (2, 8))
        labels[:, -2:] = -100  # mask

        loss = compute_cross_entropy_loss(logits, labels)

        assert loss > 0


class TestDataFiltering:
    """测试数据过滤"""

    def test_filter_duplicates(self):
        """测试重复数据过滤"""
        texts = ["hello world", "hello world", "new text"]

        filtered = filter_duplicates(texts)

        assert len(filtered) == 2

    def test_filter_low_quality(self):
        """测试低质量数据过滤"""
        texts = ["hello", "a", "This is a normal sentence"]

        filtered = filter_low_quality(texts, min_length=5)

        assert "This is a normal sentence" in filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
