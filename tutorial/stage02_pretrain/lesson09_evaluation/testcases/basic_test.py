"""
L09: 模型评估 - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from evaluation.perplexity import compute_perplexity


class TestPerplexity:
    """测试 Perplexity 计算"""

    def test_perplexity_basic(self):
        """测试基本 Perplexity"""
        # 模拟 logits 和 labels
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))

        ppl = compute_perplexity(logits, labels)

        assert ppl > 0
        assert not torch.isnan(ppl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
