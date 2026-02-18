"""
L09: 模型评估 - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file))))))

from evaluation.perplexity import compute_perplexity


class TestPerplexityAdvanced:
    """测试 Perplexity 进阶"""

    def test_perplexity_per_sequence(self):
        """测试逐序列 Perplexity"""
        logits = torch.randn(3, 10, 1000)
        labels = torch.randint(0, 1000, (3, 10))

        ppl = compute_perplexity(logits, labels, reduction='none')

        assert ppl.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
