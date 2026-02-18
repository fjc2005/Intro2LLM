"""
L15: 高效注意力机制 - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file))))))

from utils.flash_attention import flash_attention


class TestFlashAttentionAdvanced:
    """测试 Flash Attention 进阶"""

    def test_flash_mask(self):
        """测试带掩码的 Flash Attention"""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)

        # 因果掩码
        mask = torch.tril(torch.ones(8, 8))

        output = flash_attention(q, k, v, mask=mask)

        assert output.shape == q.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
