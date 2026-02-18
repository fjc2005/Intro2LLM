"""
L15: 高效注意力机制 - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file))))))

from utils.flash_attention import flash_attention


class TestFlashAttention:
    """测试 Flash Attention"""

    def test_flash_basic(self):
        """测试 Flash Attention 基本功能"""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)

        output = flash_attention(q, k, v)

        assert output.shape == q.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
