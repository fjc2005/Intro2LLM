"""
L14: LoRA 与 QLoRA - 进阶测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file))))))

from model.lora import LoRALayer


class TestLoRAAdvanced:
    """测试 LoRA 进阶"""

    def test_lora_rank(self):
        """测试不同 rank"""
        for rank in [4, 8, 16]:
            layer = LoRALayer(64, 64, rank=rank)
            x = torch.randn(1, 5, 64)
            output = layer(x)
            assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
