"""
L14: LoRA 与 QLoRA - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file))))))

from model.lora import LoRALayer


class TestLoRA:
    """测试 LoRA"""

    def test_lora_basic(self):
        """测试 LoRA 基本功能"""
        layer = LoRALayer(128, 128, rank=8)

        x = torch.randn(2, 10, 128)
        output = layer(x)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
