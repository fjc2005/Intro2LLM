"""
课时11基础测试：LoRA和Flash Attention基础功能验证
"""

import pytest
import torch


class TestLoRA:
    """测试LoRA"""

    def test_lora_initialization(self):
        """测试LoRA初始化"""
        # TODO: 验证A随机初始化，B零初始化
        pass

    def test_lora_forward(self):
        """测试LoRA前向"""
        # TODO: 验证输出 = base + lora
        pass

    def test_lora_scaling(self):
        """测试LoRA缩放"""
        # TODO: 验证alpha/r缩放
        pass


class TestFlashAttention:
    """测试Flash Attention"""

    def test_flash_attention_output(self):
        """测试Flash Attention输出"""
        # TODO: 验证输出与标准attention一致
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
