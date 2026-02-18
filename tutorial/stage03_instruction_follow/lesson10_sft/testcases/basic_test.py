"""
L10: 监督微调 SFT - 基础测试
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.sft_dataset import SFTDataset
from training.sft_trainer import SFTTrainer


class TestSFTDataset:
    """测试 SFT 数据集"""

    def test_dataset_creation(self):
        """测试数据集创建"""
        data = [
            {"instruction": "Hello", "response": "Hi there"},
            {"instruction": "How are you?", "response": "I'm fine"}
        ]

        dataset = SFTDataset(data, max_length=128)

        assert len(dataset) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
