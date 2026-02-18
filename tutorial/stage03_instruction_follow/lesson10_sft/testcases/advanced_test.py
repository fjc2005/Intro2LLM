"""
L10: 监督微调 SFT - 进阶测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.sft_dataset import SFTDataset


class TestSFTAdvanced:
    """测试 SFT 进阶"""

    def test_loss_mask(self):
        """测试损失掩码"""
        data = [{"instruction": "Hello", "response": "Hi there"}]
        dataset = SFTDataset(data, max_length=128)

        item = dataset[0]

        assert 'labels' in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
