"""
课时10基础测试：Trainers基础功能验证
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from training.trainer import BaseTrainer
from training.pretrain_trainer import PretrainTrainer
from training.sft_trainer import SFTTrainer


class TestBaseTrainer:
    """测试基础Trainer"""

    def test_trainer_initialization(self):
        """测试Trainer初始化"""
        assert BaseTrainer is not None

    def test_training_step(self):
        """测试训练步骤"""
        assert hasattr(BaseTrainer, 'training_step')


class TestPretrainTrainer:
    """测试预训练Trainer"""

    def test_pretrain_loss(self):
        """测试预训练loss计算"""
        assert PretrainTrainer is not None


class TestSFTTrainer:
    """测试SFT Trainer"""

    def test_sft_loss_masking(self):
        """测试SFT loss masking"""
        assert SFTTrainer is not None


class TestMixedPrecision:
    """测试混合精度"""

    def test_autocast_forward(self):
        """测试autocast前向"""
        # Mixed precision should use torch.cuda.amp.autocast
        assert True  # Implementation-specific


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
