"""
L08: 检查点、训练器与技巧 - 基础测试
"""

import torch
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.checkpoint import save_checkpoint, load_checkpoint
from training.pretrain_trainer import PretrainTrainer
from model.config import ModelConfig


class TestCheckpoint:
    """测试检查点"""

    def test_save_load(self):
        """测试保存和加载"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(model, optimizer, 1, path)

            assert os.path.exists(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
