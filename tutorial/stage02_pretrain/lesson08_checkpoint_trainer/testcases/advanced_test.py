"""
L08: 检查点、训练器与技巧 - 进阶测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.checkpoint import save_checkpoint, load_checkpoint
import tempfile
import torch


class TestCheckpointAdvanced:
    """测试检查点进阶"""

    def test_save_with_optimizer(self):
        """测试保存优化器状态"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(model, optimizer, 5, path)

            # 验证可以加载
            ckpt = torch.load(path)
            assert 'optimizer' in ckpt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
