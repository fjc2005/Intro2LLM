"""
课时12基础测试：Evaluation基础功能验证
"""

import pytest
import torch
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from loss.cross_entropy import CrossEntropyLoss


class TestPerplexity:
    """测试困惑度"""

    def test_ppl_computation(self):
        """测试PPL计算"""
        # PPL = exp(loss)
        loss_fn = CrossEntropyLoss()

        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        loss = loss_fn(logits, labels)
        ppl = torch.exp(loss)

        # PPL should be positive
        assert ppl.item() >= 1.0

    def test_ppl_range(self):
        """测试PPL范围"""
        # PPL >= 1 for any model
        loss = torch.tensor(0.5)
        ppl = torch.exp(loss)

        assert ppl.item() >= 1.0


class TestCheckpoint:
    """测试检查点管理"""

    def test_checkpoint_save_load(self):
        """测试检查点保存加载"""
        import tempfile
        import shutil

        # Create a simple tensor
        tensor = torch.randn(10, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            torch.save({'model': tensor}, path)

            loaded = torch.load(path)

            assert 'model' in loaded
            assert torch.equal(loaded['model'], tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
