"""
课时12基础测试：Evaluation基础功能验证
"""

import pytest
import torch


class TestPerplexity:
    """测试困惑度"""

    def test_ppl_computation(self):
        """测试PPL计算"""
        # TODO: 验证PPL = exp(loss)
        pass

    def test_ppl_range(self):
        """测试PPL范围"""
        # TODO: 验证PPL >= 1
        pass


class TestCheckpoint:
    """测试检查点管理"""

    def test_checkpoint_save_load(self):
        """测试检查点保存加载"""
        # TODO: 验证状态一致
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
