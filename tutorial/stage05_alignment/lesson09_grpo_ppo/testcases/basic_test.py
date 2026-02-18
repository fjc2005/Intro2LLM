"""
课时9基础测试：GRPO和PPO基础功能验证
"""

import pytest
import torch


class TestGRPOLoss:
    """测试GRPO损失"""

    def test_grpo_initialization(self):
        """测试GRPO损失初始化"""
        # TODO: 创建GRPOLoss
        pass

    def test_group_baseline(self):
        """测试组内基线"""
        # TODO: 验证基线为组内平均
        pass

    def test_advantage_computation(self):
        """测试优势计算"""
        # TODO: 验证A = reward - baseline
        pass


class TestPPOLoss:
    """测试PPO损失"""

    def test_ppo_initialization(self):
        """测试PPO损失初始化"""
        # TODO: 创建PPOLoss
        pass

    def test_clip_function(self):
        """测试裁剪函数"""
        # TODO: 验证ratio被正确裁剪
        pass

    def test_gae_computation(self):
        """测试GAE计算"""
        # TODO: 验证GAE返回值和优势
        pass


class TestGRPOTrainer:
    """测试GRPO训练器"""

    def test_group_generation(self):
        """测试组生成"""
        # TODO: 验证生成group_size个输出
        pass


class TestPPOTrainer:
    """测试PPO训练器"""

    def test_experience_collection(self):
        """测试经验收集"""
        # TODO: 验证收集states, actions, rewards
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
