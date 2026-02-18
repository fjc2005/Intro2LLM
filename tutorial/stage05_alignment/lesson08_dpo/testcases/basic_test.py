"""
课时8基础测试：DPO基础功能验证
"""

import pytest
import torch
import torch.nn as nn


class TestDPOLoss:
    """测试DPO损失"""

    def test_dpo_loss_initialization(self):
        """测试DPO损失初始化"""
        # TODO: 创建DPOLoss(beta=0.1)
        pass

    def test_dpo_loss_computation(self):
        """测试DPO损失计算"""
        # TODO: 提供policy和reference的logprobs
        # 验证返回标量损失
        pass

    def test_beta_effect(self):
        """测试beta参数影响"""
        # TODO: 对比不同beta值的loss
        pass

    def test_logsigmoid_range(self):
        """测试logsigmoid输出范围"""
        # TODO: 验证loss为正数
        pass


class TestDPOTrainer:
    """测试DPO训练器"""

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        # TODO: 创建DPOTrainer
        # 验证model和ref_model正确设置
        pass

    def test_reference_model_frozen(self):
        """测试reference模型冻结"""
        # TODO: 验证ref_model参数requires_grad=False
        pass

    def test_compute_logps(self):
        """测试log概率计算"""
        # TODO: 验证能正确计算response的log概率
        pass

    def test_prompt_masking(self):
        """测试prompt masking"""
        # TODO: 验证只计算response部分的log概率
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
