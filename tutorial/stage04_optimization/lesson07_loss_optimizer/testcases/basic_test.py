"""
课时7基础测试：Loss和Optimizer基础功能验证
"""

import pytest
import torch
import torch.nn as nn
import math


class TestCrossEntropyLoss:
    """测试交叉熵损失"""

    def test_loss_initialization(self):
        """测试损失函数初始化"""
        # TODO: 创建CrossEntropyLoss
        pass

    def test_basic_loss_computation(self):
        """测试基本loss计算"""
        # TODO: logits [batch, vocab], labels [batch]
        # 验证输出标量loss
        pass

    def test_ignore_index(self):
        """测试ignore_index功能"""
        # TODO: labels中包含-100
        # 验证-100位置不计算loss
        pass

    def test_label_smoothing(self):
        """测试label smoothing"""
        # TODO: label_smoothing=0.1
        # 验证loss值与hard label不同
        pass


class TestAdamW:
    """测试AdamW优化器"""

    def test_adamw_initialization(self):
        """测试AdamW初始化"""
        # TODO: 创建AdamW，验证参数组
        pass

    def test_single_step_update(self):
        """测试单步更新"""
        # TODO: 创建简单参数和梯度
        # 执行step，验证参数更新
        pass

    def test_momentum_accumulation(self):
        """测试动量累积"""
        # TODO: 多步更新，验证一阶和二阶矩累积
        pass

    def test_weight_decay_decoupled(self):
        """测试解耦权重衰减"""
        # TODO: 验证权重衰减与学习率解耦
        pass


class TestLion:
    """测试Lion优化器"""

    def test_lion_initialization(self):
        """测试Lion初始化"""
        # TODO: 创建Lion优化器
        pass

    def test_sign_update(self):
        """测试符号更新"""
        # TODO: 验证使用sign(momentum)更新
        pass

    def test_memory_efficiency(self):
        """测试内存效率"""
        # TODO: 验证只保存一阶矩
        pass


class TestWarmupCosineScheduler:
    """测试学习率调度"""

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        # TODO: 创建WarmupCosineScheduler
        pass

    def test_warmup_phase(self):
        """测试warmup阶段"""
        # TODO: 验证学习率线性增加
        pass

    def test_cosine_decay_phase(self):
        """测试cosine decay阶段"""
        # TODO: 验证学习率余弦衰减
        pass

    def test_final_lr(self):
        """测试最终学习率"""
        # TODO: 验证total_steps时lr接近min_lr
        pass


class TestDPOLoss:
    """测试DPO损失"""

    def test_dpo_initialization(self):
        """测试DPO损失初始化"""
        # TODO: 创建DPOLoss
        pass

    def test_loss_computation(self):
        """测试loss计算"""
        # TODO: 提供policy和reference的logprobs
        # 验证返回标量loss
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
