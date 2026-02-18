"""
课时4基础测试：FFN和Transformer Block基础功能验证
"""

import pytest
import torch
import torch.nn as nn


class TestSwiGLU:
    """测试SwiGLU基础功能"""

    def test_swiglu_initialization(self):
        """测试SwiGLU正确初始化"""
        # TODO: 创建SwiGLU(hidden_size=256, intermediate_size=688)
        # 验证gate_proj, up_proj, down_proj存在
        # 验证无bias
        pass

    def test_swiglu_output_shape(self):
        """测试SwiGLU输出形状"""
        # TODO: 输入x = torch.randn(2, 10, 256)
        # 验证输出形状为[2, 10, 256]
        pass

    def test_swiglu_activation(self):
        """测试Swish/SiLU激活"""
        # TODO: 验证门控使用SiLU激活
        # 对比手动计算和模块输出
        pass

    def test_swiglu_gating(self):
        """测试门控机制"""
        # TODO: 验证gate和up的逐元素乘法
        pass


class TestGeGLU:
    """测试GeGLU基础功能"""

    def test_geglu_initialization(self):
        """测试GeGLU正确初始化"""
        # TODO: 创建GeGLU(hidden_size=256, intermediate_size=688)
        pass

    def test_geglu_output_shape(self):
        """测试GeGLU输出形状"""
        # TODO: 输入x = torch.randn(2, 10, 256)
        # 验证输出形状为[2, 10, 256]
        pass

    def test_geglu_activation(self):
        """测试GELU激活"""
        # TODO: 验证门控使用GELU激活
        pass


class TestFeedForward:
    """测试统一FFN接口"""

    def test_ffn_interface(self):
        """测试FFN接口一致性"""
        # TODO: 测试不同激活函数输出形状一致
        pass

    def test_ffn_dropout(self):
        """测试FFN dropout"""
        # TODO: 验证训练和eval时dropout行为不同
        pass


class TestTransformerBlock:
    """测试Transformer Block基础功能"""

    def test_block_initialization(self):
        """测试Block正确初始化"""
        # TODO: 创建TransformerBlock
        # 验证包含: input_layernorm, self_attn, post_attention_layernorm, mlp
        pass

    def test_block_output_shape(self):
        """测试Block输出形状"""
        # TODO: 输入x = torch.randn(2, 10, 256)
        # 验证输出形状为[2, 10, 256]
        pass

    def test_pre_ln_structure(self):
        """测试Pre-LN结构"""
        # TODO: 验证顺序: norm → sublayer → residual
        pass

    def test_residual_connection(self):
        """测试残差连接"""
        # TODO: 对比有无残差连接的输出差异
        # 验证输出 = 输入 + 变换
        pass

    def test_kv_cache_integration(self):
        """测试KV缓存集成"""
        # TODO: 设置use_cache=True
        # 验证返回present_key_value
        pass


class TestDropout:
    """测试Dropout功能"""

    def test_dropout_training_mode(self):
        """测试训练模式dropout"""
        # TODO: model.train()模式下
        # 验证每次前向输出不同
        pass

    def test_dropout_eval_mode(self):
        """测试评估模式dropout"""
        # TODO: model.eval()模式下
        # 验证每次前向输出相同
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
