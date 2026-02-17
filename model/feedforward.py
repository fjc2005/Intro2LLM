"""
前馈网络模块 (Feed-Forward Network)
包含 GeGLU 和 SwiGLU 两种门控激活单元。

现代 LLM 不再使用原始的 ReLU FFN，而是采用门控机制:
- Gated Linear Units (GLU) 变体通过门控机制控制信息流动
- GeGLU 使用 GELU 作为门控激活
- SwiGLU 使用 SiLU/Swish 作为门控激活

这些设计在保持参数量相似的同时，提供了更好的表达能力。
"""

import torch
import torch.nn as nn
from typing import Callable


class FeedForward(nn.Module):
    """
    GeGLU (Gated Linear Unit with GELU) 前馈网络

    论文:
    - GLU Variants Improve Transformer (Shazeer, 2020)
      https://arxiv.org/abs/2002.05202
    - PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)

    计算公式:
        GeGLU(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down

    其中:
        - W_gate: 门控投影矩阵，[hidden_size, intermediate_size]
        - W_up: 上采样投影矩阵，[hidden_size, intermediate_size]
        - W_down: 下采样投影矩阵，[intermediate_size, hidden_size]
        - GELU: 门控激活函数
        - *: 逐元素乘法 (Hadamard product)

    与原始 FFN 的对比:
    - 原始: FFN(x) = ReLU(x @ W1) @ W2
    - GeGLU: 使用两个输入投影 (gate 和 up)，通过门控选择信息

    参数量:
    - W_gate: hidden_size * intermediate_size
    - W_up: hidden_size * intermediate_size
    - W_down: intermediate_size * hidden_size
    - 总计: ~3 * hidden_size * intermediate_size
    """

    def __init__(self, config):
        """
        初始化 GeGLU FFN。

        Args:
            config: 模型配置，包含:
                - hidden_size: 隐藏层维度
                - intermediate_size: 中间层维度
                - hidden_act: 激活函数类型，此处应为 "gelu"

        需要创建的参数:
            - gate_proj: 门控投影，nn.Linear(hidden_size, intermediate_size)
            - up_proj: 上采样投影，nn.Linear(hidden_size, intermediate_size)
            - down_proj: 下采样投影，nn.Linear(intermediate_size, hidden_size)

        注意:
            激活函数固定使用 GELU
        """
        super().__init__()
        # 从配置中提取参数
        # 初始化三个线性层
        # gate_proj 和 up_proj 将 hidden_size 映射到 intermediate_size
        # down_proj 将 intermediate_size 映射回 hidden_size
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 [batch_size, seq_len, hidden_size]

        Returns:
            输出张量，形状 [batch_size, seq_len, hidden_size]

        计算步骤:
            Step 1: 门控投影
                    gate = gate_proj(x)
                    形状: [batch, seq_len, intermediate_size]

            Step 2: 应用门控激活
                    gate = GELU(gate)
                    形状: [batch, seq_len, intermediate_size]

            Step 3: 上采样投影
                    up = up_proj(x)
                    形状: [batch, seq_len, intermediate_size]

            Step 4: 门控乘法 (Gating)
                    # 逐元素相乘，门控机制选择激活的信息
                    gated = gate * up
                    形状: [batch, seq_len, intermediate_size]

            Step 5: 下采样投影
                    output = down_proj(gated)
                    形状: [batch, seq_len, hidden_size]

        维度追踪:
            输入:  [batch, seq, hidden]
                   ↓ gate_proj, up_proj
            中间: [batch, seq, intermediate]
                   ↓ down_proj
            输出: [batch, seq, hidden]
        """
        pass


class SwiGLU(nn.Module):
    """
    SwiGLU (Gated Linear Unit with Swish/SiLU) 前馈网络

    论文:
    - GLU Variants Improve Transformer (Shazeer, 2020)
    - Swish: A Self-Gated Activation Function (Ramachandran et al., 2017)

    计算公式:
        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    其中:
        - SiLU(x) = x * sigmoid(x)，也称为 Swish 激活
        - 其他同 GeGLU

    与 GeGLU 的区别:
    - GeGLU 使用 GELU 作为门控激活
    - SwiGLU 使用 SiLU/Swish 作为门控激活

    现代 LLM 偏好:
    - LLaMA、Qwen、Mistral 等使用 SwiGLU
    - PaLM 使用 GeGLU
    - 两者性能相当，SwiGLU 略受欢迎

    SiLU 特性:
    - 自门控: SiLU(x) = x * sigmoid(x)
    - 平滑非单调，有负值区域
    - 在 Transformer 中表现良好
    """

    def __init__(self, config):
        """
        初始化 SwiGLU FFN。

        Args:
            config: 模型配置，包含:
                - hidden_size: 隐藏层维度
                - intermediate_size: 中间层维度
                - hidden_act: 激活函数类型，此处应为 "silu" 或 "swish"

        需要创建的参数:
            - gate_proj: 门控投影，nn.Linear(hidden_size, intermediate_size)
            - up_proj: 上采样投影，nn.Linear(hidden_size, intermediate_size)
            - down_proj: 下采样投影，nn.Linear(intermediate_size, hidden_size)

        注意:
            激活函数固定使用 SiLU (nn.SiLU)
        """
        super().__init__()
        # 从配置中提取参数
        # 初始化三个线性层，结构同 GeGLU
        # 激活函数使用 SiLU
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 [batch_size, seq_len, hidden_size]

        Returns:
            输出张量，形状 [batch_size, seq_len, hidden_size]

        计算步骤 (与 GeGLU 相同，只是激活函数不同):
            Step 1: 门控投影
                    gate = gate_proj(x)
                    形状: [batch, seq_len, intermediate_size]

            Step 2: 应用 Swish/SiLU 激活
                    gate = SiLU(gate)  # 或 Swish
                    形状: [batch, seq_len, intermediate_size]

            Step 3: 上采样投影
                    up = up_proj(x)
                    形状: [batch, seq_len, intermediate_size]

            Step 4: 门控乘法
                    gated = gate * up
                    形状: [batch, seq_len, intermediate_size]

            Step 5: 下采样投影
                    output = down_proj(gated)
                    形状: [batch, seq_len, hidden_size]

        SiLU 计算细节:
            SiLU(x) = x * sigmoid(x)
            其中 sigmoid(x) = 1 / (1 + exp(-x))

            这种自门控机制让 SwiGLU 比 ReLU 更平滑，比 GELU 有更直接的梯度流
        """
        pass


# 便捷函数：根据配置自动选择 FFN 类型
def get_feed_forward(config):
    """
    根据配置返回对应的 FFN 模块。

    Args:
        config: 模型配置

    Returns:
        对应的 FFN 模块类实例

    逻辑:
        if config.use_swiglu:
            return SwiGLU(config)
        else:
            return FeedForward(config)  # GeGLU
    """
    pass
