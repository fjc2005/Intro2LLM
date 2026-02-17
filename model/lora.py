"""
LoRA (Low-Rank Adaptation) 模块
实现参数高效微调 (PEFT) 的 LoRA 方法。

LoRA 核心思想:
- 冻结预训练模型的大部分参数
- 在特定层 (通常是 Attention 和 FFN) 注入低秩矩阵
- 只训练这些低秩矩阵，大幅减少可训练参数

优势:
1. 参数高效: 可训练参数减少 1000x ~ 10000x
2. 显存节省: 不需要存储大部分参数的梯度
3. 训练快速: 参数少，前向反向更快
4. 部署灵活: 低秩矩阵可以与原权重合并，无推理开销

论文: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
链接: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class LoRALayer(nn.Module):
    """
    LoRA 层

    在原始线性层旁添加低秩分解分支:
        h = W_0 * x + (B * A) * x * scaling

    其中:
        - W_0: 预训练权重 (冻结)
        - A: 降维矩阵 [r, d_in]
        - B: 升维矩阵 [d_out, r]
        - r: 秩 (rank)，通常 4-64
        - scaling: 缩放因子 alpha/r
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        初始化 LoRA 层。

        Args:
            in_features: 输入维度
            out_features: 输出维度
            r: LoRA 秩，控制低秩矩阵的大小
            lora_alpha: LoRA 缩放参数
            lora_dropout: Dropout 概率
            merge_weights: 是否合并权重到原始层

        缩放因子计算:
            scaling = lora_alpha / r

        初始化策略:
            - A (lora_A): 正态分布初始化，N(0, 1/r)
            - B (lora_B): 零初始化
            这样初始时 LoRA 分支输出为 0，不改变原模型输出
        """
        super().__init__()
        # 保存参数
        # 创建低秩矩阵 A 和 B
        # A: [r, in_features]
        # B: [out_features, r]
        # 初始化 A 为正态分布，B 为 0
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，[..., in_features]

        Returns:
            输出张量，[..., out_features]

        计算:
            # 原始输出 (如果 W_0 存在)
            original = F.linear(x, self.weight) if hasattr(self, 'weight') else 0

            # LoRA 分支
            # Step 1: 降维
            h = F.linear(x, self.lora_A)  # [..., r]

            # Step 2: Dropout (训练时)
            if self.training:
                h = F.dropout(h, p=self.lora_dropout)

            # Step 3: 升维并缩放
            lora_output = F.linear(h, self.lora_B) * self.scaling

            # Step 4: 合并输出
            return original + lora_output

        训练时的梯度流:
            只有 lora_A 和 lora_B 有梯度，原权重无梯度
        """
        pass

    def merge(self):
        """
        合并 LoRA 权重到原权重。

        计算:
            W_merged = W_0 + B @ A * scaling

        合并后:
        - 可以直接使用原始层进行推理
        - 无额外计算开销
        - 适合部署
        """
        pass

    def unmerge(self):
        """
        解合并，恢复原始权重。

        如果需要继续训练，可以解合并。
        """
        pass


class LinearWithLoRA(nn.Module):
    """
    包装线性层，添加 LoRA 能力。

    用法:
        original_linear = nn.Linear(768, 768)
        lora_linear = LinearWithLoRA(original_linear, r=8)

        # 冻结原权重
        for param in original_linear.parameters():
            param.requires_grad = False
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
    ):
        """
        初始化。

        Args:
            linear_layer: 原始线性层
            r: LoRA 秩
            lora_alpha: 缩放参数
            lora_dropout: Dropout 概率
        """
        super().__init__()
        # 保存原始线性层
        # 创建 LoRA 层
        # 冻结原始层参数
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        pass


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none"):
    """
    只标记 LoRA 参数为可训练。

    Args:
        model: 模型
        bias: 是否训练 bias，"none" / "all" / "lora_only"

    流程:
        Step 1: 冻结所有参数
                for param in model.parameters():
                    param.requires_grad = False

        Step 2: 解冻 LoRA 参数
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        Step 3: 处理 bias (根据配置)
    """
    pass


def get_lora_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    target_modules: List[str] = None,
) -> nn.Module:
    """
    将模型转换为 LoRA 模型。

    Args:
        model: 原始模型
        r: LoRA 秩
        lora_alpha: 缩放参数
        lora_dropout: Dropout 概率
        target_modules: 要添加 LoRA 的模块名列表
                       如 ["q_proj", "v_proj", "k_proj", "o_proj"]

    Returns:
        添加了 LoRA 的模型

    流程:
        Step 1: 遍历模型的所有模块
        Step 2: 检查模块名是否在 target_modules 中
        Step 3: 替换为 LoRA 版本
        Step 4: 冻结原参数
    """
    pass


def merge_lora_weights(model: nn.Module):
    """
    合并模型中所有 LoRA 权重。

    Args:
        model: LoRA 模型
    """
    pass


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计 LoRA 参数数量。

    Args:
        model: 模型

    Returns:
        参数字典:
            - total: 总参数
            - trainable: 可训练参数
            - lora: LoRA 参数
            - lora_percentage: LoRA 参数占比
    """
    pass
