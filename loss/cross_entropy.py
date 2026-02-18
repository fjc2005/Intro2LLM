"""
交叉熵损失模块
用于语言模型的预训练和监督微调 (SFT)。

交叉熵损失是分类任务的标准损失函数，在语言模型中用于衡量
模型预测下一个 token 的分布与真实分布的差异。

数学定义:
    CE(p, q) = -sum(p(x) * log(q(x)))

对于语言模型:
    - p(x) 是真实分布 (one-hot，目标 token 为 1，其余为 0)
    - q(x) 是模型预测的概率分布 (softmax 后的 logits)
    - 因此 CE = -log(q(target))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    用于语言模型的交叉熵损失

    支持:
    - 标准交叉熵 (预训练)
    - 忽略特定索引 (如 padding)
    - 标签平滑 (可选)
    - 按 token 数平均损失

    与 nn.CrossEntropyLoss 的区别:
    - 自动处理 logits 的 reshape
    - 支持语言模型特定的选项
    """

    def __init__(
        self,
        config = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        初始化交叉熵损失。

        Args:
            config: 模型配置 (可选)
            ignore_index: 忽略的索引，对应位置的损失不计算
                         通常用于 padding token 和 prompt 部分
            label_smoothing: 标签平滑系数，0 表示不平滑
                            用于防止模型过于自信，提高泛化
            reduction: 损失缩减方式，"mean" | "sum" | "none"
                      "mean": 按有效 token 数平均 (推荐)
                      "sum": 直接求和
                      "none": 不缩减，返回每个位置的损失

        标签平滑原理:
            将 hard target (one-hot) 变为 soft target:
                original: [0, 0, 1, 0, 0]
                smoothed: [0.02, 0.02, 0.92, 0.02, 0.02] (smoothing=0.08)

            公式:
                smoothed = (1 - smoothing) * one_hot + smoothing / vocab_size
        """
        super().__init__()
        # 保存配置参数
        # 创建 nn.CrossEntropyLoss 或自行实现
        pass

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算交叉熵损失。

        Args:
            logits: 模型输出的 logits
                   形状: [batch_size, seq_len, vocab_size]
                   或 [batch_size * seq_len, vocab_size]
            labels: 目标 token ID
                   形状: [batch_size, seq_len]
                   或 [batch_size * seq_len]
                   ignore_index 的位置会被忽略

        Returns:
            标量损失值 (如果 reduction != "none")
            或每个位置的损失 (如果 reduction == "none")

        计算流程:

            Step 1: 验证输入形状
                    logits: [batch, seq_len, vocab_size]
                    labels: [batch, seq_len]

            Step 2: 展平为 2D 张量
                    将三维 logits 和二维 labels 展平为二维
                    logits_flat 形状: [batch * seq_len, vocab_size]
                    labels_flat 形状: [batch * seq_len]

            Step 3: 计算交叉熵
                    使用 F.cross_entropy 或手动计算:

                    手动计算过程:
                    - 应用 log_softmax 将 logits 转换为对数概率
                    - 使用 gather 操作收集目标 token 位置的对数概率
                    - 取负得到负对数似然损失

            Step 4: 处理 ignore_index
                    标记 ignore_index 位置为无效，不参与损失计算
                    创建一个掩码，忽略_index 对应位置的损失

            Step 5: 缩减损失
                    根据 reduction 参数:
                    - "mean": 按有效 token 数求平均
                    - "sum": 求和
                    - "none": 保留每个位置的损失

            Step 6: 返回损失

        边界条件:
            - 如果所有 token 都是 ignore_index，避免除零
            - 如果 vocab_size 很大，注意数值稳定性 (使用 log_softmax)
        """
        pass

    def compute_perplexity(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算困惑度 (Perplexity, PPL)。

        困惑度是语言模型的标准评估指标，定义为:
            PPL = exp(CE_loss)

        直观理解: 模型每次预测下一个 token 时，面对的选择数量。
        - PPL = 100 表示模型不确定度相当于从 100 个等概率选择中猜测
        - 越低越好，随机猜测时 PPL = vocab_size

        Args:
            logits: 模型输出的 logits，[batch, seq, vocab_size]
            labels: 目标标签，[batch, seq]

        Returns:
            困惑度值，标量张量
        """
        pass
