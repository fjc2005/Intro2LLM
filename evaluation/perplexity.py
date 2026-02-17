"""
困惑度 (Perplexity) 评估模块
计算语言模型在测试集上的困惑度。

困惑度 (PPL) 是语言模型的标准评估指标，定义为:
    PPL = exp(-1/N * sum(log P(x_i | x_<i)))

直观理解:
- 模型有多"困惑"，即模型预测下一个词时面临的选择数量
- PPL = 100 相当于从 100 个等概率选择中猜测
- 越低越好，随机猜测时 PPL = vocab_size

应用场景:
- 语言模型预训练评估
- 机器翻译质量评估
- 语音识别评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import math


class PerplexityEvaluator:
    """
    困惑度评估器

    计算语言模型在数据集上的困惑度。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
    ):
        """
        初始化困惑度评估器。

        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        计算困惑度。

        Args:
            dataloader: 数据加载器
            max_samples: 最大评估样本数 (None 表示全部)

        Returns:
            包含以下字段的字典:
                - perplexity: 困惑度
                - loss: 平均交叉熵损失
                - num_tokens: 评估的总 token 数

        计算流程:
            Step 1: 设置模型为评估模式
                    model.eval()

            Step 2: 初始化统计变量
                    total_loss = 0
                    total_tokens = 0

            Step 3: 遍历数据集
                    with torch.no_grad():
                        for batch in dataloader:
                            Step 3.1: 移动数据到设备
                            Step 3.2: 前向传播
                                     outputs = model(**batch)
                                     loss = outputs.loss

                            Step 3.3: 统计
                                     # 获取有效 token 数
                                     num_valid_tokens = (batch["labels"] != -100).sum()
                                     total_loss += loss.item() * num_valid_tokens
                                     total_tokens += num_valid_tokens

                            Step 3.4: 检查样本数限制
                                     if max_samples and samples >= max_samples:
                                         break

            Step 4: 计算平均损失和困惑度
                    avg_loss = total_loss / total_tokens
                    perplexity = math.exp(avg_loss)

            Step 5: 返回结果
        """
        pass

    def evaluate_text(self, text: str, max_length: int = 2048) -> float:
        """
        计算单个文本的困惑度。

        Args:
            text: 输入文本
            max_length: 最大序列长度

        Returns:
            困惑度值
        """
        pass


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    便捷的困惑度计算函数。

    Args:
        model: 语言模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        困惑度值
    """
    pass
