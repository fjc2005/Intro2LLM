"""
监督微调训练器模块
用于指令微调 (Instruction Tuning / SFT)。

SFT 与预训练的区别:
1. 数据格式: 指令-回复对 vs 纯文本
2. Loss Mask: 只计算回复部分的损失，mask 指令部分
3. 训练目标: 学习遵循指令生成高质量回复

训练策略:
- 学习率通常比预训练小 (1e-5 ~ 1e-6)
- 训练步数少 (几个 epoch)
- 可以使用 LoRA 等高效微调方法
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

from .trainer import Trainer


class SFTTrainer(Trainer):
    """
    监督微调训练器

    实现指令微调的训练流程。

    与预训练的主要区别:
    - 数据是 (instruction, output) 对
    - 只有 output 部分参与损失计算
    - 通常使用较小的学习率
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        初始化 SFT 训练器。

        Args: 同 Trainer

        SFT 特有配置:
            - label_smoothing: 标签平滑，防止过拟合
            - response_only_loss: 是否只计算回复部分的损失 (通常为 True)
        """
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device,
        )
        pass

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行单步 SFT 训练。

        Args:
            batch: 批次数据，包含:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - labels: [batch_size, seq_len] (包含 -100 mask)

        Returns:
            训练指标字典

        流程:
            Step 1: 前向传播
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

            Step 2: 获取损失
                    loss = outputs.loss
                    # 注意: labels 中的 -100 会被 CrossEntropyLoss 自动忽略

            Step 3-6: 反向传播、梯度裁剪、优化器步骤
                      (同 PretrainTrainer)

            Step 7: 收集指标
                    # 可以额外计算有效 token 数量
                    valid_tokens = (labels != -100).sum().item()
                    metrics = {
                        "loss": loss.item(),
                        "valid_tokens": valid_tokens,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    }

        数据格式示例:
            input_ids:  [prompt tokens... response tokens... eos]
            labels:     [-100, -100, ..., token_id, token_id, ..., eos_id]
                        ↑ mask (不计算损失) ↑ 计算损失
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估 SFT 模型。

        评估指标:
        - loss: 平均损失
        - 可选: 生成质量评估 (BLEU、ROUGE 等)

        生成评估:
        1. 从验证集中采样一些样本
        2. 使用模型生成回复
        3. 与参考回复比较
        """
        pass

    def generate_sample(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        生成单个样本，用于监控训练效果。

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        pass
