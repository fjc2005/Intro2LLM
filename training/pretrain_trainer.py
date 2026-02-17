"""
预训练训练器模块
用于因果语言模型的无监督预训练。

预训练任务:
- 给定一段文本，预测下一个 token
- 使用因果掩码确保模型只能看到过去的上下文
- 标准语言建模目标: maximize P(x_t | x_<t)

训练特点:
- 数据量大 (TB 级别)
- 训练时间长 (数天到数周)
- 需要分布式训练
- 需要混合精度加速
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Optional, Any
import torch.nn.functional as F

from .trainer import Trainer


class PretrainTrainer(Trainer):
    """
    预训练训练器

    实现标准的因果语言模型预训练流程。

    数据格式:
    - 输入: input_ids, attention_mask, labels
    - 任务: next token prediction

    与基类的主要区别:
    - 实现了具体的 train_step() 方法
    - 支持梯度累积
    - 支持混合精度训练
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        初始化预训练训练器。

        Args:
            model: 因果语言模型 (CausalLM)
            train_dataloader: 训练数据 (PretrainDataset)
            val_dataloader: 验证数据
            optimizer: 优化器 (默认 AdamW)
            lr_scheduler: 学习率调度器
            config: 训练配置
            device: 训练设备

        额外配置选项:
            - mixed_precision: 是否使用混合精度 (fp16/bf16)
            - gradient_accumulation_steps: 梯度累积步数
            - max_grad_norm: 梯度裁剪阈值
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
        执行单步预训练。

        Args:
            batch: 批次数据，包含:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - labels: [batch_size, seq_len]

        Returns:
            训练指标字典，包含 loss、learning_rate 等

        训练流程:

            Step 1: 模型前向传播
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    # outputs 包含 logits 和 loss

            Step 2: 获取损失
                    loss = outputs.loss  # 标量张量

                    如果 outputs 没有 loss:
                    logits = outputs.logits  # [batch, seq, vocab]
                    # 手动计算交叉熵损失
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100
                    )

            Step 3: 梯度累积处理
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                        # 累积多步后再更新参数

            Step 4: 反向传播
                    loss.backward()

            Step 5: 梯度裁剪 (可选)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm
                        )

            Step 6: 优化器步骤 (如果达到累积步数)
                    if (self.global_step + 1) % gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

            Step 7: 收集指标
                    metrics = {
                        "loss": loss.item() * gradient_accumulation_steps,
                        "perplexity": torch.exp(loss).item(),
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    }

            Step 8: 返回指标
                    return metrics

        混合精度训练 (如果使用):
            with torch.cuda.amp.autocast():
                outputs = model(...)
                loss = outputs.loss

            # 缩放梯度防止下溢
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        """
        pass

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算预训练损失。

        Args:
            logits: 模型输出，[batch_size, seq_len, vocab_size]
            labels: 目标标签，[batch_size, seq_len]

        Returns:
            标量损失值

        计算:
            loss = CrossEntropyLoss(logits.flatten(0, 1), labels.flatten())
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估。

        评估指标:
        - loss: 平均损失
        - perplexity: 困惑度，exp(loss)

        流程:
            model.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = self._move_to_device(batch)
                    outputs = model(**batch)
                    total_loss += outputs.loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss))

            return {"loss": avg_loss, "perplexity": perplexity.item()}
        """
        pass

    def save_model(self, output_dir: str, save_optimizer: bool = True):
        """
        保存训练好的模型。

        Args:
            output_dir: 输出目录
            save_optimizer: 是否同时保存优化器状态

        保存内容:
            - pytorch_model.bin: 模型权重
            - config.json: 模型配置
            - tokenizer.json: 分词器 (如果有)
            - optimizer.pt: 优化器状态 (可选)
            - scheduler.pt: 调度器状态 (可选)
        """
        pass
