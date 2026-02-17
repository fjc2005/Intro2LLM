"""
DPO (Direct Preference Optimization) 训练器模块
用于基于偏好数据的模型对齐。

DPO 训练流程:
1. 准备偏好数据: (prompt, chosen, rejected) 三元组
2. 使用参考模型 (SFT 模型) 计算参考对数概率
3. 训练策略模型，优化 DPO 损失
4. 监控奖励和 KL 散度

参考模型:
- 通常是经过 SFT 的模型
- 参数冻结，不更新
- 提供策略优化的参考点
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple
import copy

from .trainer import Trainer
from loss.dpo_loss import DPOLoss


class DPOTrainer(Trainer):
    """
    DPO 训练器

    实现 DPO 论文中的直接偏好优化算法。

    与 SFT 训练器的区别:
    - 数据是偏好对 (chosen vs rejected)
    - 需要参考模型
    - 损失函数是 DPO loss 而非交叉熵
    - 需要计算和存储两组 log probabilities
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        初始化 DPO 训练器。

        Args:
            model: 策略模型 (待训练)
            ref_model: 参考模型 (冻结)，通常是 SFT 模型
            train_dataloader: 偏好数据加载器 (DPODataset)
            val_dataloader: 验证数据
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            config: 训练配置，包含:
                - beta: DPO 温度系数，默认 0.1
                - label_smoothing: 标签平滑
                - reference_free: 是否使用无参考模型模式
            device: 训练设备

        参考模型处理:
            Step 1: 保存参考模型
            Step 2: 将参考模型设为评估模式
            Step 3: 冻结参考模型参数 (requires_grad = False)
            Step 4: 将参考模型移至设备

        内存优化选项:
            - 参考模型可以使用 fp16/bf16
            - 参考模型可以 offload 到 CPU
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
        # 初始化参考模型
        # 创建 DPO Loss
        pass

    def _prepare_reference_model(self, ref_model: nn.Module) -> nn.Module:
        """
        准备参考模型。

        Args:
            ref_model: 原始参考模型

        Returns:
            处理后的参考模型 (冻结、eval 模式)

        处理步骤:
            Step 1: 复制模型 (如果需要)
            Step 2: 设为评估模式
                    ref_model.eval()

            Step 3: 冻结所有参数
                    for param in ref_model.parameters():
                        param.requires_grad = False

            Step 4: 移至设备
                    ref_model.to(device)

            Step 5: 返回
        """
        pass

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对 chosen 和 rejected 样本进行拼接前向传播。

        为了提高效率，将 chosen 和 rejected 拼接成一个批次同时前向传播。

        Args:
            model: 要前向传播的模型 (策略或参考)
            batch: 批次数据，包含:
                - chosen_input_ids: [batch_size, chosen_len]
                - chosen_attention_mask: [batch_size, chosen_len]
                - rejected_input_ids: [batch_size, rejected_len]
                - rejected_attention_mask: [batch_size, rejected_len]

        Returns:
            (chosen_logits, rejected_logits, chosen_labels, rejected_labels)

        流程:
            Step 1: 拼接输入
                    concatenated_input_ids = cat([chosen_input_ids, rejected_input_ids], dim=0)
                    concatenated_attention_mask = cat([chosen_attention_mask, rejected_attention_mask], dim=0)
                    形状: [2*batch_size, max_seq_len]

            Step 2: 前向传播
                    outputs = model(
                        input_ids=concatenated_input_ids,
                        attention_mask=concatenated_attention_mask,
                    )
                    logits = outputs.logits
                    形状: [2*batch_size, seq_len, vocab_size]

            Step 3: 分离 chosen 和 rejected
                    chosen_logits = logits[:batch_size]
                    rejected_logits = logits[batch_size:]

            Step 4: 返回
        """
        pass

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行单步 DPO 训练。

        Args:
            batch: 批次数据，包含 chosen 和 rejected 的输入

        Returns:
            训练指标字典

        训练流程:

            Step 1: 策略模型前向传播
                    policy_chosen_logits, policy_rejected_logits = \
                        self.concatenated_forward(self.model, batch)

            Step 2: 计算策略模型的 log probabilities
                    policy_chosen_logps = dpo_loss.compute_log_probs(
                        policy_chosen_logits, batch["chosen_labels"]
                    )
                    policy_rejected_logps = dpo_loss.compute_log_probs(
                        policy_rejected_logits, batch["rejected_labels"]
                    )

            Step 3: 参考模型前向传播 (无梯度)
                    with torch.no_grad():
                        ref_chosen_logits, ref_rejected_logits = \
                            self.concatenated_forward(self.ref_model, batch)

            Step 4: 计算参考模型的 log probabilities
                    ref_chosen_logps = dpo_loss.compute_log_probs(
                        ref_chosen_logits, batch["chosen_labels"]
                    )
                    ref_rejected_logps = dpo_loss.compute_log_probs(
                        ref_rejected_logits, batch["rejected_labels"]
                    )

            Step 5: 计算 DPO 损失
                    loss_dict = self.dpo_loss(
                        policy_chosen_logps=policy_chosen_logps,
                        policy_rejected_logps=policy_rejected_logps,
                        reference_chosen_logps=ref_chosen_logps,
                        reference_rejected_logps=ref_rejected_logps,
                    )
                    loss = loss_dict["loss"]

            Step 6: 反向传播和优化
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    # 优化器步骤
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            Step 7: 收集指标
                    metrics = {
                        "loss": loss.item(),
                        "chosen_rewards": loss_dict["chosen_rewards"].item(),
                        "rejected_rewards": loss_dict["rejected_rewards"].item(),
                        "reward_margin": loss_dict["reward_margin"].item(),
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    }

            Step 8: 返回指标

        内存优化:
            - 参考模型前向在 no_grad 环境下
            - 使用 concatenated_forward 减少 CUDA kernel 启动开销
        """
        pass

    def compute_log_probs_with_mask(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算考虑 mask 的序列对数概率。

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]，-100 表示忽略

        Returns:
            每个序列的平均对数概率，[batch_size]
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估 DPO 模型。

        评估指标:
        - loss: 平均 DPO 损失
        - accuracy: 策略模型正确排序偏好的比例
                   (policy_chosen_logps > policy_rejected_logps)
        - reward_margin: 平均奖励差距
        """
        pass

    def save_checkpoint(self, output_dir: str, epoch: int, step: int):
        """
        保存检查点。

        DPO 特有:
        - 只保存策略模型，参考模型不变
        - 可以保存参考模型的配置信息
        """
        pass
