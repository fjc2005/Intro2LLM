"""
GRPO (Group Relative Preference Optimization) 训练器模块
实现组内相对偏好优化的完整训练流程。

GRPO 训练流程:
1. 对每个 prompt 生成多个候选回复
2. 使用奖励模型或规则给每个回复打分
3. 在组内归一化奖励
4. 优化策略模型以偏好高分回复

与 PPO 的区别:
- GRPO 使用离线生成的样本组
- 组内归一化消除奖励尺度影响
- 无需价值模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List

from .trainer import Trainer
from loss.grpo_loss import GRPOLoss


class GRPOTrainer(Trainer):
    """
    GRPO 训练器

    实现 GRPO 算法的完整训练流程。

    需要:
    - 策略模型 (待训练)
    - 参考模型 (冻结)
    - 奖励模型或奖励函数
    - 偏好数据 (prompt 和生成的回复组)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        初始化 GRPO 训练器。

        Args:
            model: 策略模型 (actor)
            ref_model: 参考模型 (冻结)
            reward_model: 奖励模型或奖励函数
            train_dataloader: 训练数据
            val_dataloader: 验证数据
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            config: 训练配置
            device: 训练设备

        配置选项:
            - num_generations: 每个 prompt 生成的回复数
            - group_size: 组大小
            - temperature: 生成温度
            - max_new_tokens: 最大生成长度
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
        # 初始化参考模型和奖励模型
        # 创建 GRPO Loss
        pass

    def generate_responses(
        self,
        prompts: torch.Tensor,
        num_generations: int,
        temperature: float = 1.0,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        """
        为每个 prompt 生成多个回复。

        Args:
            prompts: 输入 prompts，[batch_size, prompt_len]
            num_generations: 每个 prompt 生成的回复数
            temperature: 采样温度
            max_new_tokens: 最大生成 token 数

        Returns:
            生成的回复，[batch_size, num_generations, total_len]

        流程:
            Step 1: 重复 prompts
                    prompts_repeated = prompts.repeat_interleave(num_generations, dim=0)
                    形状: [batch_size * num_generations, prompt_len]

            Step 2: 生成回复
                    with torch.no_grad():
                        responses = model.generate(
                            prompts_repeated,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True
                        )
                    形状: [batch_size * num_generations, prompt_len + response_len]

            Step 3: reshape
                    responses = responses.view(batch_size, num_generations, -1)

            Step 4: 返回
        """
        pass

    def compute_rewards(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算奖励。

        Args:
            prompts: prompts，[batch_size, prompt_len]
            responses: 回复，[batch_size, num_generations, seq_len]

        Returns:
            奖励值，[batch_size, num_generations]

        可以使用:
        - 奖励模型打分
        - 规则验证 (如代码编译通过)
        - 答案匹配 (如数学题答案正确)
        """
        pass

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算模型对序列的对数概率。

        Args:
            model: 模型 (策略或参考)
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码

        Returns:
            每个序列的平均对数概率
        """
        pass

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行单步 GRPO 训练。

        与 DPO 的区别:
        - DPO 使用预先生成的偏好对
        - GRPO 可以在训练时在线生成回复组

        Args:
            batch: 批次数据，包含 prompts

        Returns:
            训练指标

        训练流程 (离线版本，使用预生成数据):

            Step 1: 获取 batch
                    prompts = batch["prompts"]  # [batch, prompt_len]
                    responses = batch["responses"]  # [batch, num_samples, seq_len]
                    rewards = batch["rewards"]  # [batch, num_samples]

            Step 2: 计算策略模型的 log probs
                    # 对每个样本计算
                    policy_log_probs = self.compute_log_probs(
                        self.model, prompts_and_responses
                    )
                    # reshape 为 [batch, num_samples]

            Step 3: 计算参考模型的 log probs
                    with torch.no_grad():
                        ref_log_probs = self.compute_log_probs(
                            self.ref_model, prompts_and_responses
                        )

            Step 4: 计算 GRPO 损失
                    loss_dict = self.grpo_loss(
                        policy_log_probs=policy_log_probs,
                        reference_log_probs=ref_log_probs,
                        rewards=rewards,
                    )

            Step 5: 反向传播和优化
                    loss = loss_dict["loss"]
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            Step 6: 返回指标

        在线版本 (训练时生成):
            - 在 Step 1 使用 generate_responses() 实时生成
            - 使用 compute_rewards() 打分
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估。

        评估指标:
        - 平均奖励
        - 策略与参考模型的 KL 散度
        - 组内奖励方差
        """
        pass
