"""
PPO (Proximal Policy Optimization) 训练器模块
实现 RLHF 的完整 PPO 训练流程。

PPO 训练涉及四个模型:
1. Actor (策略模型): 生成回复，需要训练
2. Critic (价值模型): 估计状态价值，需要训练
3. Reward Model: 给回复打分，冻结
4. Reference Model: 提供 KL 约束，冻结

训练循环:
1. 使用当前策略生成样本 (Rollout)
2. 计算奖励和优势 (Reward + GAE)
3. 多次 PPO 更新 (使用同一批数据)
4. 重复

这是 RLHF 的标准实现，但训练较复杂。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple, List
import copy

from .trainer import Trainer
from loss.ppo_loss import PPOLoss, GAE


class PPOTrainer(Trainer):
    """
    PPO 训练器

    实现 RLHF 的完整 PPO 训练流程。

    需要管理的模型:
    - actor: 策略模型 (π_θ)
    - critic: 价值模型 (V_φ)
    - reward_model: 奖励模型 (R)，冻结
    - ref_model: 参考模型 (π_ref)，冻结
    """

    def __init__(
        self,
        model: nn.Module,
        critic_model: nn.Module,
        reward_model: nn.Module,
        ref_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        初始化 PPO 训练器。

        Args:
            model: Actor 策略模型
            critic_model: Critic 价值模型
            reward_model: 奖励模型 (冻结)
            ref_model: 参考模型 (冻结)
            train_dataloader: 训练数据 (prompts)
            val_dataloader: 验证数据
            actor_optimizer: Actor 优化器
            critic_optimizer: Critic 优化器
            lr_scheduler: 学习率调度器
            config: 训练配置
            device: 训练设备

        配置选项:
            - ppo_epochs: 每批数据的 PPO 更新轮数
            - num_rollouts: 每次生成的样本数
            - max_new_tokens: 最大生成长度
            - gamma: GAE 折扣因子
            - lam: GAE lambda 参数
        """
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=actor_optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device,
        )
        # 初始化 critic 和优化器
        # 初始化奖励模型和参考模型
        # 创建 PPO Loss 和 GAE
        pass

    def rollout(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        使用当前策略生成样本。

        Args:
            prompts: 输入 prompts，[batch_size, prompt_len]
            max_new_tokens: 最大生成长度
            temperature: 采样温度

        Returns:
            包含以下字段的字典:
                - input_ids: 完整序列 (prompt + response)
                - log_probs: 策略模型的 log 概率
                - ref_log_probs: 参考模型的 log 概率
                - rewards: 奖励模型打分
                - values: 价值模型估计
                - masks: 注意力掩码

        流程:
            Step 1: 生成回复
                    with torch.no_grad():
                        responses = actor.generate(prompts, max_new_tokens)

            Step 2: 计算策略 log probs
                    logits = actor(responses).logits
                    log_probs = compute_log_probs(logits, responses)

            Step 3: 计算参考模型 log probs
                    with torch.no_grad():
                        ref_logits = ref_model(responses).logits
                        ref_log_probs = compute_log_probs(ref_logits, responses)

            Step 4: 计算奖励
                    with torch.no_grad():
                        rewards = reward_model(responses)

            Step 5: 计算价值估计
                    with torch.no_grad():
                        values = critic_model(responses)

            Step 6: 返回所有数据
        """
        pass

    def compute_advantages_and_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 GAE 计算优势和回报。

        Args:
            rewards: 奖励，[batch, seq_len]
            values: 价值估计，[batch, seq_len]
            masks: 掩码，[batch, seq_len]

        Returns:
            (advantages, returns)

        使用 GAE 类计算:
            gae = GAE(gamma, lambda)
            advantages, returns = gae.compute_advantages(rewards, values, masks)
        """
        pass

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行 PPO 训练步骤。

        PPO 的一个 "step" 包含:
        1. 生成样本 (rollout)
        2. 计算优势和回报
        3. 多次 PPO 更新 (ppo_epochs 次)

        Args:
            batch: 批次数据，包含 prompts

        Returns:
            训练指标

        训练流程:

            Step 1: 生成样本
                    prompts = batch["prompts"]
                    rollout_data = self.rollout(prompts)

            Step 2: 计算优势和回报
                    advantages, returns = self.compute_advantages_and_returns(
                        rollout_data["rewards"],
                        rollout_data["values"],
                        rollout_data["masks"],
                    )

            Step 3: PPO 更新循环 (多次)
                    for ppo_epoch in range(ppo_epochs):
                        # 重新计算当前策略的 log probs 和 values
                        logits = actor(rollout_data["input_ids"]).logits
                        log_probs = compute_log_probs(logits, ...)

                        values = critic(rollout_data["input_ids"])

                        # 计算 PPO 损失
                        loss_dict = ppo_loss(
                            logits=logits,
                            values=values,
                            log_probs=log_probs,
                            old_log_probs=rollout_data["log_probs"],
                            old_values=rollout_data["values"],
                            returns=returns,
                            advantages=advantages,
                            ref_log_probs=rollout_data["ref_log_probs"],
                        )

                        # 更新 Actor
                        actor_optimizer.zero_grad()
                        loss_dict["total_loss"].backward()
                        actor_optimizer.step()

                        # 更新 Critic (可选)
                        if update_critic:
                            critic_optimizer.zero_grad()
                            loss_dict["value_loss"].backward()
                            critic_optimizer.step()

            Step 4: 返回平均指标
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估。

        评估指标:
        - 平均奖励
        - 策略 KL 散度
        - 生成质量 (如困惑度)
        """
        pass
