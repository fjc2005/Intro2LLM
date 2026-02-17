"""
DPO (Direct Preference Optimization) 损失模块
实现基于偏好数据的直接优化。

DPO 核心思想:
不同于 RLHF 需要训练奖励模型 + PPO，DPO 直接从偏好数据学习策略。
将强化学习目标转化为监督学习形式，更简单高效。

理论基础:
- Bradley-Terry 模型: 建模成对偏好
- KL 散度约束: 防止策略偏离参考模型太远

相比 RLHF-PPO:
- 更简单: 无需奖励模型、无需价值模型、无需在线采样
- 更稳定: 没有 PPO 的训练不稳定性
- 更高效: 离线训练，数据复用率高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DPOLoss(nn.Module):
    """
    DPO (Direct Preference Optimization) 损失

    论文: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
          (Rafailov et al., 2023)
    链接: https://arxiv.org/abs/2305.18290

    核心公式:
        L_DPO(π_θ; π_ref) = -E[(x,y_w,y_l)~D] [
            log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x))
                    - β * log(π_θ(y_l|x) / π_ref(y_l|x)))
        ]

    其中:
        - π_θ: 当前策略 (待训练模型)
        - π_ref: 参考策略 (通常是 SFT 模型，冻结参数)
        - x: 输入 prompt
        - y_w: 偏好回复 (chosen，更好的)
        - y_l: 非偏好回复 (rejected，更差的)
        - β: 温度系数，控制 KL 散度惩罚强度
        - σ: sigmoid 函数

    直观理解:
    - 增加偏好回复的相对对数概率
    - 减少非偏好回复的相对对数概率
    - 相对是指相对于参考模型

    为什么有效:
    1. 将 RL 问题转化为分类问题 (偏好分类)
    2. 直接优化策略，无需显式奖励模型
    3. KL 约束隐式包含在损失中
    """

    def __init__(
        self,
        config = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        初始化 DPO 损失。

        Args:
            config: 模型配置 (可选)
            beta: 温度系数，默认 0.1
                  较大的 β: 更强的 KL 约束，策略更接近参考模型
                  较小的 β: 更激进的优化，可能偏离参考模型
            label_smoothing: 标签平滑系数，默认 0

        β 的选择建议:
            - 0.1: 标准设置，大多数情况适用
            - 0.5: 更强的正则化
            - 0.01: 更激进的优化
        """
        super().__init__()
        # 保存 beta 参数
        # label_smoothing 用于缓解过拟合
        pass

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 DPO 损失。

        Args:
            policy_chosen_logps: 策略模型对偏好回复的对数概率
                                [batch_size]，每个元素是该样本的平均 log p
            policy_rejected_logps: 策略模型对非偏好回复的对数概率
                                  [batch_size]
            reference_chosen_logps: 参考模型对偏好回复的对数概率
                                   [batch_size]
            reference_rejected_logps: 参考模型对非偏好回复的对数概率
                                     [batch_size]

        Returns:
            包含以下字段的字典:
                - loss: 标量损失值
                - chosen_rewards: 偏好回复的奖励 (用于监控)
                - rejected_rewards: 非偏好回复的奖励 (用于监控)
                - reward_margin: 奖励差距 (chosen - rejected)

        计算步骤:

            Step 1: 计算对数比率 (log ratio)
                    策略模型相对于参考模型的对数概率比

                    chosen_logratios = policy_chosen_logps - reference_chosen_logps
                    rejected_logratios = policy_rejected_logps - reference_rejected_logps

                    形状: [batch_size]

            Step 2: 计算 DPO 对数几率 (logits)
                    logits = beta * (chosen_logratios - rejected_logratios)
                           = beta * (policy_chosen_logps - reference_chosen_logps
                                    - policy_rejected_logps + reference_rejected_logps)

                    形状: [batch_size]

                    直观: 如果策略模型给偏好回复的相对概率更高，logits 为正

            Step 3: 计算损失
                    # DPO 损失是负对数似然 (偏好 > 非偏好)
                    loss = -F.logsigmoid(logits)

                    如果 label_smoothing > 0:
                        # 标签平滑: 不完全相信标签
                        loss = (1 - smoothing) * loss + smoothing * F.logsigmoid(-logits)

                    形状: [batch_size]

            Step 4: 聚合批次损失
                    loss = loss.mean()

            Step 5: 计算奖励用于监控 (可选)
                    chosen_rewards = beta * chosen_logratios
                    rejected_rewards = beta * rejected_logratios
                    reward_margin = chosen_rewards - rejected_rewards

            Step 6: 返回结果
                    return {
                        "loss": loss,
                        "chosen_rewards": chosen_rewards.mean(),
                        "rejected_rewards": rejected_rewards.mean(),
                        "reward_margin": reward_margin.mean(),
                    }

        数值稳定性:
            - 使用 log probabilities 避免下溢
            - 使用 logsigmoid 而不是 log(sigmoid(x)) 更稳定
        """
        pass

    def compute_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算序列的对数概率。

        Args:
            logits: 模型输出 logits，[batch_size, seq_len, vocab_size]
            labels: 目标标签，[batch_size, seq_len]
                   忽略标签为 -100 的位置

        Returns:
            每个样本的平均对数概率，[batch_size]

        计算步骤:
            Step 1: 计算每个位置的 log 概率
                    log_probs = F.log_softmax(logits, dim=-1)
                    形状: [batch, seq, vocab]

            Step 2: 收集目标 token 的 log 概率
                    # 使用 gather 获取对应位置的 log 概率
                    token_log_probs = log_probs.gather(
                        dim=-1,
                        index=labels.unsqueeze(-1)
                    ).squeeze(-1)
                    形状: [batch, seq]

            Step 3: 创建有效位置掩码
                    loss_mask = (labels != -100).float()
                    形状: [batch, seq]

            Step 4: 应用掩码并求和
                    # 只计算有效位置的 log 概率
                    masked_log_probs = token_log_probs * loss_mask
                    sequence_log_probs = masked_log_probs.sum(dim=-1)
                    形状: [batch]

            Step 5: 计算平均 (按有效 token 数)
                    valid_lengths = loss_mask.sum(dim=-1)
                    average_log_probs = sequence_log_probs / valid_lengths
                    形状: [batch]

            Step 6: 返回
                    return average_log_probs
        """
        pass


class IPO_Loss(nn.Module):
    """
    IPO (Identity Preference Optimization) 损失

    论文: "A General Theoretical Paradigm to Understand Learning from Human Preferences"
          (Azar et al., 2023)

    DPO 的变体，使用不同的目标函数。
    与 DPO 区别:
    - DPO 最大化偏好概率的 margin
    - IPO 最小化预测与实际偏好之间的差距 (均方误差)

    公式:
        L_IPO = (log(π(y_w|x) / π(y_l|x)) - τ^-1)^2

    其中 τ 是温度参数。

    IPO 优势:
    - 在某些情况下比 DPO 更稳定
    - 对异常值不敏感
    """

    def __init__(self, config=None, tau: float = 0.1):
        """
        初始化 IPO 损失。

        Args:
            config: 模型配置
            tau: 温度参数
        """
        super().__init__()
        pass

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 IPO 损失。

        公式:
            chosen_ratio = policy_chosen_logps - reference_chosen_logps
            rejected_ratio = policy_rejected_logps - reference_rejected_logps
            logits = chosen_ratio - rejected_ratio
            loss = (logits - 1/tau)^2
        """
        pass
