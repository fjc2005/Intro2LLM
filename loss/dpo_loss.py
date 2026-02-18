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
                    计算策略模型相对于参考模型的对数概率差
                    对于偏好回复和非偏好回复分别计算:
                    - 偏好对数比率 = 策略模型对偏好的对数概率 - 参考模型对偏好的对数概率
                    - 非偏好对数比率 = 策略模型对非偏好的对数概率 - 参考模型对非偏好的对数概率
                    这表示策略模型相对于参考模型对某个回复的偏好程度

            Step 2: 计算 DPO 对数几率 (logits)
                    计算偏好对数比率与非偏好对数率之差
                    然后乘以温度系数 beta 进行缩放
                    数学公式: logits = β × (偏好对数比率 - 非偏好对数比率)
                    如果策略模型给偏好回复的相对概率更高，这个值为正

            Step 3: 计算 DPO 损失
                    DPO 损失是负对数 Sigmoid 函数值
                    数学公式: loss = -log(sigmoid(logits))
                    数学上这对应于偏好分类的负对数似然
                    当 logits 越大 (偏好明显)，损失越小

                    如果启用标签平滑:
                    - 不完全相信 hard label
                    - 将标准损失与翻转标签的损失进行插值
                    - 这可以防止模型过于自信，提高泛化能力

            Step 4: 聚合批次损失
                    对批次中所有样本的损失求平均，得到标量损失

            Step 5: 计算奖励用于监控 (可选)
                    将对数比率乘以 beta 得到隐式奖励
                    - 偏好奖励 = β × 偏好对数比率
                    - 非偏好奖励 = β × 非偏好对数比率
                    - 奖励差距 = 偏好奖励 - 非偏好奖励
                    这些值用于监控训练进度，不直接参与梯度计算

            Step 6: 返回结果
                    返回包含损失和监控指标的字典

        数值稳定性:
            - 直接使用对数概率而不是概率，避免数值下溢
            - 使用稳定的对数 Sigmoid 计算，而不是先 Sigmoid 再取对数
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
                    对 logits 沿词表维度应用 log-softmax 函数
                    这将对数几率转换为对数概率，数值更稳定
                    结果形状: [batch, seq, vocab]

            Step 2: 收集目标 token 的 log 概率
                    对于每个序列位置，从 log 概率分布中提取目标 token 对应的对数概率
                    使用 gather 操作按标签索引选取
                    结果形状: [batch, seq]

            Step 3: 创建有效位置掩码
                    创建二进制掩码，标记哪些位置是有效的 (标签不为 -100)
                    -100 通常表示 padding 位置，应该被忽略
                    掩码形状: [batch, seq]

            Step 4: 应用掩码并求和
                    将目标 token 的对数概率与掩码相乘，padding 位置变为 0
                    沿序列维度求和，得到每个样本的总对数概率
                    结果形状: [batch]

            Step 5: 计算平均 (按有效 token 数)
                    计算每个样本的有效 token 数量 (掩码之和)
                    将总对数概率除以有效 token 数，得到平均对数概率
                    这确保了不同长度序列的可比性
                    结果形状: [batch]

            Step 6: 返回
                    返回每个样本的平均对数概率
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

        计算流程:
            Step 1: 计算对数概率比率
                    分别计算偏好回复和非偏好回复的策略模型与参考模型的对数概率差

            Step 2: 计算 logits
                    计算偏好对数比率与非偏好对数比率之差
                    这表示策略模型对偏好回复相对于非偏好回复的偏好程度

            Step 3: 计算 IPO 损失
                    使用均方误差损失，目标是让 logits 接近 1/tau
                    数学公式: loss = (logits - 1/τ)²
                    这迫使模型保持适中的置信度，不像 DPO 那样追求极端的 margin
        """
        pass
