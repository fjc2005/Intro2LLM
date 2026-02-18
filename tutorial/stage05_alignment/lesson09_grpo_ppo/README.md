# 课时9：GRPO与PPO强化学习

## 学习目标

1. 深入理解GRPO（组相对策略优化）的核心思想与优势
2. 掌握PPO（近端策略优化）的完整算法
3. 理解Actor-Critic架构与GAE（广义优势估计）
4. 实现GRPO和PPO Trainer

---

## 1. GRPO (Group Relative Policy Optimization)

### 1.1 GRPO核心思想

**论文**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

```
GRPO核心洞察：
- 不需要单独的Value Network（Critic）
- 通过组内采样估计基线（baseline）
- 直接使用奖励信号而非学习价值函数

优势：
- 减少模型参数量（无需Critic）
- 降低内存占用
- 训练更稳定
```

### 1.2 GRPO算法流程

```
输入：Prompt x，Policy模型 π_θ，旧Policy π_old，参考模型 π_ref

对每个Prompt执行：
1. 从旧Policy采样G个输出：{o_1, o_2, ..., o_G} ~ π_old(y|x)

2. 计算每个输出的奖励：{r_1, r_2, ..., r_G}

3. 计算组内平均奖励作为基线：
   baseline = (1/G) * Σ_i r_i

4. 计算优势函数：
   A_i = r_i - baseline

5. 计算相对log概率比：
   ratio_i = π_θ(o_i|x) / π_old(o_i|x)

6. 计算裁剪后的目标：
   loss = -E[min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i)]

7. 添加KL散度约束：
   KL = β * KL(π_θ || π_ref)
```

### 1.3 GRPO vs PPO对比

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic网络 | 需要 | 不需要 |
| 基线估计 | Value Network | 组内平均奖励 |
| 内存占用 | 高（2x模型） | 低（1x模型） |
| 计算复杂度 | 高 | 低 |
| 适用场景 | 通用RLHF | 可验证奖励任务 |

---

## 2. PPO (Proximal Policy Optimization)

### 2.1 策略梯度基础

```
策略梯度定理：
∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]

其中 A(s,a) = Q(s,a) - V(s) 是优势函数

直观理解：
- 如果动作带来正优势，增加该动作概率
- 如果动作带来负优势，减少该动作概率
```

### 2.2 Actor-Critic架构

```
Actor（策略网络）：
- 输入：状态 s
- 输出：动作概率分布 π(a|s)
- 更新：根据Critic的反馈调整策略

Critic（价值网络）：
- 输入：状态 s
- 输出：状态价值估计 V(s)
- 更新：最小化TD误差
```

### 2.3 广义优势估计 (GAE)

**目的**：平衡偏差和方差，计算优势函数

```
TD残差：
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

GAE公式：
Â_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

其中：
- γ：折扣因子（未来奖励衰减）
- λ：GAE参数（0到1之间）
  - λ=0：高偏差，低方差（Â_t = δ_t）
  - λ=1：低偏差，高方差（蒙特卡洛）
```

### 2.4 PPO裁剪目标

```
标准策略梯度问题：
- 策略更新可能过大，导致训练不稳定
- 新旧策略差异过大时，重要性采样不准确

PPO解决方案 - 裁剪目标：
L^{CLIP}(θ) = E[min(
    ratio * A,
    clip(ratio, 1-ε, 1+ε) * A
)]

其中：
- ratio = π_θ(a|s) / π_old(a|s)
- ε：裁剪参数（通常0.1或0.2）
- clip(ratio, 1-ε, 1+ε)：将ratio限制在[1-ε, 1+ε]范围

直观理解：
- 当ratio超出范围时，停止增加目标函数
- 防止策略更新过大
```

### 2.5 PPO完整损失函数

```
总损失：
L^{TOTAL} = L^{CLIP} + c1 * L^{VF} + c2 * L^{ENT}

其中：
- L^{CLIP}：裁剪策略梯度损失
- L^{VF} = (V(s) - V^{target})^2：价值函数MSE损失
- L^{ENT} = -Σ π(a|s) * log π(a|s)：熵奖励（鼓励探索）
- c1, c2：系数
```

---

## 3. 实现指引

### 3.1 loss/grpo_loss.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization损失

    特点：无需Critic网络，使用组内奖励估计基线
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
    ):
        super().__init__()
        # self.clip_epsilon = clip_epsilon
        # self.kl_beta = kl_beta
        pass

    def forward(
        self,
        policy_logprobs: torch.Tensor,      # [batch, group_size]
        old_policy_logprobs: torch.Tensor,  # [batch, group_size]
        rewards: torch.Tensor,              # [batch, group_size]
        reference_logprobs: torch.Tensor,   # [batch, group_size]
    ) -> torch.Tensor:
        """
        计算GRPO损失

        Args:
            policy_logprobs: 当前策略的log概率
            old_policy_logprobs: 旧策略的log概率
            rewards: 奖励值
            reference_logprobs: 参考模型的log概率
        """
        # Step 1: 计算组内基线（平均奖励）
        # baseline = rewards.mean(dim=-1, keepdim=True)

        # Step 2: 计算优势函数
        # advantages = rewards - baseline
        # 可选：归一化优势
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Step 3: 计算概率比率
        # log_ratios = policy_logprobs - old_policy_logprobs
        # ratios = torch.exp(log_ratios)

        # Step 4: 计算裁剪目标
        # surr1 = ratios * advantages
        # surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        # policy_loss = -torch.min(surr1, surr2).mean()

        # Step 5: 计算KL散度约束
        # kl_div = (policy_logprobs - reference_logprobs).mean()

        # Step 6: 总损失
        # loss = policy_loss + self.kl_beta * kl_div

        pass
```

### 3.2 loss/ppo_loss.py

```python
class PPOLoss(nn.Module):
    """
    Proximal Policy Optimization损失

    包含：CLIP损失 + 价值函数损失 + 熵奖励
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        super().__init__()
        # self.clip_epsilon = clip_epsilon
        # self.value_loss_coef = value_loss_coef
        # self.entropy_coef = entropy_coef
        pass

    def forward(
        self,
        policy_logprobs: torch.Tensor,      # [batch]
        old_policy_logprobs: torch.Tensor,  # [batch]
        values: torch.Tensor,               # [batch] - Critic预测
        returns: torch.Tensor,              # [batch] - 实际回报
        advantages: torch.Tensor,           # [batch] - GAE估计
        entropy: torch.Tensor,              # [batch] - 策略熵
    ) -> torch.Tensor:
        """
        计算PPO总损失
        """
        # Step 1: 计算概率比率
        # log_ratios = policy_logprobs - old_policy_logprobs
        # ratios = torch.exp(log_ratios)

        # Step 2: 计算CLIP策略损失
        # surr1 = ratios * advantages
        # surr2 = torch.clamp(ratios, 1-ε, 1+ε) * advantages
        # policy_loss = -torch.min(surr1, surr2).mean()

        # Step 3: 计算价值函数损失
        # value_loss = F.mse_loss(values, returns)

        # Step 4: 熵奖励（鼓励探索）
        # entropy_loss = -entropy.mean()

        # Step 5: 总损失
        # loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

        pass

    def compute_gae(
        self,
        rewards: torch.Tensor,      # [batch, seq_len]
        values: torch.Tensor,       # [batch, seq_len]
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)

        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len] (GAE + values)
        """
        # Step 1: 计算TD残差
        # deltas = rewards + gamma * values[:, 1:] - values[:, :-1]

        # Step 2: 反向计算累积优势
        # advantages = torch.zeros_like(values)
        # gae = 0
        # for t in reversed(range(len(rewards))):
        #     gae = deltas[t] + gamma * lam * gae
        #     advantages[t] = gae

        # Step 3: 计算returns
        # returns = advantages + values

        pass
```

### 3.3 training/grpo_trainer.py

```python
class GRPOTrainer:
    """
    GRPO训练器

    无需Critic网络，使用组内采样估计基线
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.1,
    ):
        # self.model = model  # Actor策略模型
        # self.ref_model = ref_model  # 参考模型（冻结）
        # self.group_size = group_size
        # self.grpo_loss = GRPOLoss(clip_epsilon, kl_beta)
        pass

    def generate_group_outputs(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从旧策略采样一组输出

        Returns:
            outputs: [batch, group_size, seq_len]
            log_probs: [batch, group_size]
        """
        # Step 1: 对每个prompt采样group_size个输出
        # Step 2: 计算每个输出的log概率
        pass

    def compute_rewards(
        self,
        prompts: torch.Tensor,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算奖励

        可以是：
        - 规则奖励（如数学正确性）
        - 模型奖励（如ORM/PRM）
        """
        pass

    def training_step(self, batch: dict) -> dict:
        """单步训练"""
        # Step 1: 生成组内样本
        # outputs, old_logprobs = self.generate_group_outputs(batch["prompts"])

        # Step 2: 计算奖励
        # rewards = self.compute_rewards(batch["prompts"], outputs)

        # Step 3: 计算参考模型log概率
        # with torch.no_grad():
        #     ref_logprobs = self.ref_model(outputs).log_probs

        # Step 4: 计算当前策略log概率
        # policy_logprobs = self.model(outputs).log_probs

        # Step 5: 计算GRPO损失
        # loss = self.grpo_loss(policy_logprobs, old_logprobs, rewards, ref_logprobs)

        # Step 6: 反向传播
        pass
```

### 3.4 training/ppo_trainer.py

```python
class PPOTrainer:
    """
    PPO训练器

    使用Actor-Critic架构
    """

    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: nn.Module,
        ref_model: nn.Module,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        # self.actor = actor_model  # 策略网络
        # self.critic = critic_model  # 价值网络
        # self.ref_model = ref_model  # 参考模型
        # self.ppo_loss = PPOLoss(clip_epsilon)
        # self.gamma = gamma
        # self.lam = lam
        pass

    def collect_experiences(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> dict:
        """
        收集训练经验

        Returns:
            experiences: 包含states, actions, rewards, values等
        """
        # Step 1: 使用当前策略生成序列
        # Step 2: 计算每个step的奖励
        # Step 3: 计算Critic的价值估计
        # Step 4: 计算参考模型的log概率（用于KL约束）
        pass

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用GAE计算优势和回报
        """
        # return self.ppo_loss.compute_gae(rewards, values, self.gamma, self.lam)
        pass

    def update_policy(
        self,
        experiences: dict,
        num_epochs: int = 4,
    ) -> float:
        """
        多轮更新策略

        PPO特点：使用同一批经验进行多轮更新
        """
        # for epoch in range(num_epochs):
        #     # 重新计算当前策略的logprobs和values
        #     # 计算PPO损失
        #     # 反向传播更新Actor和Critic
        pass
```

---

## 4. 关键公式总结

### GRPO损失
```
baseline = mean(rewards)
advantages = rewards - baseline
ratio = π_θ / π_old
L_GRPO = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL
```

### GAE
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
Â_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
```

### PPO CLIP
```
ratio = π_θ(a|s) / π_old(a|s)
L^CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
```

---

## 5. 常见陷阱与注意事项

1. **优势归一化**：GRPO中建议对组内优势进行归一化
2. **梯度累积**：多步经验收集后的梯度处理
3. **KL散度监控**：训练过程中监控KL散度，防止策略偏离太远
4. **经验缓冲区**：PPO需要维护经验缓冲区进行多轮更新
5. **奖励缩放**：奖励值过大或过小都会影响训练稳定性
6. **Critic更新**：PPO中Critic通常比Actor更新更多次
7. **裁剪参数**：ε=0.2是常用值，可根据任务调整

---

## 6. 课后练习

1. **手动计算GAE**：给定具体奖励和价值序列，手动计算GAE
2. **GRPO vs PPO**：分析GRPO在哪些任务上更有优势
3. **超参数调优**：实验不同group_size和clip_epsilon的影响
4. **奖励设计**：设计一个简单的规则奖励函数（如数学正确性）
5. **KL散度理解**：解释为什么需要KL散度约束
