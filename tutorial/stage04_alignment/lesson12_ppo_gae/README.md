# L12: PPO 与 GAE

## 学习目标

1. **理解** PPO 算法原理
2. **掌握** 广义优势估计 (GAE)
3. **了解** PPO 训练流程

---

## 理论背景

### 1. PPO 算法概述

#### 1.1 背景

PPO (Proximal Policy Optimization) 是 OpenAI 提出的强化学习算法，广泛用于 RLHF 的策略优化阶段。

#### 1.2 核心思想

PPO 通过限制策略更新的幅度，避免策略变化过大导致训练不稳定。

#### 1.3 目标函数

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}\left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

其中:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是优势估计
- $\epsilon$ 通常设为 0.2

### 2. 策略梯度基础

#### 2.1 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$$

#### 2.2 REINFORCE 算法

基本策略梯度方法:
1. 使用当前策略采样轨迹
2. 计算每步的回报
3. 更新策略以增加高回报动作的概率

### 3. 价值函数与优势函数

#### 3.1 价值函数

$$V(s) = \mathbb{E}_{\pi}[R_t | s_t = s]$$

#### 3.2 优势函数

$$A(s, a) = Q(s, a) - V(s)$$

优势表示在状态 s 下采取动作 a 相对于平均水平的好坏。

### 4. 广义优势估计 (GAE)

#### 4.1 TD 误差

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

#### 4.2 λ-return

GAE 使用 λ-return 来平衡偏差和方差:

$$A_t^{GAE} = (1-\lambda)(\delta_t + \gamma\lambda\delta_{t+1} + \gamma^2\lambda^2\delta_{t+2} + ...)$$

**展开形式**:
$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

#### 4.3 超参数 λ

- λ = 0: 只考虑一步 TD 误差 (高偏差，低方差)
- λ = 1: 考虑完整轨迹回报 (低偏差，高方差)
- λ 接近 0: 接近 Value Function 方法
- λ 接近 1: 接近 REINFORCE 方法

### 5. PPO 训练流程

```
1. SFT 阶段: 训练基础模型
2. Reward Model 阶段: 训练奖励模型
3. PPO 阶段:
   a. 使用当前策略采样响应
   b. 计算奖励 (RM 分数 + KL 惩罚)
   c. 使用 GAE 计算优势
   d. 优化 PPO 目标
```

---

## 代码实现

### 项目结构

```
model/
└── reward_model.py  # 奖励模型

loss/
└── ppo.py          # PPO 损失

training/
└── ppo_trainer.py  # PPO 训练器
```

---

## 实践练习

### 练习 1: 实现 GAE

打开 `loss/ppo.py`，实现广义优势估计:

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (GAE)。

    公式:
    A_t = sum_{l=0}^{inf} (gamma * lam)^l * delta_{t+l}

    其中 delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        rewards: 奖励序列 [seq_len]
        values: 价值估计序列 [seq_len + 1]
        dones: 终止标志序列 [seq_len]
        gamma: 折扣因子
        lam: GAE lambda 参数 (控制偏差-方差权衡)

    实现思路:
    1. 从后向前遍历
    2. 计算 TD 误差 delta
    3. 使用 GAE 公式累积优势
    4. 反向计算，确保每个位置的优势正确

    返回:
        advantages: 优势估计 [seq_len]
        returns: 价值目标 (advantages + values) [seq_len]
    """
    pass
```

### 练习 2: 实现 PPO 损失

```python
def compute_ppo_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2):
    """
    计算 PPO CLIP 损失。

    公式:
    L = min(r * A, clip(r, 1-e, 1+e) * A)

    其中 r = exp(log_prob - old_log_prob)

    Args:
        log_probs: 当前策略的对数概率
        old_log_probs: 旧策略的对数概率
        advantages: 优势估计
        clip_epsilon: 裁剪边界 epsilon

    返回:
        PPO 损失值
    """
    # 实现:
    # 1. 计算重要性采样比率 r
    # 2. 对 r 进行裁剪
    # 3. 计算裁剪和未裁剪损失的最小值
    # 4. 使用 advantages 加权
    pass
```

### 练习 3: 实现 PPO 训练器

```python
class PPOTrainer:
    def __init__(self, model, reward_model, ref_model, config):
        """
        PPO 训练器。

        组成部分:
        - policy_model: 要优化的语言模型
        - reward_model: 奖励模型
        - reference_model: 参考模型 (用于 KL 惩罚)
        """
        pass

    def generate_responses(self, prompts):
        """
        使用当前策略生成响应。

        Args:
            prompts: 提示列表

        返回:
            生成的响应 token IDs
        """
        pass

    def compute_rewards(self, responses, prompts):
        """
        计算奖励。

        奖励构成:
        - reward_model 的分数
        - KL 惩罚 (防止偏离参考模型太远)

        Args:
            responses: 生成的响应
            prompts: 对应的提示

        返回:
            奖励序列
        """
        pass

    def training_step(self, batch):
        """
        执行单步 PPO 训练。

        流程:
        1. 生成响应
        2. 计算奖励
        3. 计算价值估计
        4. 使用 GAE 计算优势
        5. 优化 PPO 目标
        """
        pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage04_alignment/lesson12_ppo_gae/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage04_alignment/lesson12_ppo_gae/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **PPO 论文**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **GAE 论文**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- **RLHF 流程**: 了解 InstructGPT 的完整训练流程
