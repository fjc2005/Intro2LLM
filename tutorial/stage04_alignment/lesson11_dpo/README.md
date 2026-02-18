# L11: DPO 与 IPO

## 学习目标

1. **理解** 直接偏好优化 (DPO) 的数学原理
2. **掌握** DPO 损失函数实现
3. **了解** IPO 改进

---

## 理论背景

### 1. 从 RLHF 到 DPO

#### 1.1 RLHF 概述

传统的 RLHF (Reinforcement Learning from Human Feedback) 流程:

1. **预训练模型**: 在大规模文本上预训练
2. **SFT**: 在指令数据上微调
3. **Reward Model**: 训练奖励模型预测人类偏好
4. **PPO**: 使用 PPO 算法优化模型

**问题**: RLHF 需要训练奖励模型和 PPO 优化，流程复杂，计算开销大。

#### 1.2 DPO 的核心思想

DPO (Direct Preference Optimization) 直接使用偏好数据优化，无需显式的奖励模型和强化学习。

### 2. DPO 数学推导

#### 2.1 偏好数据的定义

给定一个提示 x 和两个响应 y_w (偏好) 和 y_l (不偏好)，满足:
$$P(y_w > y_l | x) > P(y_l > y_w | x)$$

#### 2.2 奖励函数到概率

使用 Bradley-Terry 模型，偏好概率可以表示为:
$$P(y_w > y_l | x) = \sigma(f(x, y_w) - f(x, y_l))$$

其中:
- $\sigma$ 是 sigmoid 函数
- $f(x, y)$ 是模型对响应的"评分"

#### 2.3 损失函数推导

最大化偏好概率等价于最小化以下损失:
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(f(x, y_w) - f(x, y_l)) \right]$$

**简化形式**:
$$\mathcal{L}_{DPO} = -\log \sigma(\Delta f)$$

其中 $\Delta f = f(x, y_w) - f(x, y_l)$。

#### 2.4 实际实现

在实际实现中，$f(x, y)$ 使用模型的对数似然:
$$f(x, y) = \frac{1}{|y|} \log P_\theta(y | x)$$

**DPO 损失**:
```python
def dpo_loss(policy_logps, reference_logps, beta=0.1):
    """
    计算 DPO 损失。

    Args:
        policy_logps: 策略模型的对数似然
        reference_logps: 参考模型的对数似然
        beta: 温度参数，控制与参考模型的偏离程度
    """
    # 计算优势: policy - reference
    advantages = policy_logps - reference_logps

    # 使用 sigmoid 和对数损失
    loss = -F.logsigmoid(advantages / beta)
    return loss.mean()
```

### 3. σ (Sigmoid) 函数

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**性质**:
- $\sigma(0) = 0.5$
- $\sigma(x) + \sigma(-x) = 1$
- 当 x > 0 时，σ(x) > 0.5

### 4. IPO (Identity Preference Optimization)

#### 4.1 DPO 的问题

DPO 假设偏好数据是确定性的，即一个响应严格优于另一个。实际数据可能存在噪声。

#### 4.2 IPO 改进

IPO 使用更宽松的目标:
$$\mathcal{L}_{IPO} = \mathbb{E}\left[ (\tau(y_w) - \tau(y_l) - \frac{1}{2})^2 \right]$$

其中 $\tau(y) = \frac{1}{|y|} \log P_\theta(y | x)$

---

## 代码实现

### 项目结构

```
data/
└── dpo_dataset.py  # DPO 数据集

loss/
└── dpo.py          # DPO 损失

training/
└── dpo_trainer.py  # DPO 训练器
```

---

## 实践练习

### 练习 1: 实现 DPO 数据集

打开 `data/dpo_dataset.py`，实现 DPO 数据集:

```python
class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        """
        DPO 数据集。

        数据格式:
        {
            "prompt": "...",
            "chosen": "偏好响应",
            "rejected": "不偏好响应"
        }
        """
        pass

    def __getitem__(self, idx):
        """
        获取单个样本。
        """
        pass
```

### 练习 2: 实现 DPOLoss 类

打开 `loss/dpo_loss.py`，实现 DPO 损失类:

```python
class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        DPO 损失。

        Args:
            beta: 温度参数，控制与参考模型的偏离程度
            label_smoothing: 标签平滑系数
        """
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

        公式:
        L = -log σ(β * (log_ratio_chosen - log_ratio_rejected))

        其中 log_ratio = policy_logp - reference_logp

        Args:
            policy_chosen_logps: 策略模型对偏好响应的对数概率
            policy_rejected_logps: 策略模型对不偏好响应的对数概率
            reference_chosen_logps: 参考模型对偏好响应的对数概率
            reference_rejected_logps: 参考模型对不偏好响应的对数概率

        Returns:
            包含 loss, chosen_rewards, rejected_rewards, reward_margin 的字典
        """
        pass
```

### 练习 3: 实现 DPO 训练器

```python
class DPOTrainer:
    def __init__(self, model, reference_model, train_dataset, config):
        """
        DPO 训练器。

        特点:
        - 需要参考模型 (通常是 SFT 后的模型)
        - 参考模型在训练过程中保持冻结
        """
        pass

    def training_step(self, batch):
        """
        执行单步训练。

        流程:
        1. 计算策略模型对偏好/不偏好响应的对数概率
        2. 计算参考模型对偏好/不偏好响应的对数概率
        3. 计算 DPO 损失
        4. 反向传播更新策略模型
        """
        pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage04_alignment/lesson11_dpo/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage04_alignment/lesson11_dpo/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **DPO 论文**: "Direct Preference Optimization: Your Language Model is a Reward Model" (Rafailov et al., 2023)
- **IPO 论文**: "A General Theoretical Paradigm to Understand Learning from Human Preferences" (Liu et al., 2023)
- **偏好数据**: 了解 Anthropic HH-RLHF、OpenAssistant 等偏好数据集
