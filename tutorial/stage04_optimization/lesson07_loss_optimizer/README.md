# 课时7：损失函数与优化器

## 学习目标

1. 深入理解交叉熵损失的数学原理与数值稳定性
2. 掌握AdamW优化器的完整算法
3. 理解Lion优化器的设计思想
4. 掌握学习率调度策略：Warmup + Cosine Decay
5. 了解DPO损失的初步概念

---

## 1. 交叉熵损失 (Cross Entropy Loss)

### 1.1 数学定义

**分类问题的交叉熵**:
```
对于一个样本，真实标签为y (one-hot)，预测概率为p:
    CE(y, p) = -Σ_i y_i * log(p_i)

由于y是one-hot，只有一个位置为1:
    CE(y, p) = -log(p_y)  (其中y是正确类别的索引)

对于批量样本:
    L = (1/N) * Σ_{i=1}^{N} -log(p_{y_i})
```

### 1.2 数值稳定性问题

```
问题: Softmax容易产生数值溢出

Softmax: p_i = exp(z_i) / Σ_j exp(z_j)

如果z_i很大:
    exp(1000) = inf (浮点数溢出)

解决方案: Softmax稳定化技巧
    p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

减去最大值保证指数的最大值为0:
    exp(z_i - max(z)) ∈ (0, 1]
```

**PyTorch中的稳定实现**:
```python
# 方法1: 直接使用CrossEntropyLoss (推荐)
# 内部已实现log-softmax + nll_loss的稳定版本
loss = F.cross_entropy(logits, targets)

# 方法2: 手动实现稳定版本
log_probs = F.log_softmax(logits, dim=-1)  # 稳定的log-softmax
loss = F.nll_loss(log_probs, targets)      # 负对数似然
```

### 1.3 Label Smoothing

**目的**: 防止模型过度自信，提高泛化能力

```
原始标签: y = [0, 0, 1, 0]  (硬标签)

平滑后:   y' = [0.025, 0.025, 0.925, 0.025]  (软标签)
        其中平滑系数ε=0.1

计算:
    y'_i = (1 - ε) * y_i + ε / K
    K是类别数

损失函数:
    L = -Σ_i y'_i * log(p_i)
```

### 1.4 Ignore Index

**使用场景**: 在SFT中mask掉prompt部分

```python
# labels中-100表示忽略
labels = [1, 2, 3, -100, -100, 6, 7]
#         ↑ 需要计算loss
#                  ↑ 忽略(不参与loss计算)

loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

---

## 2. AdamW优化器

### 2.1 从SGD到Adam

```
SGD (随机梯度下降):
    θ = θ - η * ∇L(θ)

问题: 所有参数使用相同学习率，无法自适应

AdaGrad:
    累积梯度平方: r = r + ∇L²
    自适应学习率: θ = θ - η / √(r + ε) * ∇L

问题: 学习率单调递减，可能过早停止

RMSProp:
    指数移动平均: r = β * r + (1-β) * ∇L²
    解决AdaGrad学习率递减问题

Adam (Adaptive Moment Estimation):
    一阶矩(动量): m = β₁ * m + (1-β₁) * ∇L
    二阶矩(自适应): v = β₂ * v + (1-β₂) * ∇L²
    偏差修正: m̂ = m / (1-β₁^t), v̂ = v / (1-β₂^t)
    更新: θ = θ - η * m̂ / (√v̂ + ε)
```

### 2.2 AdamW: 解耦权重衰减

**论文**: [Decoupled Weight Decay Regularization (2017)](https://arxiv.org/abs/1711.05101)

```
传统L2正则化 (Adam中的实现):
    L' = L + (λ/2) * ||θ||²
    ∇L' = ∇L + λ * θ
    # 梯度更新时应用权重衰减

    问题: 与自适应学习率耦合
          二阶矩v会使大梯度参数的权重衰减变小

AdamW (解耦权重衰减):
    梯度更新: θ = θ - η * m̂ / (√v̂ + ε)
    权重衰减: θ = θ - η * λ * θ

    权重衰减与学习率解耦，更稳定
```

**AdamW完整算法**:
```
初始化:
    m_0 = 0  (一阶矩)
    v_0 = 0  (二阶矩)
    t = 0

超参数:
    α: 学习率
    β₁, β₂: 衰减率 (通常0.9, 0.999)
    ε: 数值稳定性 (1e-8)
    λ: 权重衰减系数

迭代:
    t = t + 1
    g_t = ∇L(θ_{t-1})  (计算梯度)

    m_t = β₁ * m_{t-1} + (1-β₁) * g_t  (更新一阶矩)
    v_t = β₂ * v_{t-1} + (1-β₂) * g_t² (更新二阶矩)

    m̂_t = m_t / (1-β₁^t)  (偏差修正)
    v̂_t = v_t / (1-β₂^t)

    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)  (参数更新)
    θ_t = θ_t - α * λ * θ_{t-1}              (解耦权重衰减)
```

---

## 3. Lion优化器

**论文**: [Symbolic Discovery of Optimization Algorithms (2023)](https://arxiv.org/abs/2302.06675)

### 3.1 核心思想

```
特点:
1. 只使用一阶动量 (无自适应学习率)
2. 使用符号函数更新 (sign momentum)
3. 更少的内存占用
4. 通常需要更大batch size

算法:
    c_t = β₁ * m_{t-1} + (1-β₁) * g_t  (动量计算)
    m_t = β₂ * m_{t-1} + (1-β₂) * g_t  (更新动量)
    θ_t = θ_{t-1} - α * sign(c_t)      (符号更新)
    θ_t = θ_t - λ * θ_{t-1}            (权重衰减)
```

### 3.2 AdamW vs Lion

| 特性 | AdamW | Lion |
|------|-------|------|
| 动量 | 一阶+二阶 | 仅一阶 |
| 内存 | 2×参数 | 1×参数 |
| 更新方向 | 自适应 | sign(momentum) |
| 典型batch size | 较小 | 较大 |
| 学习率 | 通常较小 | 通常较大(3-10×) |
| 权重衰减 | 解耦 | 解耦 |

---

## 4. 学习率调度 (LR Scheduler)

### 4.1 Warmup + Cosine Decay

```
学习率调度策略:

Phase 1: Warmup (预热)
    lr = base_lr * (step / warmup_steps)
    从0线性增加到base_lr
    目的: 训练初期稳定梯度，防止过大更新

Phase 2: Cosine Decay (余弦衰减)
    lr = base_lr * 0.5 * (1 + cos(π * (step - warmup) / (total - warmup)))
    从base_lr平滑衰减到接近0
    目的: 后期精细调整，帮助收敛
```

**可视化**:
```
学习率
  │╲
  │  ╲
  │    ╲
  │      ╲
  │        ╲
  │          ╲___
  └───────────────→ 步数
    ↑warmup   ↓cosine decay
```

### 4.2 其他调度策略

```python
# Linear Decay
lr = base_lr * (1 - step / total_steps)

# Polynomial Decay
lr = base_lr * (1 - step / total_steps) ** power

# Constant with Warmup
lr = base_lr  (after warmup)

# Inverse Square Root
lr = base_lr / sqrt(step)
```

---

## 5. DPO损失初步

### 5.1 Bradley-Terry模型

**基础概念**: 建模成对偏好比较

```
假设: P(y₁ ≻ y₂) = σ(r(x, y₁) - r(x, y₂))

其中:
    σ是sigmoid函数
    r(x, y)是奖励函数
    y₁ ≻ y₂表示y₁优于y₂
```

### 5.2 DPO损失函数

**核心思想**: 直接用策略模型和参考模型的对数概率比作为隐式奖励

```
隐式奖励:
    r(x, y) = β * log(π(y|x) / π_ref(y|x))

DPO损失:
    L_DPO = -log σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x)))

简化:
    L_DPO = -log σ(β * (log_ratio_chosen - log_ratio_rejected))

其中:
    y_w: 偏好回答 (chosen)
    y_l: 非偏好回答 (rejected)
    π: 策略模型 (当前训练模型)
    π_ref: 参考模型 (通常是SFT模型，冻结)
    β: 温度系数，控制与参考模型的偏离程度
```

---

## 6. 实现指引

### 6.1 loss/cross_entropy.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    支持label smoothing和ignore_index的交叉熵损失
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        # Step 1: 保存配置
        # self.ignore_index = ignore_index
        # self.label_smoothing = label_smoothing
        # self.reduction = reduction
        pass

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, vocab_size] 或 [batch, vocab_size]
            labels: [batch, seq_len] 或 [batch]
        Returns:
            loss: 标量
        """
        # Step 1: 如果label_smoothing > 0，手动实现带平滑的交叉熵
        # 否则使用F.cross_entropy

        # Step 2: 使用view/flatten处理多维输入
        # logits.view(-1, vocab_size)
        # labels.view(-1)

        pass
```

### 6.2 optimizer/adamw.py

```python
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """
    AdamW优化器实现

    参数:
        lr: 学习率
        betas: (β₁, β₂) 动量系数
        eps: 数值稳定性常数
        weight_decay: 权重衰减系数
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """单步更新"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 获取超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Step 1: 初始化状态
                if len(state) == 0:
                    # state['step'] = 0
                    # state['exp_avg'] = torch.zeros_like(p)  # m_t
                    # state['exp_avg_sq'] = torch.zeros_like(p)  # v_t
                    pass

                # Step 2: 获取状态
                # exp_avg = state['exp_avg']
                # exp_avg_sq = state['exp_avg_sq']
                # step = state['step']

                # Step 3: 更新一阶矩和二阶矩
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

                # Step 4: 偏差修正
                # bias_correction1 = 1 - beta1 ** step
                # bias_correction2 = 1 - beta2 ** step
                # step_size = lr / bias_correction1

                # Step 5: 计算自适应学习率
                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Step 6: 参数更新
                # p.data = p.data - step_size * exp_avg / denom

                # Step 7: 解耦权重衰减
                # p.data = p.data - lr * weight_decay * p.data

                pass

        return loss
```

### 6.3 optimizer/lion.py

```python
class Lion(Optimizer):
    """
    Lion优化器实现

    特点: 使用符号动量，仅一阶矩，内存高效
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """单步更新"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Step 1: 初始化状态
                if len(state) == 0:
                    # state['step'] = 0
                    # state['exp_avg'] = torch.zeros_like(p)  # 动量
                    pass

                # Step 2: 获取状态
                # exp_avg = state['exp_avg']
                # step = state['step']

                # Step 3: 计算更新方向 (符号动量组合)
                # update = beta1 * exp_avg + (1 - beta1) * grad

                # Step 4: 更新动量
                # exp_avg = beta2 * exp_avg + (1 - beta2) * grad

                # Step 5: 符号更新
                # p.data = p.data - lr * torch.sign(update)

                # Step 6: 权重衰减
                # p.data = p.data - lr * weight_decay * p.data

                pass

        return loss
```

### 6.4 optimizer/scheduler.py

```python
import math

class WarmupCosineScheduler:
    """
    Warmup + Cosine Decay学习率调度
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
    ):
        # Step 1: 保存参数
        # self.optimizer = optimizer
        # self.warmup_steps = warmup_steps
        # self.total_steps = total_steps
        # self.base_lr = base_lr
        # self.min_lr = min_lr
        pass

    def step(self, current_step: int):
        """
        根据当前步数计算学习率

        Args:
            current_step: 当前训练步数
        """
        # Step 1: Warmup阶段
        # if current_step < warmup_steps:
        #     lr = base_lr * (current_step / warmup_steps)

        # Step 2: Cosine Decay阶段
        # else:
        #     progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        #     cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        #     lr = min_lr + (base_lr - min_lr) * cosine_decay

        # Step 3: 更新优化器学习率
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr

        pass

    def get_lr(self) -> float:
        """获取当前学习率"""
        pass
```

### 6.5 loss/dpo_loss.py (初步)

```python
class DPOLoss(nn.Module):
    """
    Direct Preference Optimization损失

    简化版实现，完整版在课时8
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        # self.beta = beta
        pass

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,      # π(y_w|x)
        policy_rejected_logps: torch.Tensor,    # π(y_l|x)
        reference_chosen_logps: torch.Tensor,   # π_ref(y_w|x)
        reference_rejected_logps: torch.Tensor, # π_ref(y_l|x)
    ) -> torch.Tensor:
        """
        Args:
            *_logps: 对数概率，形状[batch]
        """
        # Step 1: 计算log ratios
        # chosen_ratio = policy_chosen_logps - reference_chosen_logps
        # rejected_ratio = policy_rejected_logps - reference_rejected_logps

        # Step 2: 计算logits ( Bradley-Terry模型 )
        # logits = beta * (chosen_ratio - rejected_ratio)

        # Step 3: 计算损失
        # loss = -F.logsigmoid(logits).mean()

        pass
```

---

## 7. 常见陷阱与注意事项

1. **Adam的epsilon位置**: PyTorch中eps是加在分母外，注意与其他实现的区别
2. **学习率单位**: Adam通常需要较小的学习率(1e-3~1e-4)，Lion需要较大(1e-4~1e-6)
3. **Warmup比例**: 通常占总步数的1-10%
4. **Weight decay范围**: 通常0.01-0.1，太大会导致欠拟合
5. **梯度裁剪**: 常与优化器配合使用，防止梯度爆炸
6. **二阶动量初始化**: Adam的二阶矩初始为0，前几步需要偏差修正

---

## 8. 课后练习

1. **手动计算Adam更新**: 给定g_t=0.5, m_{t-1}=0.1, v_{t-1}=0.2，计算θ_t
2. **学习率曲线**: 画出warmup=1000, total=10000, base_lr=1e-4的学习率曲线
3. **Label Smoothing对比**: 对比hard label和smoothed label的梯度
4. **Lion vs Adam**: 分析为什么Lion可以使用更大的学习率
5. **DPO直观理解**: 解释为什么chosen的logprob要大于rejected的logprob
