# L07: 优化器与学习率调度

## 学习目标

1. **理解** AdamW 和 Lion 优化器的数学原理
2. **掌握** Warmup + Cosine 学习率调度器的数学描述
3. **能够** 配置训练优化器

---

## 理论背景

### 1. AdamW 优化器

#### 1.1 动量法 (Momentum)

动量法通过累积历史梯度来加速收敛:

$$v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t$$

其中 $g_t$ 是当前梯度，$\beta_1$ 通常设为 0.9。

#### 1.2 RMSProp 自适应学习率

RMSProp 对每个参数使用自适应学习率:

$$s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2$$

其中 $s_t$ 是梯度平方的指数移动平均，$\beta_2$ 通常设为 0.999。

#### 1.3 偏差校正

由于动量和方差的初始值为零，需要进行偏差校正:

$$\hat{v}_t = \frac{v_t}{1 - \beta_1^t}$$
$$\hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$

#### 1.4 Adam 更新公式

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

其中 $\eta$ 是学习率，$\epsilon$ 是数值稳定常数 (通常为 1e-8)。

#### 1.5 AdamW vs Adam

**Adam**:
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

**AdamW** (权重衰减):
$$\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon} + \lambda \theta_t\right)$$

AdamW 将权重衰减与自适应学习率解耦，比 Adam + L2 正则化效果更好。

### 2. Lion 优化器

#### 2.1 核心思想

Lion (Layer-wise Adaptive Rate) 使用符号函数替代动量:

$$v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(v_t)$$

#### 2.2 特点

- **更简单**: 移除了 RMSProp 部分
- **内存更少**: 只保存一个状态而非两个
- **效果相当**: 在许多任务上与 AdamW 效果相当甚至更好

### 3. 学习率调度器

#### 3.1 Warmup (预热)

**目的**: 训练初期使用较小的学习率，避免不稳定。

**线性预热**:
$$\eta_t = \eta_{max} \cdot \frac{t}{warmup\_steps}$$

#### 3.2 Cosine Annealing (余弦退火)

**目的**: 在训练后期平滑降低学习率，有助于收敛到更好的极小值。

**余弦退火**:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - warmup\_steps}{T - warmup\_steps}\pi\right)\right)$$

其中 $T$ 是总步数。

#### 3.3 Warmup + Cosine 组合

完整的学习率调度曲线:

1. **预热阶段** [0, warmup_steps]: 线性增长到 $\eta_{max}$
2. **余弦退火阶段** [warmup_steps, T]: 按余弦曲线下降到 $\eta_{min}$

---

## 代码实现

### 项目结构

```
optimizer/
├── adamw.py      # AdamW 实现
├── lion.py       # Lion 实现
└── scheduler.py  # 学习率调度器
```

---

## 实践练习

### 练习 1: 实现 AdamW

打开 `optimizer/adamw.py`，实现 AdamW 优化器:

```python
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        """
        AdamW 优化器实现。

        参数:
        - lr: 学习率
        - betas: (beta1, beta2) 动量衰减系数
        - eps: 数值稳定常数
        - weight_decay: 权重衰减系数 λ

        实现要点:
        1. 维护两个状态: m (动量) 和 v (梯度平方)
        2. 每步更新: 先计算梯度，更新 m 和 v，偏差校正，最后参数更新
        3. 权重衰减直接应用到参数更新中，而非 L2 正则化
        """
        # 实现 AdamW 的参数更新逻辑
        pass
```

### 练习 2: 实现学习率调度器

打开 `optimizer/scheduler.py`，实现 Warmup + Cosine 调度器:

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr_ratio=0.1):
        """
        预热 + 余弦退火学习率调度器。

        调度曲线:
        1. [0, warmup_steps): 线性增长
        2. [warmup_steps, total_steps): 余弦退火

        Args:
            optimizer: 优化器实例
            warmup_steps: 预热步数
            total_steps: 总训练步数
            min_lr_ratio: 最小学习率与初始学习率的比值
        """
        # 实现: 初始化调度器参数
        pass

    def step(self, step):
        """
        更新学习率。

        Args:
            step: 当前训练步数
        """
        # 实现: 根据当前步数计算学习率
        # - 预热阶段: 线性增长
        # - 退火阶段: 余弦曲线下降
        pass
```

### 练习 3: 配置优化器

```python
# 使用示例: 为训练配置优化器和调度器
# 1. 创建优化器 (AdamW 或 Lion)
# 2. 创建学习率调度器
# 3. 在训练循环中更新调度器
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage02_pretrain/lesson07_optimizer_scheduler/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage02_pretrain/lesson07_optimizer_scheduler/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **AdamW 论文**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- **Lion 论文**: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
- **学习率调度**: 了解其他调度策略如 constant, polynomial, exponential 等
