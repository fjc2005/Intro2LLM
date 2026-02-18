# 课时8：DPO直接偏好优化

## 学习目标

1. 深入理解RLHF流程与DPO的核心思想
2. 掌握Bradley-Terry偏好模型的数学基础
3. 完整推导DPO损失函数
4. 实现Reference Model管理与Beta参数调优
5. 掌握DPOTrainer的完整实现

---

## 1. RLHF回顾与DPO动机

### 1.1 传统RLHF流程

```
Step 1: 训练SFT模型
    - 在instruction数据上微调
    - 输出: π^SFT (监督微调策略)

Step 2: 训练奖励模型 (RM)
    - 收集偏好数据 (pairwise comparison)
    - 训练模型 r(x, y) 预测人类偏好
    - 需要大量标注资源和训练成本

Step 3: PPO优化
    - 使用PPO算法最大化奖励
    - 同时约束与SFT模型的KL散度
    - 流程复杂、训练不稳定

问题:
    - 需要训练额外的奖励模型
    - PPO训练复杂、超参数敏感
    - 需要维护多个模型 (policy, value, reward, reference)
```

### 1.2 DPO核心思想

**论文**: [Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290)

```
核心洞察: 奖励模型可以被解析地表达为最优策略的函数

关键结论: 不需要显式训练奖励模型!

DPO直接优化策略，使其满足偏好数据
```

**RLHF vs DPO对比**:

| 特性 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 奖励模型 | 需要单独训练 | 隐式包含 |
| 训练算法 | PPO (复杂) | 直接梯度下降 (简单) |
| 模型数量 | 4个 | 2个 (policy + reference) |
| 训练稳定性 | 较不稳定 | 稳定 |
| 超参数 | 较多 | 较少 |
| 计算成本 | 高 | 低 |

---

## 2. Bradley-Terry模型

### 2.1 成对偏好建模

```
假设: 人类偏好比率服从以下模型

P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

其中:
    y_w: 偏好回答 (win)
    y_l: 非偏好回答 (loss)
    r(x, y): 奖励函数
    σ: sigmoid函数

直观理解:
    - 奖励差越大，偏好概率越高
    - 奖励相同时，偏好概率为0.5
```

### 2.2 从偏好数据学习

```
给定偏好数据集 D = {(x, y_w, y_l)}

最大似然估计:
    L(r) = -E_{(x,y_w,y_l)~D} [log σ(r(x, y_w) - r(x, y_l))]

目标: 找到奖励函数r最大化观测偏好的似然
```

---

## 3. DPO损失函数推导

### 3.1 RLHF的目标

```
传统RLHF的目标是找到最优策略 π*:

π* = argmax_π E_{x~D, y~π}[r(x, y)] - β * D_KL(π || π_ref)

约束项 (KL散度):
    - 防止策略偏离参考模型太远
    - β控制偏离程度

闭式解:
    π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x, y) / β)

其中 Z(x) = Σ_y π_ref(y|x) * exp(r(x, y) / β) 是配分函数
```

### 3.2 奖励的解析表达

```
从闭式解反解奖励:
    π*(y|x) ∝ π_ref(y|x) * exp(r(x, y) / β)

取对数:
    log π*(y|x) = log π_ref(y|x) + r(x, y) / β - log Z(x)

整理:
    r(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)

由于Z(x)与y无关，在偏好比较中抵消:
    r(x, y_w) - r(x, y_l) = β * log(π*(y_w|x) / π_ref(y_w|x))
                           - β * log(π*(y_l|x) / π_ref(y_l|x))
```

### 3.3 DPO损失函数

```
将奖励表达代入Bradley-Terry模型:

P(y_w ≻ y_l | x) = σ(β * log(π(y_w|x) / π_ref(y_w|x))
                    - β * log(π(y_l|x) / π_ref(y_l|x)))

                  = σ(β * (log_ratio_winner - log_ratio_loser))

DPO损失 (负对数似然):
    L_DPO(π; π_ref) = -E_{(x,y_w,y_l)~D} [
        log σ(β * log(π(y_w|x)/π_ref(y_w|x))
              - β * log(π(y_l|x)/π_ref(y_l|x)))
    ]

简写:
    let r̂(x, y) = β * log(π(y|x) / π_ref(y|x))
    L_DPO = -log σ(r̂(x, y_w) - r̂(x, y_l))
```

---

## 4. DPO关键组件

### 4.1 Reference Model

```
作用:
    - 提供参考分布 π_ref
    - 通常是SFT模型 (冻结参数)
    - 防止策略偏离太远

实现:
    reference_model = copy.deepcopy(sft_model)
    for param in reference_model.parameters():
        param.requires_grad = False

注意:
    - reference模型不更新
    - 只用于计算log概率作为对比基准
```

### 4.2 Beta参数 (β)

```
β的作用:
    - 控制与参考模型的偏离程度
    - β → 0: 完全遵循偏好数据，可能过拟合
    - β → ∞: 保持接近参考模型，变化小

典型值:
    - 0.1 ~ 0.5 (根据任务调整)
    - 通常0.1是一个好的起点

调优策略:
    - 如果模型偏离太远 (输出不稳定): 增大β
    - 如果模型变化太小 (改进不明显): 减小β
```

### 4.3 Log概率计算

```
log π(y|x) = Σ_{t=1}^{|y|} log π(y_t | x, y_{<t})

实现步骤:
    1. 将prompt+response编码为input_ids
    2. 前向传播得到logits
    3. 取response对应位置的log概率
    4. 对response tokens求和

注意:
    - 需要mask掉prompt部分 (只计算response)
    - 使用log_softmax获取对数概率
```

---

## 5. 实现指引

### 5.1 loss/dpo_loss.py (完整版)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization损失

    L_DPO = -log σ(β * (log_π(y_w) - log_π_ref(y_w))
                   - β * (log_π(y_l) - log_π_ref(y_l)))
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        # Step 1: 保存超参数
        # self.beta = beta
        # self.label_smoothing = label_smoothing
        pass

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,      # [batch]
        policy_rejected_logps: torch.Tensor,    # [batch]
        reference_chosen_logps: torch.Tensor,   # [batch]
        reference_rejected_logps: torch.Tensor, # [batch]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算DPO损失

        Args:
            policy_chosen_logps: 策略模型对偏好回答的对数概率
            policy_rejected_logps: 策略模型对非偏好回答的对数概率
            reference_chosen_logps: 参考模型对偏好回答的对数概率
            reference_rejected_logps: 参考模型对非偏好回答的对数概率

        Returns:
            loss: 标量损失
            metrics: 包含辅助指标的字典
        """
        # Step 1: 计算隐式奖励 (scaled log ratios)
        # policy_chosen_logratios = policy_chosen_logps - reference_chosen_logps
        # policy_rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Step 2: 计算logits (Bradley-Terry模型输入)
        # logits = self.beta * (policy_chosen_logratios - policy_rejected_logratios)

        # Step 3: 计算DPO损失 (负对数似然)
        # losses = -F.logsigmoid(logits)

        # Step 4: Label smoothing (可选)
        # if self.label_smoothing > 0:
        #     losses = (1 - label_smoothing) * losses + label_smoothing * F.logsigmoid(-logits)

        # Step 5: 计算辅助指标
        # chosen_rewards = self.beta * policy_chosen_logratios
        # rejected_rewards = self.beta * policy_rejected_logratios
        # reward_margin = chosen_rewards - rejected_rewards
        # accuracy = (reward_margin > 0).float().mean()

        pass

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时计算chosen和rejected的log概率

        技巧: 将chosen和rejected拼接，一次前向传播
        """
        # Step 1: 拼接输入
        # concatenated_input_ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]])
        # concatenated_attention_mask = torch.cat([...])

        # Step 2: 前向传播
        # outputs = model(concatenated_input_ids, attention_mask=concatenated_attention_mask)

        # Step 3: 分割结果
        # all_logps = self._get_batch_logps(...)
        # chosen_logps = all_logps[:batch_size]
        # rejected_logps = all_logps[batch_size:]

        pass

    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        """
        计算一批样本的log概率

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]，prompt部分为-100
            average_log_prob: 是否取平均(长度归一化)

        Returns:
            log_probs: [batch]，每个样本的log概率
        """
        # Step 1: 计算log softmax
        # log_probs = F.log_softmax(logits, dim=-1)

        # Step 2: 收集目标token的log概率
        # per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # Step 3: mask掉prompt部分 (labels == -100)
        # loss_mask = (labels != -100).float()
        # per_token_logps = per_token_logps * loss_mask

        # Step 4: 求和或平均
        # if average_log_prob:
        #     return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # else:
        #     return (per_token_logps * loss_mask).sum(-1)

        pass
```

### 5.2 training/dpo_trainer.py

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

class DPOTrainer:
    """
    DPO训练器

    同时维护policy模型和reference模型
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional = None,
    ):
        # Step 1: 保存模型
        # self.model = model  # policy模型 (可训练)
        # self.ref_model = ref_model  # reference模型 (冻结)

        # Step 2: 冻结reference模型
        # for param in self.ref_model.parameters():
        #     param.requires_grad = False

        # Step 3: 初始化损失函数
        # self.loss_fn = DPOLoss(beta=beta)

        # Step 4: 保存优化器和调度器
        pass

    def compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算模型对response的log概率

        Args:
            model: 要计算的模型
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            prompt_length: [batch]，每个样本的prompt长度

        Returns:
            logps: [batch]，每个样本的log概率
        """
        # Step 1: 前向传播
        # outputs = model(input_ids, attention_mask=attention_mask)
        # logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        # Step 2: 构建labels (mask掉prompt)
        # labels = input_ids.clone()
        # for i, pl in enumerate(prompt_length):
        #     labels[i, :pl] = -100

        # Step 3: 计算log概率
        # log_probs = self.loss_fn._get_batch_logps(logits, labels)

        pass

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        单步训练

        Args:
            batch: 包含chosen和rejected的数据

        Returns:
            metrics: 损失和辅助指标
        """
        # Step 1: 计算policy模型的logps
        # policy_chosen_logps = self.compute_logps(
        #     self.model, batch["chosen_input_ids"], ..., batch["prompt_length"]
        # )
        # policy_rejected_logps = self.compute_logps(...)

        # Step 2: 计算reference模型的logps (无梯度)
        # with torch.no_grad():
        #     ref_chosen_logps = self.compute_logps(self.ref_model, ...)
        #     ref_rejected_logps = self.compute_logps(self.ref_model, ...)

        # Step 3: 计算DPO损失
        # loss, metrics = self.loss_fn(
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     ref_chosen_logps,
        #     ref_rejected_logps,
        # )

        # Step 4: 反向传播和更新
        # loss.backward()
        # self.optimizer.step()
        # self.lr_scheduler.step()
        # self.optimizer.zero_grad()

        pass

    def save_model(self, output_dir: str):
        """保存policy模型"""
        pass
```

---

## 6. 关键公式总结

### Bradley-Terry模型
```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

### 隐式奖励
```
r̂(x, y) = β * log(π(y|x) / π_ref(y|x))
```

### DPO损失
```
L_DPO = -log σ(β * (r̂(x, y_w) - r̂(x, y_l)))
```

### Log概率计算
```
log π(y|x) = Σ_{t=1}^{|y|} log π(y_t | x, y_{<t})
```

---

## 7. 常见陷阱与注意事项

1. **Reference模型必须冻结**: 不要意外更新reference模型
2. **Log概率的数值稳定性**: 使用log_softmax而不是softmax后取log
3. **Prompt masking**: 确保只计算response部分的log概率
4. **Batch size**: DPO通常需要比SFT更大的batch size
5. **Beta调优**: 从0.1开始，根据生成质量调整
6. **梯度累积**: 支持梯度累积，但要注意正确平均loss
7. **Reference模型加载**: 使用训练前的checkpoint，不要与policy共享

---

## 8. 课后练习

1. **手动计算DPO损失**: 给定具体的log概率值，手动计算loss
2. **Beta影响实验**: 测试不同beta值对训练的影响
3. **奖励可视化**: 训练过程中可视化隐式奖励的变化
4. **对比实验**: 对比DPO和SFT在相同偏好数据上的表现
5. **DPO推导**: 完整推导DPO损失的闭式解
