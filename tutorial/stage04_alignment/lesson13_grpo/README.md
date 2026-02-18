# L13: GRPO

## 学习目标

1. **理解** 群组相对偏好优化 (GRPO) 原理
2. **掌握** GRPO 损失函数实现
3. **了解** GRPO 与 DPO 的对比

---

## 理论背景

### 1. GRPO 概述

#### 1.1 背景

GRPO (Group Relative Preference Optimization) 由 DeepSeek 提出，旨在解决 DPO 的一些局限性。

#### 1.2 核心思想

GRPO 对同一问题采样多个响应，根据组内相对排名计算损失，而非像 DPO 那样需要成对的偏好数据。

### 2. GRPO 数学原理

#### 2.1 问题设定

给定一个问题 x，采样 G 个响应 {y_1, y_2, ..., y_G}。

#### 2.2 组内排名

对每个响应计算得分 (通常使用奖励模型或规则):
$$s_i = \text{reward}(x, y_i)$$

按得分排序得到组内排名。

#### 2.3 损失函数

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[ \sum_{i=1}^{G} \frac{\exp(s_i)}{\sum_{j=1}^{G} \exp(s_j)} \log P(y_i | x) \right]$$

**简化理解**: 响应被选中的概率与其在组内的相对排名成正比。

#### 2.4 实际实现

```python
def grpo_loss(log_probs, rewards, beta=0.1):
    """
    计算 GRPO 损失。

    公式:
    L = -E[ sum_i (softmax(rewards)_i * log_prob_i) ]

    即: 加权交叉熵，权重是 softmax 后的奖励
    """
    # 计算 softmax 权重
    weights = F.softmax(rewards / beta, dim=0)

    # 加权损失
    loss = -(weights * log_probs).sum()
    return loss
```

### 3. GRPO vs DPO

| 特性 | DPO | GRPO |
|------|-----|------|
| 数据需求 | 成对偏好数据 | 单问题多响应 |
| 参考模型 | 需要 | 不需要 |
| 显存占用 | 较高 (参考模型) | 较低 |
| 训练稳定性 | 可能不稳定 | 相对稳定 |
| 奖励信号 | 隐式学习 | 直接使用 |

### 4. GRPO 的优势

1. **无需参考模型**: 减少显存占用
2. **直接使用奖励**: 可以使用规则或训练的奖励模型
3. **更稳定**: 组内排名机制减少噪声影响
4. **适合强化学习**: 可以与任何奖励函数结合

---

## 代码实现

### 项目结构

```
data/
└── grpo_dataset.py  # GRPO 数据集

loss/
└── grpo.py         # GRPO 损失

training/
└── grpo_trainer.py # GRPO 训练器
```

---

## 实践练习

### 练习 1: 实现 GRPO 数据集

打开 `data/grpo_dataset.py`，实现 GRPO 数据集:

```python
class GRPODataset(Dataset):
    def __init__(self, data_path, tokenizer, num_samples=4, max_length=2048):
        """
        GRPO 数据集。

        数据格式:
        {
            "prompt": "...",
            "responses": ["response1", "response2", ...]  # 同一问题的多个响应
        }

        或者:
        {
            "prompt": "...",
            "rewards": [r1, r2, ...]  # 每个响应的奖励
        }

        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            num_samples: 每个问题采样的响应数
            max_length: 最大序列长度
        """
        pass

    def __getitem__(self, idx):
        """
        获取单个样本。

        返回:
        - prompt_ids: 提示的 token IDs
        - response_ids: 多个响应的 token IDs
        - rewards: 每个响应的奖励 (如果可用)
        """
        pass
```

### 练习 2: 实现 GRPO 损失

打开 `loss/grpo.py`，实现 GRPO 损失函数:

```python
def compute_grpo_loss(log_probs, rewards, beta=0.1):
    """
    计算 GRPO 损失。

    GRPO 核心思想:
    - 对同一问题采样多个响应
    - 根据组内相对排名加权损失

    公式:
    L = -sum_i (softmax(rewards / beta)_i * log_prob_i)

    Args:
        log_probs: 每个响应的对数概率 [num_samples]
        rewards: 每个响应的奖励 [num_samples]
        beta: 温度参数，控制分布平滑度

    实现要点:
    1. 使用 softmax 将奖励转换为概率分布
    2. 以该分布为权重计算加权交叉熵
    3. 奖励高的响应权重更大

    返回:
        GRPO 损失值
    """
    pass
```

### 练习 3: 实现 GRPO 训练器

```python
class GRPOTrainer:
    def __init__(self, model, train_dataset, config):
        """
        GRPO 训练器。

        特点:
        - 无需参考模型
        - 可以使用奖励模型或规则作为奖励信号
        """
        pass

    def generate_responses(self, prompts, num_samples=4):
        """
        为每个提示生成多个响应。

        Args:
            prompts: 提示列表
            num_samples: 每个提示生成的响应数

        返回:
            响应列表和对应的对数概率
        """
        pass

    def compute_rewards(self, prompts, responses):
        """
        计算奖励。

        可以使用:
        - 奖励模型
        - 规则 (如正确性检查)
        - 人工设计的方法

        Args:
            prompts: 提示列表
            responses: 响应列表

        返回:
            奖励值列表
        """
        pass

    def training_step(self, batch):
        """
        执行单步 GRPO 训练。

        流程:
        1. 生成多个响应
        2. 计算奖励
        3. 计算 GRPO 损失
        4. 更新模型
        """
        pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage04_alignment/lesson13_grpo/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage04_alignment/lesson13_grpo/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **GRPO 论文**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
- **GRPO vs DPO**: 比较两种偏好优化方法
- **奖励设计**: 了解如何设计有效的奖励函数
