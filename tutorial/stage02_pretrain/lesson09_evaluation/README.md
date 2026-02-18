# L09: 模型评估

## 学习目标

1. **掌握** Perplexity 评估方法
2. **了解** MMLU 和 HumanEval 评估方法
3. **能够** 实现模型评估

---

## 理论背景

### 1. Perplexity (困惑度)

#### 1.1 定义

Perplexity 是衡量语言模型预测能力的指标，表示模型对下一个 token 的"困惑程度"。

**数学定义**:
$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P_{\theta}(x_t | x_{1:t-1})\right)$$

#### 1.2 直观理解

- Perplexity 可以理解为模型预测时"分支因子"的期望
- Perplexity = 10 表示模型在每一步大约有 10 个等可能的选项
- Perplexity 越低越好 (理想情况下为 1)

#### 1.3 与交叉熵的关系

$$\text{PPL} = \exp(H(P, Q))$$

其中 $H(P, Q)$ 是真实分布与模型预测的交叉熵。

#### 1.4 注意事项

- Perplexity 对数据集敏感，不同数据集的 PPL 不能直接比较
- 需要使用标准化的测试集进行评估
- 长文本的 PPL 通常更低

### 2. MMLU (Multi-task Language Understanding)

#### 2.1 简介

MMLU 是一个大规模多任务语言理解基准测试，包含 57 个任务，涵盖:

- 基础数学
- 社会科学
- 自然科学
- 历史、法律、伦理等

#### 2.2 评估方式

- **选择题格式**: 每个问题提供 4 个选项 (A, B, C, D)
- **评估指标**: 准确率 (Accuracy)
- **计算**: $\text{Accuracy} = \frac{\text{正确数}}{\text{总数}}$

#### 2.3 提示格式

```
问题: {question}

A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

答案: {correct_answer}
```

### 3. HumanEval (代码生成评估)

#### 3.1 简介

HumanEval 是评估代码生成能力的基准数据集，包含 164 个人工编写的编程问题。

#### 3.2 评估指标

**Pass@K**: 模型生成 K 个代码样例中至少有一个正确的概率

**计算方法**:
$$\text{pass@K} = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

其中:
- n: 总问题数
- c: 正确问题数
- k: 采样数量

#### 3.3 评估流程

1. 给定问题描述和函数签名
2. 模型生成代码
3. 在测试用例上运行生成的代码
4. 检查是否通过所有测试用例

---

## 代码实现

### 项目结构

```
evaluation/
├── perplexity.py  # Perplexity 计算
├── mmlu.py       # MMLU 评估
├── humaneval.py  # HumanEval 评估
└── benchmark.py  # 基准测试框架
```

---

## 实践练习

### 练习 1: 实现 Perplexity 计算

打开 `evaluation/perplexity.py`，实现困惑度计算:

```python
def compute_perplexity(model, dataloader, device="cuda"):
    """
    计算语言模型的困惑度。

    公式: PPL = exp(-1/T * sum_t log P(x_t | x_<t))

    Args:
        model: 语言模型
        dataloader: 测试数据加载器
        device: 计算设备

    实现思路:
    1. 遍历数据集，前向传播计算损失
    2. 累积总损失和总 token 数
    3. 计算平均损失并转换为 perplexity

    返回:
        困惑度值
    """
    pass
```

### 练习 2: 实现 MMLU 评估

```python
def evaluate_mmlu(model, dataset, few_shot_examples=5):
    """
    在 MMLU 数据集上评估模型。

    Args:
        model: 语言模型
        dataset: MMLU 数据集
        few_shot_examples: 每个任务的 few-shot 示例数

    评估流程:
    1. 加载 MMLU 数据集
    2. 对每个任务，构建 few-shot 提示
    3. 将问题提供给模型，让模型选择答案
    4. 统计准确率

    返回:
        各任务的准确率字典
    """
    pass
```

### 练习 3: 实现 HumanEval 评估

```python
def evaluate_humaneval(model, dataset, num_samples=200):
    """
    在 HumanEval 数据集上评估代码生成能力。

    评估指标: Pass@K

    Args:
        model: 语言模型
        dataset: HumanEval 数据集
        num_samples: 每个问题生成的代码样例数

    评估流程:
    1. 加载 HumanEval 问题
    2. 给定函数签名和文档字符串，让模型生成代码
    3. 使用代码执行器运行生成的代码
    4. 检查是否通过测试用例
    5. 计算 Pass@K

    返回:
        Pass@K 值
    """
    pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage02_pretrain/lesson09_evaluation/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage02_pretrain/lesson09_evaluation/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **MMLU 论文**: "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021)
- **HumanEval 论文**: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
- **更多评估基准**: 了解 BigBench、AGIEval 等
