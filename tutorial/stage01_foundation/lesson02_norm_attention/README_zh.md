# L02: 归一化层与注意力机制

> **课程定位**：这是 Intro2LLM 课程的**第二个实验**，是 LLM 模型核心结构的重要组成部分。在 lesson01 中，我们学习了如何将文本转换为向量表示；在本实验中，我们将学习如何通过**归一化层**稳定训练，以及通过**注意力机制**捕捉 token 之间的关系。

## 实验目的

本实验主要讲解归一化层（LayerNorm 和 RMSNorm）和注意力机制（Multi-Head Attention）的原理与实现。归一化层是深度学习模型中稳定训练的关键组件，而注意力机制则是 Transformer 架构的核心，负责建模序列中不同位置之间的依赖关系。

### 本章你将学到

- **归一化层**：理解 LayerNorm 和 RMSNorm 的区别，掌握为什么现代 LLM 偏好 RMSNorm
- **注意力机制**：掌握缩放点积注意力的数学原理，理解多头注意力的实现
- **工程实现**：学会使用 Python 实现完整的归一化和注意力模块

---

## 第一部分：归一化层 (Normalization)

### 1.1 为什么需要归一化？

在深度神经网络中，**内部协变量偏移 (Internal Covariate Shift)** 是一个经典问题：随着网络层数加深，每一层的输入分布会不断偏移，导致：
- 梯度消失或爆炸
- 训练收敛变慢
- 需要非常小的学习率

归一化层通过将激活值调整到稳定的分布，来解决这个问题。

### 1.2 LayerNorm (层归一化)

#### 1.2.1 算法原理

LayerNorm 由 Ba 等人在 2016 年提出，是 Transformer 架构中的标准归一化方法。

**计算公式**：
$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \odot \gamma + \beta
$$

其中：
- $\mathbb{E}[x]$：对最后一个维度求均值
- $\text{Var}[x]$：对最后一个维度求方差
- $\gamma$ (weight)：可学习的缩放参数
- $\beta$ (bias)：可学习的平移参数
- $\epsilon$：防止除零的小常数（通常 1e-6）

#### 1.2.2 LayerNorm vs BatchNorm

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | batch 维度 | 特征维度 |
| 依赖 batch | 是（训练时需要 batch） | 否 |
| 适用场景 | CV（图像 batch 较大） | NLP（序列长度可变） |
| RNN 适用性 | 不适用 | 适用 |

**关键区别**：
- BatchNorm：对 batch 维度归一化，同一特征在不同样本间归一化
- LayerNorm：对特征维度归一化，同一样本内不同特征间归一化

LayerNorm 更适合 NLP 任务，因为：
1. NLP 中序列长度可变，batch 大小也可能变化
2. 不依赖 batch 统计量，推理时更稳定

#### 1.2.3 Pre-LN vs Post-LN Transformer

原始 Transformer (Post-LN) 使用：
```
x → SubLayer(x) → Add & Norm → ... → Output
```

Pre-LN Transformer 使用：
```
x → Norm → SubLayer → Add → ... → Output
```

**Pre-LN 的优势**：
- 梯度更加稳定，不易出现梯度爆炸
- 训练更鲁棒，学习率调节更简单
- 现在大多数 LLM 采用 Pre-LN 结构

#### 1.2.4 LayerNorm 实现要点

**所在文件**：[model/norm.py](../../../model/norm.py)

**需要补全的代码位置**：
- `LayerNorm.__init__` 方法（第38-53行）
- `LayerNorm.forward` 方法（第55-82行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 创建形状为 `[normalized_shape]` 的 weight 参数，初始化为全 1
   - 创建形状为 `[normalized_shape]` 的 bias 参数，初始化为全 0
   - 使用 `nn.Parameter` 包装使其成为可学习参数

2. **前向传播方法中**：
   - 第一步：保存原始数据类型，将输入转为 float32 提高数值稳定性
   - 第二步：沿最后一个维度计算均值，形状变为 `[..., 1]`
   - 第三步：沿最后一个维度计算方差（使用无偏估计=False）
   - 第四步：标准化 `(x - 均值) / sqrt(方差 + eps)`
   - 第五步：应用可学习参数 `output = normalized * weight + bias`
   - 第六步：恢复原始数据类型

---

### 1.3 RMSNorm (均方根层归一化)

#### 1.3.1 算法原理

RMSNorm 由 Zhang 和 Sennrich 在 2019 年提出，是对 LayerNorm 的简化。

**核心思想**：**去掉中心化操作（去除均值）**，只保留均方根（RMS）缩放。

**计算公式**：
$$
\text{RMS}(x) = \sqrt{\mathbb{E}[x^2] + \epsilon}
$$
$$
y = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

#### 1.3.2 为什么 RMSNorm 更快？

1. **少计算一个均值**：
   - LayerNorm：需要计算均值 + 方差 = 2 次统计
   - RMSNorm：只需计算均方值 = 1 次统计

2. **少一个偏置参数**：
   - LayerNorm：有 weight + bias 两个参数
   - RMSNorm：只有 weight 一个参数

3. **数学运算更简单**：
   - 省去减均值的操作

#### 1.3.3 为什么现代 LLM 偏好 RMSNorm？

1. **计算效率高**：减少约 30% 的归一化计算时间

2. **Pre-LN 结构不需要 bias**：
   - Post-LN 中，残差连接后的 bias 用于重新定位激活分布
   - Pre-LN 中，每层先归一化再计算残差，不需要 bias 来"重新定位"
   - RMSNorm 去掉 bias 是合理的

3. **Empirical 表现相当或更好**：
   - LLaMA、Qwen、Mistral、Gemma 等都采用 RMSNorm

#### 1.3.4 RMSNorm 实现要点

**所在文件**：[model/norm.py](../../../model/norm.py)

**需要补全的代码位置**：
- `RMSNorm.__init__` 方法（第107-121行）
- `RMSNorm.forward` 方法（第123-162行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 创建形状为 `[hidden_size]` 的 weight 参数，初始化为全 1（乘法单位元）
   - 使用 `nn.Parameter` 包装

2. **前向传播方法中**：
   - 第一步：保存原始数据类型，转为 float32
   - 第二步：计算均方值 `MS = mean(x^2)`，形状 `[..., 1]`
   - 第三步：使用 `rsqrt` 计算平方根倒数 `inv_rms = 1 / sqrt(MS + eps)`
   - 第四步：归一化 `x * inv_rms`，利用广播机制
   - 第五步：应用可学习缩放 `normalized * weight`
   - 第六步：恢复原始数据类型

---

## 第二部分：注意力机制 (Attention)

### 2.1 注意力机制的直观理解

注意力机制的核心思想是：**在处理当前位置时，决定应该"关注"前面哪些位置的信息**。

**类比**：阅读一段文字时，我们不会记住每一个细节，而是有选择地关注关键词。注意力机制就是在模拟这个过程。

**数学表达**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$ (Query)：当前位置"想要什么"
- $K$ (Key)：每个位置"提供什么"
- $V$ (Value)：每个位置的"实际内容"
- $d_k$：Key 的维度

### 2.2 缩放点积注意力

#### 2.2.1 数学推导

**点积注意力的优势**：
- 计算效率高（矩阵乘法，GPU 友好）
- 可以并行计算

**为什么要缩放（除以 $\sqrt{d_k}$）**？

假设 $Q$ 和 $K$ 的各元素是均值为 0、方差为 1 的独立随机变量，则：
- $QK^T$ 的每个元素的均值为 0
- $QK^T$ 的每个元素的方差为 $d_k$

当 $d_k$ 较大时，方差也会很大，导致 softmax 函数的输入趋于无穷大，梯度变得非常小（梯度消失）。

**解决方法是除以 $\sqrt{d_k}$**：
- 缩放后 $QK^T$ 的方差恢复到 1
- softmax 函数的输入分布在合理范围内
- 梯度更稳定

> **证明**：
> 设 $q_i, k_j \sim \mathcal{N}(0, 1)$，则 $E[q_i k_j] = 0$，$Var(q_i k_j) = E[q_i^2]E[k_j^2] - (E[q_i]E[k_j])^2 = 1 \cdot 1 - 0 = 1$（假设独立）
> 因此 $Var(\sum_i q_i k_i) = d_k$

#### 2.2.2 掩码机制

**因果掩码 (Causal Mask)**：
- 在自回归生成中，当前位置只能看到之前的位置
- 实现方法：将 $QK^T$ 的上三角设为 $-\infty$

### 2.3 Multi-Head Attention (MHA)

#### 2.3.1 算法原理

单头注意力只能捕捉一种类型的依赖关系。多头注意力通过**多个独立的注意力头**，让模型同时关注不同类型的信息。

**核心公式**：
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$
$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**参数关系**：
- `num_heads * head_dim = hidden_size`
- 每个头的维度 `head_dim = hidden_size // num_heads`

#### 2.3.2 特点

**优点**：
- 每个头可以学习不同类型的依赖关系（如语法、语义、位置）
- 并行计算，效率高

### 2.4 MultiHeadAttention 实现要点

**所在文件**：[model/attention.py](../../../model/attention.py)

**需要补全的代码位置**：
- `MultiHeadAttention.__init__` 方法（第37-60行）
- `MultiHeadAttention.forward` 方法（第62-144行）

**实现步骤**：

1. **初始化方法中**：
   - 从配置中提取 hidden_size、num_attention_heads、attention_dropout
   - 验证 hidden_size 能被 num_heads 整除
   - 创建四个线性投影层：q_proj, k_proj, v_proj, o_proj
   - 投影维度都是 [hidden_size, hidden_size]

2. **前向传播方法中**：
   - **Step 1**：线性投影得到 Q、K、V
   - **Step 2**：reshape 为多头形式 `[batch, num_heads, seq, head_dim]`
   - **Step 3**：应用 RoPE（旋转位置编码）
   - **Step 4**：计算缩放点积注意力分数 `scores = (Q · K^T) / sqrt(d_k)`
   - **Step 5**：应用注意力掩码
   - **Step 6**：softmax 得到注意力权重
   - **Step 7**：计算加权输出 `output = weights · V`
   - **Step 8**：reshape 回原始维度
   - **Step 9**：输出投影

---

## 代码补全位置汇总

### 文件 1: [model/norm.py](../../../model/norm.py)

| 类 | 方法 | 行号 | 功能 |
|------|------|------|------|
| `LayerNorm` | `__init__` | 38-53 | 创建 weight 和 bias 参数 |
| `LayerNorm` | `forward` | 55-82 | 实现层归一化 |
| `RMSNorm` | `__init__` | 107-121 | 创建 weight 参数 |
| `RMSNorm` | `forward` | 123-162 | 实现 RMS 归一化 |

### 文件 2: [model/attention.py](../../../model/attention.py)

| 类 | 方法 | 行号 | 功能 |
|------|------|------|------|
| `MultiHeadAttention` | `__init__` | 37-60 | 初始化 Q/K/V/O 投影层 |
| `MultiHeadAttention` | `forward` | 62-144 | 实现多头注意力 |

---

## 练习

### 对实验报告的要求

- 基于 markdown 格式来完成，以文本方式为主
- 填写各个基本练习中要求完成的报告内容
- 列出你认为本实验中重要的知识点，以及与对应的 LLM 原理中的知识点，并简要说明你对二者的含义、关系、差异等方面的理解

### 练习 1：理解 LayerNorm vs RMSNorm

阅读 `model/norm.py`，思考并回答：

1. LayerNorm 和 RMSNorm 的核心区别是什么？为什么 RMSNorm 更快？
2. Pre-LN Transformer 为什么不需要 LayerNorm 中的 bias 参数？
3. 如果将 RMSNorm 的 weight 初始化为全 0，会发生什么？为什么？

### 练习 2：理解注意力机制

阅读 `model/attention.py`，思考并回答：

1. 缩放因子 $\sqrt{d_k}$ 的作用是什么？如果不除以这个值，会出现什么问题？
2. 注意力掩码（causal mask）的作用是什么？如何实现？
3. 为什么 MultiHeadAttention 需要对 Q、K、V 分别做线性投影，而不是直接使用输入？

### 练习 3：验证你的实现

运行以下测试代码，验证你的实现是否正确：

```python
# 测试 LayerNorm
import torch
from model.norm import LayerNorm

norm = LayerNorm(normalized_shape=128)
x = torch.randn(4, 16, 128)
output = norm(x)

# 验证形状
assert output.shape == x.shape, f"形状应为 {x.shape}，实际为 {output.shape}"

# 验证归一化效果
mean = output.mean(dim=-1)
std = output.std(dim=-1)
assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), "均值应接近 0"
print("LayerNorm 测试通过！")

# 测试 RMSNorm
from model.norm import RMSNorm

rms_norm = RMSNorm(hidden_size=128)
x = torch.randn(4, 16, 128)
output = rms_norm(x)

assert output.shape == x.shape
print("RMSNorm 测试通过！")

# 测试 MultiHeadAttention
from model.attention import MultiHeadAttention
from model.config import ModelConfig

config = ModelConfig(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=512,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    hidden_act="silu",
    use_rms_norm=True,
    use_rope=True,
    use_swiglu=True,
)

mha = MultiHeadAttention(config)
x = torch.randn(2, 16, config.hidden_size)
output, _ = mha(x)

assert output.shape == x.shape
print("MultiHeadAttention 测试通过！")

print("\n所有测试通过！")
```

---

## 常见问题 FAQ

**Q1: LayerNorm 和 RMSNorm 在实际应用中有多大差别？**
A: 在大多数情况下，两者的表现非常接近。RMSNorm 的优势主要在于计算速度更快（约 30%）和参数更少。由于 Pre-LN Transformer 的流行，RMSNorm 已成为现代 LLM 的默认选择。

**Q2: 为什么要除以 $\sqrt{d_k}$ 而不是 $d_k$？**
A: 如果除以 $d_k$，会使得注意力分数的值过小，导致 softmax 接近均匀分布，模型无法学习到显著的注意力模式。除以 $\sqrt{d_k}$ 使得方差恢复到 1，softmax 的输入分布更合理。

**Q3: 为什么每个头需要独立的 Q、K、V 投影？**
A: 独立的投影让每个头可以学习不同的注意力模式，捕捉不同类型的信息（如语法、语义、位置关系）。如果共享投影，所有头将学习到相同的注意力模式。

**Q4: 因果掩码在哪里应用？**
A: 因果掩码应用在 $QK^T$ 矩阵上，将当前位置之后的分数设为 $-\infty$，这样 softmax 后这些位置的注意力权重趋近于 0。

**Q5: 多头注意力中 head_dim 必须是整数吗？**
A: 是的，hidden_size 必须能被 num_attention_heads 整除，否则无法均匀分割成多个头。

---

## 延伸阅读

### 原始论文

1. **LayerNorm**: "Layer Normalization" - Ba et al., 2016
   - https://arxiv.org/abs/1607.06450

2. **RMSNorm**: "Root Mean Square Layer Normalization" - Zhang & Sennrich, 2019
   - https://arxiv.org/abs/1910.07467

3. **Attention**: "Attention Is All You Need" - Vaswani et al., 2017
   - https://arxiv.org/abs/1706.03762

### 实践参考

- **Flash Attention**: 了解 GPU 优化的注意力实现（https://github.com/Dao-AILab/flash-attention）

### 拓展阅读

- **Pre-LN Transformer**: "On Layer Normalization in the Transformer Architecture" - Xiong et al., 2020
- 了解 Pre-LN vs Post-LN 的区别及其对训练稳定性的影响
