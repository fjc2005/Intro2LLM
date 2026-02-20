# L03: FeedForward 与 Transformer Block

> **课程定位**：这是 Intro2LLM 课程的**第三个实验**，是 LLM 模型核心结构的重要组成部分。在 lesson01 中，我们学习了如何将文本转换为向量表示（Tokenizer 与 Embedding）；在 lesson02 中，我们学习了归一化层和注意力机制；在本实验中，我们将学习 **前馈网络（FeedForward）** 的原理，以及如何将各个组件组合成完整的 **Transformer Block**。

## 实验目的

本实验主要讲解前馈网络（Basic FFN、GeGLU 和 SwiGLU）的原理与实现，以及 Transformer Block 的结构设计（Pre-LN vs Post-LN）。前馈网络是 Transformer 架构中负责对每个位置进行非线性变换的关键组件，而 Transformer Block 则是构成整个 LLM 的基本单元。

### 本章你将学到

- **Basic FFN**：理解原始 Transformer 中 FFN 的结构
- **门控线性单元（GLU）**：理解 GeGLU 和 SwiGLU 的区别与优势，掌握门控机制的原理
- **Pre-LN vs Post-LN**：理解两种 Transformer Block 结构的区别，掌握为什么现代 LLM 偏好 Pre-LN
- **残差连接**：理解残差连接在深层网络训练中的作用
- **工程实现**：学会使用 Python 实现完整的前馈网络和 Transformer Block

---

## 第一部分：前馈网络 (FeedForward Network)

### 1.1 为什么需要前馈网络？

在 Transformer 架构中，自注意力机制（Self-Attention）负责捕捉序列中不同位置之间的依赖关系。然而，仅靠注意力机制，模型只能对已有信息进行加权组合，无法进行更复杂的非线性变换。

前馈网络（FeedForward Network，FFN）的作用：
1. **增加非线性表达能力**：通过非线性激活函数，让模型能够学习更复杂的模式
2. **特征变换**：对每个位置的表示进行独立的非线性变换
3. **增加模型容量**：FFN 通常占据模型参数的很大一部分

---

### 1.2 Basic FFN (原始 Transformer)

#### 1.2.1 算法原理

原始 Transformer 论文中使用的前馈网络结构非常简单：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

**结构**：
- 第一层线性变换：hidden_size → intermediate_size
- ReLU 激活函数
- 第二层线性变换：intermediate_size → hidden_size

**ReLU 激活函数**：
$$
\text{ReLU}(x) = \max(0, x)
$$

**特点**：
- 计算简单，效率高
- 曾是 Transformer 的标准配置
- 表达能力相对有限

#### 1.2.2 Basic FFN 实现要点

**所在文件**：[model/feedforward.py](../../../model/feedforward.py)

**需要补全的代码位置**：
- `BasicFFN.__init__` 方法
- `BasicFFN.forward` 方法

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 从配置中提取 hidden_size 和 intermediate_size
   - 创建两个线性投影层：
     - `w1`：hidden_size → intermediate_size（带偏置）
     - `w2`：intermediate_size → hidden_size（带偏置）

2. **前向传播方法中**：
   - 第一步：通过 w1 将 hidden_size 映射到 intermediate_size
   - 第二步：应用 ReLU 激活函数
   - 第三步：通过 w2 将 intermediate_size 映射回 hidden_size

---

### 1.3 门控线性单元 (GLU)

#### 1.3.1 为什么需要 GLU？

原始的 Basic FFN 使用 ReLU 激活，虽然简单有效，但表达能力有限。2020 年，Shazeer 在论文 "GLU Variants Improve Transformer" 中提出了**门控线性单元（GLU）**的概念，通过引入门控机制来控制信息流动，从而提升模型的表达能力。

**核心思想**：
- 通过一个"门控"信号来控制哪些信息可以通过
- 类似于 LSTM 中的门控机制，但更简单高效

#### 1.3.2 GLU 的数学公式

**标准 GLU** 的计算公式：

$$
\text{GLU}(x) = (xW + b) \odot \sigma(xV + c)
$$

其中：
- $W, V$：投影矩阵
- $b, c$：偏置向量
- $\sigma$：Sigmoid 激活函数
- $\odot$：逐元素乘法（Hadamard product）

**关键点**：门控信号 $\sigma(xV + c)$ 的值在 0-1 之间，可以"开关"输入信息。

---

### 1.4 GeGLU (Gated Linear Unit with GELU)

#### 1.4.1 算法原理

GeGLU 是 GLU 的一种变体，使用 **GELU** 作为门控激活函数。

**计算公式**：

$$
\text{GeGLU}(x) = \text{GELU}(xW_{\text{gate}}) \odot (xW_{\text{up}})
$$

其中：
- $W_{\text{gate}}$：门控投影矩阵
- $W_{\text{up}}$：上采样投影矩阵
- GELU 激活函数：$\text{GELU}(x) = x \cdot \Phi(x)$，$\Phi$ 是标准正态分布的 CDF

**GELU 激活函数**：

$$
\text{GELU}(x) = x \cdot P(X \le x) = x \cdot \Phi(x)
$$

近似计算：
$$
\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))
$$

**特点**：
- GELU 是一种平滑的激活函数，比 ReLU 更非线性
- GELU 可以理解为一种"软"的门控
- PaLM 使用 GeGLU

#### 1.4.2 GeGLU 实现要点

**所在文件**：[model/feedforward.py](../../../model/feedforward.py)

**需要补全的代码位置**：
- `FeedForward.__init__` 方法（第48-71行）
- `FeedForward.forward` 方法（第73-113行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 从配置中提取 hidden_size 和 intermediate_size
   - 创建三个线性投影层：
     - `gate_proj`：hidden_size → intermediate_size
     - `up_proj`：hidden_size → intermediate_size
     - `down_proj`：intermediate_size → hidden_size

2. **前向传播方法中**：
   - 第一步：门控投影 - 通过 gate_proj 将 hidden_size 映射到 intermediate_size
   - 第二步：应用 GELU 激活函数到门控投影结果
   - 第三步：上采样投影 - 通过 up_proj 获取另一组中间表示
   - 第四步：逐元素门控乘法 - 激活后的门控值 × 上采样值
   - 第五步：下采样投影 - 通过 down_proj 将 intermediate_size 映射回 hidden_size

---

### 1.5 SwiGLU (Swish-Gated Linear Unit)

#### 1.5.1 算法原理

SwiGLU 是 GLU 的另一种变体，使用 **SiLU (Sigmoid Linear Unit)** 作为门控激活函数，也称为 Swish。

**计算公式**：

$$
\text{SwiGLU}(x) = \text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})
$$

其中 SiLU 激活函数：

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**SiLU 的特点**：
- **自门控**：输入乘以自身的 sigmoid 值
- **平滑非单调**：与 ReLU 不同，SiLU 在负值区域有非零输出
- **负值区域**：可以让模型学习到"抑制"某些特征

**为什么现代 LLM 偏好 SwiGLU**：

1. **更平滑的梯度**：SiLU 在零附近的梯度更平滑
2. **更好的训练稳定性**：在深层网络中表现更好
3. **主流采用**：LLaMA、Qwen、Mistral 等都使用 SwiGLU
4. **性能相当**：与 GeGLU 相比，SwiGLU 略受欢迎

#### 1.5.2 SwiGLU 实现要点

**所在文件**：[model/feedforward.py](../../../model/feedforward.py)

**需要补全的代码位置**：
- `SwiGLU.__init__` 方法（第146-168行）
- `SwiGLU.forward` 方法（第170-208行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 从配置中提取 hidden_size 和 intermediate_size
   - 创建三个线性投影层（与 GeGLU 相同结构）

2. **前向传播方法中**：
   - 第一步：门控投影
   - 第二步：应用 SiLU 激活函数（使用 `nn.SiLU()` 或 `torch.nn.functional.silu`）
   - 第三步：上采样投影
   - 第四步：逐元素门控乘法
   - 第五步：下采样投影

---

### 1.6 FFN 参数量分析

**各类型 FFN 的参数量对比**：

假设 hidden_size = $d$, intermediate_size = $4d$（通常设置为 2-4 倍的 hidden_size）

| FFN 类型 | 参数量计算 | 总参数量 |
|----------|------------|----------|
| Basic FFN | $w1: d \times 4d$ + 偏置 $4d$ + $w2: 4d \times d$ + 偏置 $d$ | $8d^2 + 5d$ ≈ $8d^2$ |
| GeGLU/SwiGLU | gate: $d \times 4d$ + up: $d \times 4d$ + down: $4d \times d$ | $12d^2$ |

**分析**：
- GLU 变体没有偏置项（可学习偏置在 Pre-LN 结构中不需要）
- GLU 变体有 12d² 的参数量，比原始 FFN 的 8d² 多 50%
- 更多的参数量带来了更好的表达能力，这是现代 LLM 愿意付出的代价

---

## 第二部分：Transformer Block 结构

### 2.1 Pre-LN vs Post-LN

Transformer Block 有两种主要的归一化放置方式：**Pre-LN** 和 **Post-LN**。这一选择对训练稳定性有重大影响。

#### 2.1.1 两种结构

**Post-LN（原始 Transformer）**：

```
Input → Attention → Add → Norm → FFN → Add → Norm → Output
```

也称为：
```
x → SubLayer(x) → Add → Norm → Output
```

**Pre-LN（现代标准）**：

```
Input → Norm → Attention → Add → Norm → FFN → Add → Output
```

也称为：
```
x → Norm → SubLayer → Add → Output
```

#### 2.1.2 核心区别对比

| 特性 | Post-LN | Pre-LN |
|------|---------|--------|
| 归一化位置 | 子层输出之后（Add 之后） | 子层输入之前 |
| 残差连接 | 归一化之前 | 归一化之后 |
| 梯度流 | 深层不稳定，输出层梯度大 | 梯度更稳定，各层均匀 |
| 学习率 | 需要 warm-up | 可用大学习率 |
| 训练稳定性 | 深层网络易发散 | 训练更鲁棒 |
| 代表模型 | BERT, GPT-2, 原始 Transformer | LLaMA, Qwen, Mistral |

#### 2.1.3 Post-LN 的问题

**为什么 Post-LN 在深层网络中训练不稳定？**

1. **梯度消失/爆炸**：
   - 在 Post-LN 中，最后一层的梯度需要经过多个归一化层和残差连接
   - 深层网络的梯度路径更长，更容易出现梯度消失或爆炸

2. **依赖学习率 warm-up**：
   - Post-LN Transformer 需要学习率 warm-up 来稳定训练
   - 没有 warm-up 时，模型很容易发散

3. **输出层梯度大**：
   - 输出层附近的层梯度较大，容易导致训练不稳定

#### 2.1.4 Pre-LN 的优势

**为什么现代 LLM 都选择 Pre-LN？**

1. **梯度更稳定**：
   - 每一层的输入都经过归一化，激活值的尺度更稳定
   - 梯度在各层之间分布更均匀

2. **不需要 warm-up**：
   - Pre-LN 可以使用恒定的学习率
   - 也可以使用 warm-up，但即使没有 warm-up 也能稳定训练

3. **可以使用更大的学习率**：
   - 训练收敛更快
   - 超参数更容易设置

4. **理论支持**：
   - 论文 "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) 证明了 Pre-LN 的梯度更有界

> **重要结论**：Pre-LN 是现代 LLM 的标准选择，所有主流模型（LLaMA、Qwen、Mistral 等）都采用 Pre-LN 结构。

---

### 2.2 残差连接 (Residual Connection)

#### 2.2.1 原理

残差连接的核心思想是让网络学习**恒等映射**：

$$
y = F(x) + x
$$

其中 $F(x)$ 是学习到的残差。

**作用**：
1. **缓解梯度消失**：梯度可以直接传回输入
2. **稳定训练**：让网络更容易学习到恒等映射
3. **信息传递**：底层信息可以直接传递到高层

#### 2.2.2 在 Transformer 中的应用

Transformer Block 中有两处残差连接：

1. **注意力残差**：
   $$
   x_{\text{after\_attn}} = \text{Attention}(x) + x
   $$

2. **FFN 残差**：
   $$
   x_{\text{output}} = \text{FFN}(x_{\text{after\_attn}}) + x_{\text{after\_attn}}
   $$

---

### 2.3 TransformerBlock 实现要点

**所在文件**：[model/transformer_block.py](../../../model/transformer_block.py)

**需要补全的代码位置**：
- `TransformerBlock.__init__` 方法（第61-86行）
- `TransformerBlock.forward` 方法（第88-155行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 保存 layer_idx 用于标识当前层
   - 根据 config.use_rms_norm 选择 LayerNorm 或 RMSNorm
   - 创建 Pre-Attention 归一化层：`input_layernorm`
   - 创建自注意力模块：`self_attn`（根据配置选择 MHA 或 GQA）
   - 创建 Pre-FFN 归一化层：`post_attention_layernorm`
   - 创建前馈网络模块：`mlp`（根据 config.use_swiglu 选择 SwiGLU 或 GeGLU）

2. **前向传播方法中**：

   **Pre-LN 结构的自注意力子层**：
   - 第一步：保存残差（原始输入）
   - 第二步：Pre-Attention 归一化（在注意力之前）
   - 第三步：调用自注意力模块，获取输出和 KV 缓存
   - 第四步：残差连接（注意力输出 + 原始输入）

   **Pre-LN 结构的前馈网络子层**：
   - 第五步：保存残差（注意力输出）
   - 第六步：Pre-FFN 归一化（在 FFN 之前）
   - 第七步：调用前馈网络模块
   - 第八步：残差连接（FFN 输出 + 注意力输出）

   - 第九步：返回结果（hidden_states, present_key_value）

---

## 代码补全位置汇总

### 文件 1: [model/feedforward.py](../../../model/feedforward.py)

| 类 | 方法 | 行号 | 功能 |
|------|------|------|------|
| `BasicFFN` | `__init__` | - | 初始化 Basic FFN |
| `BasicFFN` | `forward` | - | 实现 Basic FFN 前向传播 |
| `FeedForward` (GeGLU) | `__init__` | 48-71 | 初始化门控投影层 |
| `FeedForward` (GeGLU) | `forward` | 73-113 | 实现 GeGLU 前向传播 |
| `SwiGLU` | `__init__` | 146-168 | 初始化门控投影层 |
| `SwiGLU` | `forward` | 170-208 | 实现 SwiGLU 前向传播 |
| `get_feed_forward` | - | 212-226 | 根据配置选择 FFN 类型 |

### 文件 2: [model/transformer_block.py](../../../model/transformer_block.py)

| 类 | 方法 | 行号 | 功能 |
|------|------|------|------|
| `TransformerBlock` | `__init__` | 61-86 | 初始化子模块 |
| `TransformerBlock` | `forward` | 88-155 | 实现 Pre-LN Transformer Block |

---

## 练习

### 对实验报告的要求

- 基于 markdown 格式来完成，以文本方式为主
- 填写各个基本练习中要求完成的报告内容
- 列出你认为本实验中重要的知识点，以及与对应的 LLM 原理中的知识点，并简要说明你对二者的含义、关系、差异等方面的理解

### 练习 1：理解 Basic FFN vs GLU 变体

阅读 `model/feedforward.py`，结合算法原理，回答以下问题：

1. Basic FFN 和 GLU 变体（GeGLU/SWiGLU）的核心区别是什么？为什么 GLU 变体可以提升模型的表达能力？
2. GLU 中的"门控"机制是如何工作的？为什么门控信号可以控制信息流动？
3. 如果将 SwiGLU 中的 SiLU 激活换成 ReLU，会发生什么？请从数学公式和实际效果两个角度分析。

### 练习 2：理解 Pre-LN vs Post-LN

思考并回答：

1. Post-LN Transformer 在深层网络训练中会遇到什么问题？为什么这些问题在深层网络中更严重？
2. Pre-LN 结构是如何解决这些问题的？请从梯度流和归一化的角度解释。
3. 为什么现代 LLM（如 LLaMA、Qwen）都采用 Pre-LN 结构？这对模型训练和推理有什么影响？

### 练习 3：理解残差连接

思考并回答：

1. 残差连接的核心思想是什么？为什么它可以缓解梯度消失问题？
2. 在 Pre-LN Transformer Block 中，残差连接应用在哪些位置？如果去掉残差连接，会对模型训练产生什么影响？
3. 残差连接的输出是 $y = F(x) + x$，如果 $F(x)$ 的输出和 $x$ 的维度不一致，应该如何处理？

### 练习 4：验证你的实现

运行以下测试代码，验证你的实现是否正确：

```python
# 测试 Basic FFN
import torch
from model.feedforward import BasicFFN
from model.config import ModelConfig

config = ModelConfig(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
)

basic_ffn = BasicFFN(config)
x = torch.randn(2, 10, config.hidden_size)
output = basic_ffn(x)

assert output.shape == x.shape, f"形状应为 {x.shape}，实际为 {output.shape}"
print("BasicFFN 测试通过！")

# 测试 SwiGLU
from model.feedforward import SwiGLU

swiglu = SwiGLU(config)
output = swiglu(x)

assert output.shape == x.shape
print("SwiGLU 测试通过！")

# 测试 GeGLU
from model.feedforward import FeedForward

geglu = FeedForward(config)
output = geglu(x)

assert output.shape == x.shape
print("GeGLU 测试通过！")

# 测试 TransformerBlock
from model.transformer_block import TransformerBlock

full_config = ModelConfig(
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

block = TransformerBlock(full_config)
output, present_kv = block(x)

assert output.shape == x.shape, f"形状应为 {x.shape}，实际为 {output.shape}"
print("TransformerBlock 测试通过！")

print("\n所有测试通过！")
```

---

## 常见问题 FAQ

**Q1: 为什么需要两个投影层（gate 和 up）而不是一个？**
A: GLU 的核心思想是通过门控机制控制信息流动。gate_proj 产生的门控信号（通过 sigmoid/silu 激活）决定了 up_proj 产生的值中有多少可以"通过"。这种设计比单一的投影更灵活，可以让模型自适应地选择哪些维度应该被激活。

**Q2: SwiGLU 中的 SiLU 和 ReLU 相比，有什么优势？**
A: SiLU 是平滑的、非单调的激活函数，在负值区域有非零输出。这使得模型可以学习到"抑制"某些特征。ReLU 在负值区域恒为 0，无法学习负向的表示。实验表明 SiLU 在 Transformer 中表现更好。

**Q3: Pre-LN 中的归一化放在子层之前还是之后，对训练有什么影响？**
A: Pre-LN（归一化在子层之前）使得每一层的输入都经过归一化，梯度更稳定。Post-LN（归一化在子层之后）会导致深层网络的梯度爆炸，需要学习率 warm-up。Pre-LN 是现代 LLM 的标准选择。

**Q4: 前馈网络的 intermediate_size 通常是 hidden_size 的几倍？**
A: 通常设置为 2-4 倍。原始 Transformer 使用 4 倍（LLaMA 使用 8/3 ≈ 2.67 倍，Qwen 使用 2.75 倍）。更大的 intermediate_size 可以增加模型容量，但也会增加计算和内存开销。

**Q5: Transformer Block 中的 LayerNorm 和 RMSNorm 可以混用吗？**
A: 现代 LLM 通常统一使用 RMSNorm（因为更快）。但从技术上讲，两种归一化可以混用（有些早期模型这样做过）。现在的标准做法是根据 config.use_rms_norm 统一选择。

---

## 延伸阅读

### 原始论文

1. **GLU Variants Improve Transformer** - Shazeer, 2020
   - https://arxiv.org/abs/2002.05202
   - 首次提出 GLU 变体在 Transformer 中的应用

2. **Swish: A Self-Gated Activation Function** - Ramachandran et al., 2017
   - https://arxiv.org/abs/1710.05941
   - 首次提出 Swish/SiLU 激活函数

3. **On Layer Normalization in the Transformer Architecture** - Xiong et al., 2020
   - https://arxiv.org/abs/2002.04745
   - Pre-LN Transformer 的理论分析

4. **PaLM: Scaling Language Modeling with Pathways** - Chowdhery et al., 2022
   - https://arxiv.org/abs/2204.02311
   - PaLM 使用 GeGLU

### 实践参考

- **LLaMA**: https://github.com/meta-llama/llama - 使用 SwiGLU + Pre-LN
- **Qwen**: https://github.com/QwenLM/Qwen - 使用 SwiGLU + Pre-LN
- **Mistral**: https://github.com/mistralai/mistral-src - 使用 SwiGLU + Pre-LN
