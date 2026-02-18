# L01: Tokenizer 与 Embedding

> **课程定位**：这是 Intro2LLM 课程的**第一个实验**，也是 LLM 模型基础结构的**起始点**。在本实验中，我们将从零开始理解文本如何被计算机"读懂"，以及这些数字如何获得"位置信息"。

## 实验目的

本实验主要讲解分词器（Tokenizer）和嵌入层（Embedding）的原理与实现。我们的 LLM 系统需要将人类可读的文本转换为计算机可处理的数字表示，再通过嵌入层获得向量表示。为了让模型理解文本的顺序信息，还需要引入位置编码机制。

### 本章你将学到

- **分词算法**：理解 BPE 和字节级分词的原理，掌握从字符到子词的词表构建方法
- **嵌入层**：理解离散 token ID 如何转换为连续向量表示
- **位置编码**：掌握 Transformer 中位置信息的注入方式（Sinusoidal 和 RoPE）
- **工程实现**：学会使用 Python 实现完整的分词器和嵌入模块

---

## 第一部分：分词器 (Tokenizer)

### 1.1 为什么需要分词？

在自然语言处理中，我们需要将**文本**（人类可读）转换为**数字**（计算机可处理）。这个过程分为两个步骤：

1. **分词 (Tokenization)**：将文本拆分成小的语义单元（tokens）
2. **编码 (Encoding)**：将 tokens 转换为数字 IDs

**核心问题**：如何切分文本？

| 切分方式 | 示例 | 优点 | 缺点 |
|---------|------|------|------|
| 词级 | "今天" → ["今天"] | 语义清晰 | OOV问题严重 |
| 字符级 | "今天" → ["今", "天"] | 无OOV | 序列太长，语义弱 |
| 子词级 | "今天" → ["今", "天"] 或 ["今天"] | 平衡语义和覆盖 | 需要算法构建 |

**OOV (Out-of-Vocabulary)**：未登录词，词表中没有的词。例如训练时没有见过"人工智能"，词级分词就无法处理。

### 1.2 BPE (Byte Pair Encoding) 算法

#### 1.2.1 算法原理

BPE 是一种**数据压缩**算法，后来被引入 NLP 用于构建子词词表。2015 年，Sennrich 等人将其从压缩领域引入机器翻译，解决了未登录词（OOV）问题。

**核心思想**：
- 从单个字符开始
- 不断合并高频相邻字符对
- 最终得到一个"子词"词表

**为什么能解决 OOV 问题**？

例如，训练时没有见过"人工智能"这个词，但我们可能见过"人工"和"智能"这两个子词。这样，"人工智能"就可以被表示为 `["人工", "智能"]`，无需特殊处理。

#### 1.2.2 算法步骤详解

**训练阶段**（构建词表和合并规则）：

```
输入: 文本语料, 目标词表大小
输出: 词表, 合并规则列表

Step 1: 初始化
  - 将文本按空格拆分，得到单词列表
  - 每个单词拆分为字符序列 (末尾添加 </w> 表示词尾)
  - 词表 = 所有出现的字符

Step 2: 迭代合并
  循环直到达到目标词表大小:
    a) 统计所有相邻字符对的频率
       例如: "aaab" → [("a","a"), ("a","a"), ("a","b")]
    b) 找到频率最高的字符对 (A, B)
    c) 将 A+B 加入词表
    d) 将语料中所有相邻的 "A B" 替换为 "AB"
```

**编码阶段**（分词）：

```
输入: 一个单词
输出: 子词列表

Step 1: 从左到右遍历
Step 2: 按最长匹配原则，优先匹配更长的子词
Step 3: 如果当前子词不在词表中，回退到单个字符
```

**解码阶段**：

```
输入: 子词列表
输出: 原始文本

Step 1: 直接拼接所有子词
Step 2: 移除词尾标记 </w>
```

#### 1.2.3 具体示例演示

假设语料：`"aaab"` 出现 3 次

**第一轮合并**：
- 字符对 "aa" 出现 2×3=6 次（每行有2对"aa"）
- 合并 "aa" → "aaa"
- 词表添加 "aaa"
- 文本变为：`"aaaab"` (每个 "aa" 合成了 "aaa")

**第二轮合并**：
- 字符对 "aaa", "aa" 各有 3 次
- 继续合并高频率的对...

**编码示例**：
假设训练得到的合并规则是：`[("e", "r"), ("er", "s")]`
- 输入单词："hello" → ['h', 'e', 'l', 'l', 'o', '</w>']
- 应用规则 ("e", "r")：['h', 'er', 'l', 'l', 'o', '</w>']
- 应用规则 ("er", "s")：['h', 'ers', 'l', 'l', 'o', '</w>']
- 最终分词结果：["hers", "llo"] 或类似

#### 1.2.4 BPE 实现要点

**所在文件**：[tokenizer/bpe_tokenizer.py](../../../tokenizer/bpe_tokenizer.py)

**需要补全的代码位置**：
- `BaseTokenizer.__init__` 方法（第40-73行）：初始化词表和特殊 token
- `train` 方法（第64-110行）：训练 BPE，构建词表和合并规则
- `encode` 方法（第112-164行）：编码文本为 token IDs
- `decode` 方法（第166-205行）：解码 token IDs 为文本
- `_pretokenize` 方法（第207-217行）：预分词，将文本拆分为单词
- 其他辅助方法

**训练实现要点**：

1. **准备阶段 - 构建单词频率表**：
   - 遍历每条文本，用空格拆分得到单词列表
   - 统计每个单词出现的次数，存入字典
   - 同时将所有出现过的字符加入初始词表集合
   - 词表需要包含一个特殊的词结束标记（如 `</w>`）

2. **迭代合并 - 找到最佳字符对**：
   - 创建一个空的字典用于统计字符对频率
   - 遍历每个单词及其出现次数
   - 将单词拆成字符列表，末尾加上结束标记
   - 对每个单词中每一对相邻字符，用字符对作为key，出现次数作为value累加
   - 循环结束后，从字典中找出value最大的字符对，这就是本轮要合并的对象

3. **更新词表和规则**：
   - 把合并后的新字符（比如 "a" + "b" = "ab"）加入词表
   - 把这个合并规则（字符对 tuple）记录到merges列表中
   - 需要更新所有单词的表示方式，把相邻的这两个字符替换成合并后的新字符

**编码实现要点**：

1. **预处理**：先把单词拆成单个字符，末尾加上结束标记

2. **按顺序应用合并规则**：
   - 从merges列表的第一个规则开始，依次尝试
   - 对每个规则，遍历当前的所有字符，检测是否有相邻的两个字符能与规则匹配
   - 如果匹配成功，将这两个字符合并为一个新字符
   - 继续用同样的规则检查下一个位置，直到这一位置无法再合并
   - 移动到下一个位置重复这个过程

3. **转换为ID**：将最终的字符列表在词表中查找对应的ID，如果找不到则使用未知token的ID

---

### 1.3 字节级分词 (Byte-Level Tokenization)

#### 1.3.1 算法原理

字节级 BPE 是 **GPT-2/RoBERTa** 采用的分词方式。2019年，Radford 等人在 GPT-2 论文中首次大规模使用这种方法。

**核心区别于普通 BPE**：
- 普通 BPE：基础词表 = 字符（可能有几千个，取决于语言）
- 字节级 BPE：基础词表 = **256 个字节**（固定）

#### 1.3.2 为什么需要字节级？

**解决 Unicode 问题**：
- 普通 BPE 依赖空格拆分，对中文等无空格语言效果差
- 字节级直接处理 UTF-8 编码的字节，**任何字符**都能表示
- 真正做到 **0 OOV**

**例子**：
- 中文 "你" 的 UTF-8 编码：`[e4, bd, a0]` (3个字节)
- 字节级 BPE 可以将 `[e4, bd]` 合并为一个 token
- 英文 "hello" 的 UTF-8 编码：`[68, 65, 6c, 6c, 6f]`

**为什么 UTF-8**？

UTF-8 是一种变长编码：
- ASCII 字符（0-127）：1 字节
- 中文、日文等：3-4 字节
- Emoji：4 字节

无论哪种语言，最终都能分解为 0-255 的字节值。

#### 1.3.3 字节到 Unicode 的映射

问题：字节值 0-255，很多是不可打印的控制字符。

解决方案：GPT-2 使用一种巧妙的映射，将每个字节映射到一个可打印的 Unicode 字符：
- 首先收集所有可打印的 ASCII 字符（从感叹号到波浪号，共95个）
- 然后将剩余的161个字节按顺序映射到 Unicode 的私有区域（从256开始）
- 最终建立一个从字节值（0-255）到单个 Unicode 字符的映射表
- 同时建立反向映射，方便从字符还原为字节

> **趣闻**：GPT-2 的映射表中有 33-126 是可打印 ASCII，剩余的映射到 Unicode 私有区，所以你在 GPT-2 的词表中会看到一些奇怪的 Unicode 字符。

#### 1.3.4 字节级 BPE 实现要点

**所在文件**：[tokenizer/byte_level_tokenizer.py](../../../tokenizer/byte_level_tokenizer.py)

**需要补全的代码位置**：
- `__init__` 方法（第41-57行）：初始化字节映射
- `_create_bytes_to_unicode` 方法（第59-75行）：创建字节到 Unicode 映射表
- `_bytes_to_unicode` 方法（第77-100行）：将文本转为字节级 Unicode
- `_unicode_to_bytes` 方法（第102-127行）：将 Unicode 转回文本
- `train` 方法（第129-147行）：训练字节级 BPE
- `encode` 方法（第149-179行）：编码文本
- `decode` 方法（第181-207行）：解码文本

**实现步骤**：

1. **初始化映射**：
   - 创建两个方向的映射字典：字节→Unicode字符，Unicode字符→字节

2. **文本转字节表示**：
   - 首先将文本编码为 UTF-8 字节序列
   - 然后把每个字节值通过映射表转换为对应的 Unicode 字符
   - 最后把所有字符拼接成一个字符串返回

3. **字节转回文本**：
   - 将字符串中的每个字符通过反向映射表转换为字节值
   - 将字节值列表组合成 bytes 对象
   - 使用 UTF-8 解码还原为原始文本（注意处理解码错误）

4. **训练和编码**：
   - 在进行 BPE 训练或编码前，先把文本转换为字节级表示
   - 后面的流程与普通 BPE 相同，只是操作的基本单元变成了字节而非字符

---

## 第二部分：Embedding（嵌入层）

### 2.1 为什么需要 Token Embedding？

**核心问题**：Transformer 的 Self-Attention 本质是**矩阵乘法**，只能处理连续数值，无法直接处理离散的 token ID。

**解决方案**：将每个 token ID 映射为一个固定维度的**向量**。

```
输入: token ID = 1234 (整数)
 ↓ 查表 (Embedding)
输出: [0.12, -0.34, 0.56, ...] (维度为 hidden_size 的向量)
```

**直观理解**：

想象一个巨大的字典，每一页（词表中的一个词）都对应一个固定长度的描述（嵌入向量）。当我们"查字典"时，不是返回文字，而是返回这个描述。这个描述是连续的值，可以进行数学运算（加减乘除），从而捕捉语义关系。

例如，通过训练我们可能发现：
- 嵌入向量("国王") - 嵌入向量("男人") + 嵌入向量("女人") ≈ 嵌入向量("王后")

### 2.2 TokenEmbedding 实现

#### 2.2.1 原理

Embedding 本质是一个**可学习的查找表 (Lookup Table)**：
- 形状：`[vocab_size, hidden_size]`
- 第 `i` 行表示第 `i` 个 token 的向量表示
- 这个矩阵是模型的**参数**，通过反向传播学习得到

**手动实现 vs 调用高级 API**：

我们不调用 PyTorch 现成的 `nn.Embedding`，而是手动创建嵌入矩阵，这样能更深入理解其原理：

- 使用 `nn.Parameter` 创建可学习参数
- 使用索引操作 `[:, index]` 手动查表

#### 2.2.2 TokenEmbedding 实现要点

**所在文件**：[model/embedding.py](../../../model/embedding.py)

**需要补全的代码位置**：
- `TokenEmbedding.__init__` 方法（第33-45行）
- `TokenEmbedding.forward` 方法（第47-59行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 创建一个形状为 `[vocab_size, hidden_size]` 的嵌入矩阵作为参数
   - 需要使用 `nn.Parameter` 包装，使其成为可学习的模型参数
   - 对嵌入矩阵进行初始化，通常使用正态分布，均值为0，标准差为0.02
   - **注意**：这里不要使用 `nn.Embedding`，而是直接创建一个 `nn.Parameter`

2. **前向传播方法中（手动查表实现）**：
   - 接收一个形状为 `[batch_size, seq_len]` 的 token ID 张量
   - 需要手动实现查表操作：
     - 获取 batch 中每个位置的 token ID
     - 用这些 ID 作为索引，从嵌入矩阵中取出对应的行
     - 返回的向量形状为 `[batch_size, seq_len, hidden_size]`
   - **实现思路**：使用索引操作 `self.embedding_table[input_ids]`

**手动实现的关键点**：

- 创建参数：`nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)`
- 查表操作：`self.embedding_table[input_ids]`
  - `input_ids` 是 `[batch, seq]` 的 LongTensor
  - 索引结果是 `[batch, seq, hidden_size]`

---

## 第三部分：位置编码 (Positional Encoding)

### 3.1 为什么需要位置编码？

**Transformer 的特点**：
- Self-Attention 机制是**置换等变 (Permutation Equivariant)** 的
- 意味着：输入顺序改变，输出结果不变！

**问题**：
- "狗咬人" 和 "人咬狗" 语义完全不同
- 如果没有位置信息，Transformer 无法区分
- Attention 计算：$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
- $QK^T$ 的计算与顺序无关！

**解决方案**：给每个位置添加一个独特的"位置编码"

### 3.2 绝对位置编码 vs 相对位置编码

| 类型 | 代表 | 特点 |
|------|------|------|
| 绝对位置编码 | Sinusoidal | 每个位置一个向量，与内容相加 |
| 相对位置编码 | RoPE | 通过旋转操作融入相对位置信息 |

**选择依据**：
- 2017-2020 年：Sinusoidal 为主（BERT、GPT-2）
- 2021 年后：RoPE 为主（LLaMA、Qwen、Mistral）

---

### 3.3 Sinusoidal 位置编码

#### 3.3.1 算法原理

来自论文 "Attention Is All You Need" (2017)，这是 Transformer 论文的核心贡献之一。

**编码公式**：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中:
  pos: 位置 (0, 1, 2, ...)
  i:   维度索引 (0, 1, 2, ..., d_model/2 - 1)
  d_model: 模型维度
```

**直观理解**：
- 偶数维度用 sin，奇数维度用 cos
- 低频维度（i 小）：周期长，可以编码远距离位置
- 高频维度（i 大）：周期短，可以编码近距离位置

```
维度 0 (i=0): sin(pos/10000^0) = sin(pos)        周期 2π
维度 1 (i=0): cos(pos/10000^0) = cos(pos)        周期 2π
维度 2 (i=1): sin(pos/10000^(2/512))             周期较长
维度 3 (i=1): cos(pos/10000^(2/512))
...
维度 510 (i=255): sin(pos/10000^(510/512))       周期很短
维度 511 (i=255): cos(pos/10000^(510/512))
```

#### 3.3.2 为什么这样设计？

1. **唯一性**：每个 (pos, i) 组合对应唯一的编码值
2. **外推性**：可以推广到训练时未见过的位置（因为是数学公式计算出来的）
3. **相对位置**：两个位置的编码存在线性关系

数学证明：
```
sin(a)cos(b) - cos(a)sin(b) = sin(a-b)
```
这意味着相对位置可以通过线性变换获得！

> **拓展思考**：为什么除以 10000？
> - 如果不用 10000，直接用 pos，sin(pos) 的周期是 2π，对于长序列，相邻位置的 sin 值变化太快
> - 10000 是一个经验值，让不同维度的周期从短到长覆盖足够广的范围

#### 3.3.3 Sinusoidal 位置编码实现要点

**所在文件**：[model/embedding.py](../../../model/embedding.py)

**需要补全的代码位置**：
- `PositionalEncoding.__init__` 方法（第84-99行）
- `PositionalEncoding._create_pe` 方法（第101-122行）
- `PositionalEncoding.forward` 方法（第122-135行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 调用内部方法 `_create_pe` 来预计算位置编码矩阵
   - 将计算结果注册为 buffer（不参与梯度更新）
   - buffer 的形状为 `[max_len, d_model]`

2. **创建位置编码矩阵方法中**：
   - 第一步：创建一个从0到max_len-1的位置索引向量，形状为 `[max_len, 1]`
   - 第二步：计算维度除数，公式为 `10000^(-2i/d_model)`，其中 i 取偶数值（0, 2, 4, ...），形状为 `[d_model // 2]`
     - 可以使用指数运算实现：`exp(arange(0, d_model, 2) * -log(10000.0) / d_model)`
   - 第三步：计算正弦和余弦编码
     - 位置向量与除数相乘得到角度
     - 偶数维度（步长2）取正弦值
     - 奇数维度（步长2）取余弦值
   - 将结果存入形状为 `[max_len, d_model]` 的矩阵

3. **前向传播方法中**：
   - 获取输入张量的序列长度
   - 从预计算的位置编码矩阵中取出对应长度的编码
   - 通过广播机制将位置编码加到输入上
   - 返回添加位置编码后的张量

---

### 3.4 RoPE (旋转位置编码)

#### 3.4.1 算法原理

来自论文 "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)，由苏剑林等人提出。目前是现代 LLM 的主流选择。

**核心思想**：不直接给 Q/K 添加位置编码，而是**旋转**它们！

**数学推导**：

对于二维向量 $(x_1, x_2)$，位置 $m$ 的旋转为：
$$
\begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

展开：
$$
x_1' = x_1 \cos(m\theta) - x_2 \sin(m\theta)
$$
$$
x_2' = x_1 \sin(m\theta) + x_2 \cos(m\theta)
$$

**关键性质**：相对位置不变性

$$
\langle f_q(m), f_k(n) \rangle = \langle q, R_{m-n}k \rangle
$$

即：两个 token 的注意力分数**只取决于相对位置 (m-n)**，而非绝对位置！

#### 3.4.2 为什么需要 RoPE？

1. **更好的长度外推**：外推到更长序列时效果更好
2. **更高效的相对位置编码**：在 attention 计算中自然融入相对位置
3. **现代 LLM 标准**：LLaMA、Qwen、Mistral 等都使用 RoPE

> **拓展**：RoPE vs Sinusoidal
> - Sinusoidal：将位置编码直接加到输入（$x + PE$）
> - RoPE：将位置信息"编织"进 Q 和 K（旋转后的 $q_m, k_n$ 做 attention）
> - RoPE 的 attention 更直接地建模了相对位置关系

#### 3.4.3 RoPE 实现要点

**所在文件**：[model/embedding.py](../../../model/embedding.py)

**需要补全的代码位置**：
- `RoPE.__init__` 方法（第163-179行）
- `RoPE._compute_cos_sin` 方法（第181-206行）
- `RoPE.rotate_half` 静态方法（第208-229行）
- `RoPE.apply_rotary_pos_emb` 静态方法（第231-264行）
- `RoPE.forward` 方法（第266-285行）

**实现步骤**：

1. **初始化方法中**：
   - 调用父类的初始化方法
   - 计算频率倒数：`inv_freq[i] = base^(-2i/dim)`，其中 i 取偶数值
   - 将其注册为 buffer，形状为 `[dim // 2]`

2. **计算 cos 和 sin 方法中**：
   - 接收位置ID张量和序列长度
   - 第一步：计算频率矩阵
     - 用位置ID（形状 `[batch, seq]`）与频率倒数（形状 `[dim//2]`）做外积
     - 结果形状为 `[batch, seq, dim//2]`
   - 第二步：扩展维度
     - 将频率矩阵在最后一维复制拼接
     - 形状变为 `[batch, seq, dim]`
   - 第三步：计算正弦和余弦值
     - 分别调用 cos 和 sin 函数
   - 返回两个张量

3. **旋转操作中**：
   - 接收形状为 `[..., dim]` 的张量
   - 将最后一维分成前后两半
   - 前半部分保持不变，后半部分取负
   - 将取负的后半部分与前半部分交换位置后拼接
   - 最终实现 `[-x2, x1]` 的效果

4. **应用旋转位置编码方法中**：
   - 接收 Q、K、cos、sin 四个张量
   - 先将 cos 和 sin 的维度扩展以便与 Q、K 的 head 维度对齐
   - 对 Q 应用旋转公式：`q * cos + rotate_half(q) * sin`
   - 对 K 应用同样的旋转公式
   - 返回旋转后的 Q 和 K

5. **前向传播方法中**：
   - 接收 Q、K 和位置ID
   - 确定序列长度和设备信息
   - 调用 `_compute_cos_sin` 计算 cos 和 sin
   - 调用 `apply_rotary_pos_emb` 应用旋转
   - 返回旋转后的 Q 和 K

---

## 代码补全位置汇总

### 文件 1: [tokenizer/bpe_tokenizer.py](../../../tokenizer/bpe_tokenizer.py)

| 方法 | 行号 | 功能 |
|------|------|------|
| `BaseTokenizer.__init__` | 40-73 | 初始化词表和特殊 token |
| `train` | 64-110 | 训练 BPE，构建词表和合并规则 |
| `encode` | 112-164 | 编码文本为 token IDs |
| `decode` | 166-205 | 解码 token IDs 为文本 |
| `_pretokenize` | 207-217 | 预分词，将文本拆分为单词 |
| `_postprocess` | 219-229 | 后处理，清理分词标记 |
| `get_vocab` | 231-233 | 获取词表 |
| `tokenize` | 235-237 | 分词，返回 token 字符串列表 |
| `save` | 239-248 | 保存分词器到文件 |
| `load` | 250-253 | 从文件加载分词器 |

### 文件 2: [tokenizer/byte_level_tokenizer.py](../../../tokenizer/byte_level_tokenizer.py)

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__` | 41-57 | 初始化字节映射 |
| `_create_bytes_to_unicode` | 59-75 | 创建字节到 Unicode 映射表 |
| `_bytes_to_unicode` | 77-100 | 将文本转为字节级 Unicode |
| `_unicode_to_bytes` | 102-127 | 将 Unicode 转回文本 |
| `train` | 129-147 | 训练字节级 BPE |
| `encode` | 149-179 | 编码文本 |
| `decode` | 181-207 | 解码文本 |
| `save` | 209-211 | 保存分词器 |
| `load` | 213-216 | 加载分词器 |

### 文件 3: [model/embedding.py](../../../model/embedding.py)

| 类 | 方法 | 行号 | 功能 |
|------|------|------|------|
| `TokenEmbedding` | `__init__` | 33-45 | 创建嵌入矩阵 |
| `TokenEmbedding` | `forward` | 47-59 | 查表转换 |
| `PositionalEncoding` | `__init__` | 84-99 | 预计算位置编码 |
| `PositionalEncoding` | `_create_pe` | 101-122 | 创建编码矩阵 |
| `PositionalEncoding` | `forward` | 122-135 | 添加位置编码 |
| `RoPE` | `__init__` | 163-179 | 预计算频率 |
| `RoPE` | `_compute_cos_sin` | 181-206 | 计算 cos 和 sin |
| `RoPE` | `rotate_half` | 208-229 | 旋转操作 |
| `RoPE` | `apply_rotary_pos_emb` | 231-264 | 应用旋转 |
| `RoPE` | `forward` | 266-285 | 完整流程 |

---

## 练习

### 对实验报告的要求

- 基于 markdown 格式来完成，以文本方式为主
- 填写各个基本练习中要求完成的报告内容
- 列出你认为本实验中重要的知识点，以及与对应的 LLM 原理中的知识点，并简要说明你对二者的含义、关系、差异等方面的理解

### 练习 1：理解 BPE 训练流程

阅读 `tokenizer/bpe_tokenizer.py` 中的 `train` 方法，结合 BPE 算法原理，回答以下问题：

1. BPE 训练的第一步是构建初始词表，请说明初始词表包含哪些内容？
2. 在迭代合并过程中，如何统计字符对的频率？为什么需要乘以单词的频率？
3. 合并规则（merges）的作用是什么？在编码时如何应用这些规则？

### 练习 2：理解字节级分词的独特之处

阅读 `tokenizer/byte_level_tokenizer.py`，思考并回答：

1. 字节级分词与普通 BPE 的核心区别是什么？
2. 为什么字节级分词可以做到"零 OOV"？
3. `_bytes_to_unicode` 和 `_unicode_to_bytes` 这两个方法分别用于什么场景？

### 练习 3：理解 TokenEmbedding 的作用

阅读 `model/embedding.py` 中的 `TokenEmbedding` 类，回答：

1. 为什么 Transformer 需要将 token ID 转换为向量？
2. 嵌入矩阵的形状 `[vocab_size, hidden_size]` 含义是什么？
3. 如果词表大小为 50000，隐藏维度为 768，嵌入矩阵有多少参数？
4. 本实验中我们手动创建了嵌入矩阵（使用 `nn.Parameter`），而不是直接调用 `nn.Embedding`。请思考：这两种方式本质上有何异同？为什么手动实现能帮助我们更好理解嵌入层的原理？

### 练习 4：理解位置编码的必要性

思考并回答：

1. 为什么 Transformer 的 Self-Attention 无法区分 "狗咬人" 和 "人咬狗"？
2. Sinusoidal 位置编码的公式 $PE(pos, 2i) = sin(pos / 10000^{2i/d})$ 中，$10000$ 这个数字的作用是什么？如果改成 $100$ 或 $1000000$ 会有什么影响？
3. RoPE 与 Sinusoidal 相比，有什么优势？

### 练习 5：验证你的实现

运行以下测试代码，验证你的实现是否正确：

```python
# 测试 BPE
from tokenizer import BPETokenizer

texts = ["hello world", "world hello", "hello hello world"]
tokenizer = BPETokenizer()
tokenizer.train(texts, vocab_size=50)

ids = tokenizer.encode("hello")
print("BPE 编码结果:", ids)

decoded = tokenizer.decode(ids)
print("BPE 解码结果:", decoded)
assert decoded == "hello", "解码应等于原文"

# 测试 Token Embedding
import torch
from model.embedding import TokenEmbedding

emb = TokenEmbedding(vocab_size=10000, hidden_size=768)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
output = emb(input_ids)
assert output.shape == (1, 5, 768), f"形状应为 (1, 5, 768)，实际为 {output.shape}"
print("TokenEmbedding 输出形状:", output.shape)

# 测试位置编码
from model.embedding import PositionalEncoding

pe = PositionalEncoding(d_model=512, max_len=100)
x = torch.randn(2, 10, 512)
x_pe = pe(x)
assert x_pe.shape == (2, 10, 512), f"形状应为 (2, 10, 512)，实际为 {x_pe.shape}"
print("PositionalEncoding 输出形状:", x_pe.shape)

# 测试 RoPE
from model.embedding import RoPE

rope = RoPE(dim=64, max_position=2048)
q = torch.randn(2, 8, 10, 64)
k = torch.randn(2, 8, 10, 64)
position_ids = torch.arange(10).unsqueeze(0).repeat(2, 1)

q_rope, k_rope = rope(q, k, position_ids)
assert q_rope.shape == (2, 8, 10, 64), f"形状应为 (2, 8, 10, 64)，实际为 {q_rope.shape}"
print("RoPE 输出形状:", q_rope.shape)

print("\n所有测试通过！")
```

---

## 延伸阅读

### 原始论文

1. **BPE**: "Neural Machine Translation of Rare Words with Subword Units" - Sennrich et al., 2015
2. **字节级 BPE**: "GPT-2: Language Models are Unsupervised Multitask Learners" - Radford et al., 2019
3. **Sinusoidal PE**: "Attention Is All You Need" - Vaswani et al., 2017
4. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" - Su et al., 2021

### 实践参考

- HuggingFace Tokenizers 库（https://github.com/huggingface/tokenizers）
- tiktoken (OpenAI 的 BPE 实现，https://github.com/openai/tiktoken)
- sentencepiece (Google 的分词库，https://github.com/google/sentencepiece)

---

## 常见问题 FAQ

**Q1: BPE 和 WordPiece 有什么区别？**
A: BPE 合并频率最高的字符对；WordPiece 合并使语言模型困惑度最低的字符对。WordPiece 需要训练一个语言模型来评估合并收益，计算成本更高。

**Q2: RoPE 比 Sinusoidal 好吗？**
A: 各有优势。RoPE 在长度外推上表现更好，是现代 LLM 的主流选择。但 Sinusoidal 是 Transformer 原始论文的方案，简洁且效果稳定。

**Q3: Embedding 层需要训练吗？**
A: 可以是可学习的（大多数情况），也可以是固定的（如 Word2Vec 预训练）。可学习的嵌入层通过反向传播更新。

**Q4: 位置编码可以加在 Q/K/V 哪个位置？**
A: Sinusoidal 加在输入（与 token embedding 相加）；RoPE 通常应用在 Q 和 K 上（通过旋转操作）。

**Q5: 为什么 RoPE 要用旋转而不是直接加位置编码？**
A: 直接相加只是给向量"加上"位置信息，而旋转是"乘上"位置信息。旋转操作在数学上更自然地表达了相对位置关系，而且可以通过数学公式证明 attention 分数只依赖于相对位置。
