# L01: Tokenizer 与 Embedding

> **课程定位**：这是 Intro2LLM 课程的**第一个实验**，也是整个 LLM 系统的“最小可用入口”。在本实验中，你要把一段人类可读文本，稳定、可复现地变成模型可消费的张量表示，并为后续 Transformer 结构做好准备。

本次课程中，我们按照以下逻辑顺序进行：

- 先搭出**最小可用链路**：文本 -> token IDs -> 向量 -> 位置信息
- 再逐层抽象：BaseTokenizer -> BPE -> Byte-Level BPE
- 最后把“能跑”变成“可维护、可复现”：批处理、保存/加载、测试与调试路径

---

## 实验目的

本实验主要讲解分词器（Tokenizer）和嵌入层（Embedding）的原理与实现。我们的 LLM 系统需要将人类可读的文本转换为计算机可处理的数字表示，再通过嵌入层获得向量表示。为了让模型理解文本的顺序信息，还需要引入位置编码机制。

### 本章你将学到

- **分词算法**：理解 BPE 与字节级 BPE 的训练逻辑、编码/解码逻辑、合并规则的作用
- **工程抽象**：为什么需要 `BaseTokenizer` 统一接口；batch/padding/mask 如何协作
- **嵌入层**：离散 ID 如何变成可学习查表；梯度如何回传到权重
- **位置编码**：Sinusoidal 的外推性与加法注入；RoPE 的旋转注入与相对位置特性
- **调试能力**：如何用最小样例手算验证、如何用测试/断点定位 shape 与广播错误

### 本章你要推进的项目进度（以仓库接口为准）

- **完成** `tokenizer/base_tokenizer.py`、`tokenizer/bpe_tokenizer.py`、`tokenizer/byte_level_tokenizer.py` 的完整实现
- **完成** `model/embedding.py` 中 `TokenEmbedding` 的从 0 实现（理解查表本质）
- **完成** `model/embedding.py` 中 `PositionalEncoding`（正余弦）与 `RoPE`（旋转位置编码）的实现

---

## 实验要求与约束（务必先读）

- 本课程提供的是“**只有接口、没有实现**”的代码框架：你看到的 `pass` 就是需要你补齐的实现。
- 你必须在已有函数/类声明上补全实现：**不允许改动接口签名**（参数列表、返回值类型、类名/方法名）。
- 实验指导书不提供任何可直接 copy-paste 的实现代码；只允许自然语言伪代码与步骤描述。
- 你的实现要与仓库内文档注释描述一致；并且能支撑后续 lesson 使用。
- 优先保证正确性与可读性，再考虑性能优化（优化必须不改变行为）。

---

## 理论预备：你在这节课里到底在“表示”什么

在实现细节之前，先明确 lesson1 的四个核心对象，以及它们的“信息类型”：

1. **Token（符号）**：文本被切分后的最小处理单位（可能是字、子词、字节映射字符等）。它仍然是“符号”，不具备连续几何意义。
2. **Token ID（整数）**：token 在词表中的索引，是离散整数。它的数值大小没有语义（ID=5 不比 ID=6 更大或更重要）。
3. **Embedding（向量）**：把离散 token 映射到连续向量空间，向量之间的距离/夹角才开始承载语义与统计关联。
4. **Position（位置）**：序列顺序信息。Transformer 的注意力对“集合”敏感、对“顺序”不敏感，因此必须显式注入位置信息。

你可以把整个链路看成一次“表示类型升级”：

符号（text/token） -> 离散索引（ID） -> 连续表示（embedding） -> 带序列结构的连续表示（position-aware embedding）

后续所有模型结构（注意力、前馈、残差、归一化）都是在最后一种对象上运行的。

---

## 项目组成与执行流

### lesson1 相关目录结构

你主要会在这些文件中工作：

- `tokenizer/base_tokenizer.py`：统一接口、batch 编码、保存/加载、特殊 token 管理
- `tokenizer/bpe_tokenizer.py`：字符级 BPE（训练 merges、按 merges 编码/解码）
- `tokenizer/byte_level_tokenizer.py`：字节级 BPE（字节<->Unicode 映射 + 复用 BPE）
- `model/embedding.py`：TokenEmbedding、Sinusoidal 位置编码、RoPE

### 执行流

本实验最终需要你能够完成以下链路：

1. 文本输入 `text`
2. `tokenizer.encode(text)` 得到 `token_ids`（整数列表/张量）
3. `TokenEmbedding(token_ids)` 得到 `token_vectors`（浮点张量）
4. 位置编码：
   - 方案 A：`PositionalEncoding(token_vectors)` 直接相加注入
   - 方案 B：RoPE 注入到注意力的 Q/K（本实验实现 RoPE 本体，后续注意力会调用）

你应该始终把“输入输出契约”写在脑中：

- Tokenizer：字符串 <-> ID 序列（可保存、可加载、批处理可对齐）
- Embedding：ID 张量 -> 向量张量（梯度可回传）
- PE/RoPE：不改变 batch/seq 维，只注入位置信息（数值稳定）

---

## 你将实现的接口清单

### 文件 1：`tokenizer/base_tokenizer.py`

你需要补全：

- `BaseTokenizer.__init__`
- `BaseTokenizer.vocab_size`
- `BaseTokenizer.encode_batch`
- `BaseTokenizer.save`
- `BaseTokenizer.load`
- `BaseTokenizer.tokenize`
- `BaseTokenizer.convert_tokens_to_ids`
- `BaseTokenizer.convert_ids_to_tokens`

### 文件 2：`tokenizer/bpe_tokenizer.py`

你需要补全：

- `BPETokenizer.train`
- `BPETokenizer.encode`
- `BPETokenizer.decode`
- `BPETokenizer._pretokenize`
- `BPETokenizer._postprocess`
- `BPETokenizer.get_vocab`
- `BPETokenizer.tokenize`
- `BPETokenizer.save`
- `BPETokenizer.load`

### 文件 3：`tokenizer/byte_level_tokenizer.py`

你需要补全：

- `ByteLevelTokenizer.__init__`
- `ByteLevelTokenizer._create_bytes_to_unicode`
- `ByteLevelTokenizer._bytes_to_unicode`
- `ByteLevelTokenizer._unicode_to_bytes`
- `ByteLevelTokenizer.train`
- `ByteLevelTokenizer.encode`
- `ByteLevelTokenizer.decode`
- `ByteLevelTokenizer.save`
- `ByteLevelTokenizer.load`

### 文件 4：`model/embedding.py`

你需要补全：

- `TokenEmbedding.__init__`
- `TokenEmbedding.forward`
- `PositionalEncoding.__init__`
- `PositionalEncoding._create_pe`
- `PositionalEncoding.forward`
- `RoPE.__init__`
- `RoPE._compute_cos_sin`
- `RoPE.rotate_half`
- `RoPE.apply_rotary_pos_emb`
- `RoPE.forward`

---

## 第一部分：Tokenizer

### 1.1 为什么需要分词器

神经网络的输入是张量，不是字符串。分词器至少要解决两件事：

- **离散化**：把文本切成 token，并映射成整数 ID
- **封装约定**：特殊 token（pad/eos/unk/bos）在全项目内必须一致，否则训练与推理会“协议不一致”

从效果角度，分词器还决定了：

- 序列长度（影响算力与上下文容量）
- 词表大小（影响参数量与 softmax 计算）
- OOV 处理能力（能否覆盖任意输入）

### 1.1.1 “切分粒度”与三种典型方案

把文本切成 token，本质是选择一种“离散化坐标系”。常见粒度对比：

- **词级（word-level）**：token 是完整词
  - 优点：语义直观、序列短
  - 缺点：OOV 严重（新词/拼写变化/多语言），词表巨大且稀疏
- **字符级（char-level）**：token 是字符
  - 优点：几乎无 OOV（只要字符集覆盖）
  - 缺点：序列变长，长程依赖更难学；对多语言字符集仍可能膨胀
- **子词级（subword-level）**：token 是高频片段（BPE/WordPiece/Unigram 等）
  - 优点：在“词级语义”与“字符级覆盖”之间折中，现代 LLM 的主流
  - 缺点：训练与实现更复杂；分词规则会影响下游表现与可解释性

Byte-Level BPE 可以看作“把字符级底座替换为字节级底座”的子词级方案：以覆盖性换取可读性与一定长度开销。

### 1.1.2 一个工程视角：Tokenizer 是“协议”，不是“预处理小工具”

Tokenizer 需要全系统一致的原因在于：它直接定义了训练数据的离散分布与模型输入空间。

- 词表 ID 的语义必须全局一致，否则同一个 ID 在不同模块含义不同会导致不可恢复的训练错误
- 特殊 token（pad/eos/unk/bos）的策略必须一致，否则会出现：
  - 模型把 padding 当成有效内容注意
  - 生成时无法正确停止（eos 不一致）
  - 训练损失被“无意义 token”污染

### 1.2 BaseTokenizer：统一项目中的tokenizer接口

`BaseTokenizer` 是我们整个项目所依赖tokenizer的基类，后续任何训练/推理代码都只依赖它的接口，不关心你是 BPE 还是 Byte-Level。

#### 1.2.1 `BaseTokenizer.__init__`（特殊 token 与词表）

目标：你必须保证以下事实成立：

- `vocab` 是 token 字符串到 ID 的映射
- `inverse_vocab` 是 ID 到 token 字符串的映射（用于 decode）
- `pad/eos/unk` 的字符串与 ID 总是可用（即便传入空词表）
- `bos` 可选，但如果配置了也必须有合法 ID

自然语言伪代码：

1. 读取 `special_tokens`，若为空则填入默认配置（pad/eos/unk，bos 可为 None）
2. 读取 `vocab`，若为空则创建空映射
3. 依次检查每个特殊 token 的字符串是否在 `vocab` 中：
   - 如果在：记录其 ID
   - 如果不在：为它分配一个新的 ID（通常是当前词表大小），写入 `vocab`
4. 基于更新后的 `vocab` 构建 `inverse_vocab`（确保一一对应）
5. 写入并缓存各特殊 token 的 ID（pad/eos/unk/bos）

你需要在报告中解释：为什么“特殊 token 必须加入词表”是一个协议问题，而不是实现细节。

#### 1.2.2 `encode_batch`（padding、truncation、attention_mask）

目标：批处理必须把不同长度序列对齐，并生成 mask。

自然语言伪代码：

1. 对每个文本调用 `encode` 得到 ID 序列列表
2. 决定目标长度：
   - 若 `padding=False`：目标长度就是各自长度（不对齐）
   - 若 `padding=True` 或 `padding=\"longest\"`：目标长度取本 batch 最大长度
   - 若 `padding=\"max_length\"`：目标长度取 `max_length`（必须提供）
3. 如启用 `truncation=True`：
   - 若 `max_length` 不存在：应给出清晰的错误提示或约定行为
   - 若序列超长：按约定截断（通常保留前部，必要时保留 eos）
4. 对每条序列：
   - 末尾补足 `pad_token_id` 直到目标长度
   - 生成 `attention_mask`：原始有效位置为 1，padding 位置为 0
5. 若 `return_tensors` 指定：
   - 把 `input_ids`、`attention_mask` 转成对应张量/数组

你需要在报告中说明：为什么 mask 必须与 padding 一致，以及后续注意力如何用它屏蔽 padding。

#### 1.2.3 `save` / `load`（可复现）

目标：保存后加载应恢复同样的 encode 行为。

自然语言伪代码：

- `save`：
  1. 若目录不存在则创建
  2. 写出 `vocab.json`（token -> id）
  3. 写出 `special_tokens.json`（pad/eos/unk/bos 的字符串）
  4. 写出 `tokenizer_config.json`（你认为必要的其它元信息，例如类型名、版本号等）
- `load`：
  1. 读入上述文件并校验字段齐全
  2. 调用构造函数恢复实例
  3. 验证 `vocab` 与 `inverse_vocab` 一致性

提示：持久化是“工程正确性”的核心之一。没有它，你无法稳定复现实验或调试。

---

## 第二部分：BPE（字符级“合并规则”的训练与执行）

### 2.1 BPE 的直觉

把文本拆到字符级，永远不会 OOV，但序列太长、语义太弱。BPE 通过“频率驱动的合并”在两者之间折中：

- 高频片段（例如常见词根、词缀、常见词）会被合并成更长 token
- 低频片段仍可退化为更短 token（甚至单字符）

### 2.1.1 BPE 是什么：从压缩到子词词表

BPE（Byte Pair Encoding）最初是数据压缩领域的一个思想：反复把序列中最常见的相邻符号对合并成新符号，从而用更短的符号序列表示原始数据。

迁移到 NLP 后：

- “符号”可以是字符（char-BPE）或字节映射字符（byte-level）
- “合并规则”就是 merges 列表
- merges 的顺序就是一个可执行的分词程序

### 2.1.2 为什么 BPE 能缓解 OOV

OOV 的根源是“词级离散化过粗”：一旦词表没见过该词，就无法编码。

BPE 的策略是：任何新词都能被拆解成更小的子词/字符序列；同时高频片段被合并以缩短序列。

所以它不是“消灭 OOV”，而是把 OOV 从“整个词”降级到“更小单位”，使编码永远可退化到基础符号。

### 2.1.3 BPE 的两个阶段：训练 merges 与执行 merges

概念上分清两件事：

- **训练（learn merges）**：从语料统计出一串合并规则（merges），并确定最终词表
- **编码（apply merges）**：对新文本按 merges 的优先级执行合并，得到 token 序列

这类似“编译器/链接器”与“运行时”的分离：训练是构建规则，encode 是执行规则。

### 2.2 `BPETokenizer.train`（训练 merges 列表）

训练阶段的目标是得到两样东西：

- 词表 `vocab`
- 合并规则 `merges`（一个按顺序排列的相邻 token 对列表）

自然语言伪代码：

1. 初始化：
   - 从语料中收集基础符号集合（通常是字符集合）
   - 把特殊 token 加入词表（遵循 BaseTokenizer 协议）
2. 统计“单词频率表”：
   - 用 `_pretokenize` 把文本拆成基本单元（常见做法是按单词/标点切分）
   - 对每个基本单元统计出现次数
   - 把每个基本单元表示成“符号序列”（例如字符序列）
3. 迭代合并直到达到 `vocab_size` 或无法继续：
   - 统计所有相邻符号对的频率（必须按“基本单元频率”加权）
   - 选出频率最高的符号对作为本轮 merge
   - 若最高频率 < `min_frequency`：停止
   - 记录 merge 到 `merges`
   - 在所有基本单元的符号序列中执行该 merge（把相邻的 A、B 替换成 AB）
   - 将新符号 AB 加入词表
4. 训练结束：
   - 建立 `merges_dict`（pair -> 顺序索引），用于快速比较 merge 优先级

你必须在报告里回答：为什么 merge 的顺序会影响最终编码结果。

### 2.2.1 BPE 的优缺点（理论层面）

优点：

- **覆盖性强**：能编码新词、拼写变化、组合词
- **词表可控**：通过 `vocab_size` 控制模型输入空间大小
- **序列长度折中**：比字符级短、比词级长，通常是可接受的工程折中

缺点与风险：

- **贪心/局部性**：训练是局部频率驱动，未必等价于“最优语言单元”
- **可解释性与可读性**：subword 边界可能不符合人类直觉
- **语言与语料偏置**：merges 强依赖训练语料分布；换领域可能分词质量下降
- **实现复杂度**：需要维护 merges 优先级、处理空格/标点等边界

### 2.2.2 常见变体：WordPiece 与 Unigram（了解即可）

你在本实验实现的是 BPE，但需要知道它不是唯一方案：

- WordPiece：更常见于早期 BERT 系列，训练目标与合并策略略不同，常配合概率/似然视角
- Unigram：从一个较大候选词表开始，用概率模型删减词表，分词时常用 Viterbi 找最优分解

现代 LLM 里你会同时见到 BPE、SentencePiece-Unigram、Byte-Level BPE 等不同组合。

### 2.3 `BPETokenizer.encode`（把文本编码成 ID）

自然语言伪代码：

1. 用 `_pretokenize` 将文本拆成基本单元列表
2. 对每个基本单元执行 BPE：
   - 初始化为最细粒度符号序列（例如字符序列）
   - 反复执行合并，直到没有可应用的 merge：
     - 在当前符号序列中找出所有相邻符号对
     - 找到其中“在 merges 中出现且优先级最高”的那一个
     - 将该 pair 在序列中按位置合并（注意：合并后邻接关系会变化，需要继续扫描/更新）
3. 得到 token 字符串列表后，映射到 ID：
   - token 在词表中：取对应 ID
   - 不在：用 `unk_token_id`
4. 若 `add_special_tokens=True`：
   - 若定义了 `bos`：在开头加 bos
   - 在结尾加 eos（按项目约定）
5. 若启用 `truncation` 且超过 `max_length`：按约定截断

建议：先用极小样例（几个词、几个 merges）手算一次 encode，确保你的合并逻辑与顺序一致。

### 2.4 `BPETokenizer.decode`（把 ID 还原文本）

自然语言伪代码：

1. 把每个 ID 映射回 token 字符串（缺失则用 unk）
2. 若 `skip_special_tokens=True`：过滤掉 pad/eos/bos/unk 等特殊 token（按配置）
3. 拼接 token 字符串得到中间文本
4. 调用 `_postprocess` 清理分词标记与空格规范化

注意：decode 的目标通常是“可读性”和“近似还原”，不一定保证字节级完全一致（Byte-Level 才追求强可逆）。

---

## 第三部分：Byte-Level BPE（让任何 Unicode 都可编码）

### 3.1 为什么需要字节级

字符级 BPE 在多语言、稀有符号、emoji 等场景仍可能出现覆盖问题或预处理脆弱性。Byte-Level 的核心策略是：

- 先把文本转成 UTF-8 字节序列（0..255）
- 在字节空间做 BPE（基础词表固定为 256）
- 通过“字节到可打印 Unicode”的双射把字节序列表示成可处理字符串

这样做的结果是：理论上不会 OOV，因为任何字符串最终都能表示为字节。

### 3.1.1 UTF-8 与“字节级”的含义

UTF-8 是变长编码：

- 常见 ASCII 字符通常是 1 个字节
- 许多非拉丁字符可能是 2-4 个字节
- emoji 通常是 4 个字节

Byte-Level 的核心观点是：不要直接在“字符表面形态”上做 tokenization，而是在更底层、稳定的 UTF-8 字节序列上做合并与建模。

这带来两点直接影响：

- 覆盖性：任何 Unicode 字符都能编码
- 长度：某些字符会展开为多个字节，序列长度可能上升（但 BPE 合并会部分抵消）

### 3.2 `_create_bytes_to_unicode`（可逆映射）

目标：构造一个映射表，把每个字节值映射到一个可打印 Unicode 字符，且映射必须可逆。

自然语言伪代码：

1. 选取一组“天然可打印”的字节范围，直接映射到同码点字符（例如常见 ASCII 可打印区间）
2. 对剩余不可打印/冲突字节：
   - 依次分配到某个不常用、但稳定可表示的 Unicode 区间
3. 最终得到 256 个字节的一一映射（双射）

报告要求：说明为什么双射（可逆）是 Byte-Level decode 的必要条件。

### 3.2.1 为什么还要“字节 -> Unicode”映射这一步

你也许会问：既然都到字节了，为什么不直接在 bytes 上做分词？

原因是工程接口与实现便利性：

- Python 字符串处理、正则、持久化等更自然地工作在 Unicode 字符串层
- 直接处理原始 bytes 也可以，但会让很多文本操作更麻烦
- GPT-2 风格做法是把 0..255 映射成 256 个可打印字符，让“字节序列”可以安全地塞进字符串处理流程

这一步的本质是“表示层转换”，而不是 tokenization 算法本身。

### 3.3 `_bytes_to_unicode` / `_unicode_to_bytes`

自然语言伪代码：

- `_bytes_to_unicode(text)`：
  1. 将 `text` 按 UTF-8 编码成字节序列
  2. 对每个字节值查映射表得到 Unicode 字符
  3. 拼成新字符串返回（这是“字节级表示”）
- `_unicode_to_bytes(text)`：
  1. 基于映射表构造反向映射（Unicode 字符 -> 字节值）
  2. 将输入字符串逐字符映射回字节序列
  3. 将字节序列按 UTF-8 解码回原始文本（解码错误按约定处理，例如替换策略）

### 3.4 `ByteLevelTokenizer.train/encode/decode`（复用 BPE）

核心思路：Byte-Level 在 BPE 前后各加一层转换。

自然语言伪代码：

- `train`：
  1. 把训练语料逐条做 `_bytes_to_unicode`
  2. 在转换后的语料上调用/复用 BPE 的训练流程（初始词表应覆盖 256 基础符号）
- `encode`：
  1. 原文本 -> `_bytes_to_unicode` -> 在字节表示上做 BPE encode
- `decode`：
  1. BPE decode 得到字节表示字符串 -> `_unicode_to_bytes` -> 原始文本

提示：Byte-Level 的 `_pretokenize` 往往不再强依赖“单词边界”，你需要在实现中保证行为稳定。

### 3.4.1 Byte-Level 的优缺点（工程取舍）

优点：

- **几乎无 OOV**：输入空间完整覆盖
- **跨语言鲁棒**：不依赖空格分词，对多语言混排更稳
- **实现统一**：同一套逻辑适配不同字符集

缺点：

- **可读性差**：token 可能是“奇怪的字符”，调试与人工检查更痛苦
- **长度开销**：某些字符展开成多字节，未合并时序列更长
- **边界行为更敏感**：空格、换行、不可见字符的处理需要特别清晰的约定

---

## 第四部分：Embedding（从“ID”到“向量”的运行时环境）

### 4.1 TokenEmbedding：可学习查表

Tokenizer 输出的是离散 ID，但 Transformer 的计算发生在连续向量空间中。Embedding 的本质是：

- 一个形状为 `[vocab_size, hidden_size]` 的可学习参数矩阵
- 每个 token ID 选择其中一行作为该 token 的向量表示
- 梯度通过损失回传，更新被访问到的行

### 4.1.1 为什么 Embedding 不是 One-Hot

最直观的离散表示是 one-hot：词表大小为 `V` 时，一个 token 是长度 `V` 的向量，只有一个位置为 1，其余为 0。

但 one-hot 有两个问题：

- 维度巨大且稀疏，不适合直接作为神经网络输入
- 语义几何无法表达：任意两个不同 token 的 one-hot 距离几乎相同，无法体现相似性

Embedding 做的事情可以理解为“学习一个从 one-hot 到低维稠密空间的线性映射”。查表只是这种线性映射的高效实现形式。

### 4.1.2 Embedding 学到的是什么

经验上，Embedding 会把在相似上下文中出现的 token 学到相近的向量方向与距离（这与分布式表示假说一致）。

在语言模型里，Embedding 是整个模型的“输入坐标系”。如果坐标系学得不好，后续所有层都要花额外容量去补偿。

#### 4.1.1 `TokenEmbedding.__init__`

自然语言伪代码：

1. 保存 `vocab_size`、`hidden_size`
2. 创建一个可学习权重矩阵（行数为词表大小，列数为隐藏维度）
3. 进行合理初始化（例如小方差随机初始化）
4. 将其注册为模型参数，使优化器能更新它

工程提醒（与后续权重共享相关）：

- 你需要能访问到“embedding 权重矩阵本体”，以便后续做 weight tying（embedding 与 lm_head 共享权重）。

#### 4.1.2 `TokenEmbedding.forward`

自然语言伪代码：

1. 输入 `input_ids`，形状通常为 `[batch, seq_len]`
2. 以 `input_ids` 作为索引，从权重矩阵中取出对应行
3. 输出形状应为 `[batch, seq_len, hidden_size]`

报告要求：解释为什么这一步是“查表”，以及为什么梯度能回到权重矩阵。

---

## 第五部分：位置编码（把“顺序”注入向量）

### 5.1 为什么需要位置编码

注意力机制本身对 token 的排列是置换等变的：如果你只给它一堆向量而不给位置，它无法区分“第 1 个 token”与“第 10 个 token”。位置编码就是给每个位置一个可区分的信号。

### 5.1.1 一个关键事实：Self-Attention 天然不带顺序

从直觉上理解：注意力的核心是“对一组向量做相关性计算并加权求和”。如果只给一组向量而没有任何位置线索，模型看到的更像“集合（set）”，而不是“序列（sequence）”。

因此位置编码不是“锦上添花”，而是让 Transformer 从“无序集合处理器”变成“序列建模器”的必要条件。

### 5.1.2 绝对位置 vs 相对位置（你在后续会反复遇到）

- **绝对位置编码**：每个位置 `pos` 有一个位置向量 `PE(pos)`，与 token 表示直接组合（通常相加）
  - 典型：Sinusoidal、Learned Positional Embedding
- **相对位置编码**：注意力权重或 Q/K 交互显式依赖相对位移 `pos_i - pos_j`
  - 典型：RoPE、ALiBi、相对位置 bias（T5 风格）

现代 LLM 更偏好相对位置类方法（RoPE/ALiBi 等），原因通常是长度外推与长上下文更稳。

### 5.2 Sinusoidal（正余弦位置编码）

特点：

- 固定、不学习参数
- 可外推到训练未见过的长度（在合理范围内）
- 注入方式通常是“加法”：输入向量 + 位置向量

你需要实现的关键点：

- 在初始化阶段预计算 `[max_len, d_model]` 的位置表，并注册为 buffer（不参与梯度）
- 前向时按 `seq_len` 截取前缀并广播相加

### 5.2.1 Sinusoidal 是什么：一张“频率分层”的位置表

Sinusoidal 通过不同频率的 sin/cos 组合让每个位置都有独特相位。直觉上：

- 低频维度随位置变化慢，提供粗粒度位置
- 高频维度随位置变化快，提供细粒度位置

把这些维度拼起来，位置就像被编码成一组多尺度“表盘读数”。

### 5.2.2 为什么这样设计：外推与可计算相对位移

Sinusoidal 的两个常见卖点：

- **无需学习参数**：位置表由公式确定
- **可外推**：理论上你可以计算任意长度的位置编码（只要数值稳定）

此外，在一定条件下，模型可以通过线性组合近似地推断相对位移信息（因为 sin/cos 满足加法公式），这也是它在早期 Transformer 中好用的重要原因。

自然语言伪代码（创建表）：

1. 创建位置索引 `pos = 0..max_len-1`
2. 创建维度索引（只针对偶数维度）
3. 计算每个维度的频率缩放项（基于 `10000` 与维度比例）
4. 偶数维填 sin，奇数维填 cos，拼成完整表

### 5.3 RoPE（旋转位置编码）

RoPE 的直觉：不把位置信号加到输入上，而是对注意力中的 Q/K 做“按位置旋转”，让相对位移体现在内积结果中。现代 LLM 广泛采用。

你需要实现的关键点：

- `inv_freq`：频率倒数向量（维度为 `head_dim/2`）
- `cos/sin`：基于位置与频率计算得到，最终要能广播到 Q/K 的形状
- `rotate_half`：把最后一维分成两半，构造“旋转后的向量半边”
- `apply_rotary_pos_emb`：把旋转公式应用到 q/k

### 5.3.1 RoPE 是什么：把每对维度当作二维向量做旋转

RoPE 的一个常用理解方式：

- 把 `head_dim` 的最后一维按两两分组，每组是一个二维向量 `(x1, x2)`
- 不同维度组对应不同的旋转频率
- 在位置 `pos` 时，对每组二维向量旋转一个角度（角度与 `pos` 成比例）

因此它本质是在 Q/K 的表示空间里引入“位置相关的相位”。

### 5.3.2 为什么 RoPE 能表达相对位置

RoPE 的关键性质是：对 Q/K 施加位置相关旋转后，它们的内积关系会以某种形式只依赖于相对位移 `pos_i - pos_j`（而非绝对位置各自的值）。

直觉上：你把同一个向量分别旋转两次，旋转差值决定了它们的对齐程度；差值对应相对位置。

这就是为什么 RoPE 常被称为“相对位置编码”的原因之一。

### 5.3.3 RoPE 的优缺点与工程参数

优点：

- **相对位置友好**：更符合自回归建模的需求
- **长上下文实践表现好**：被大量现代 LLM 采用
- **参数少**：核心是公式生成的 cos/sin，不需要额外可学习位置表（实现方式不同会略有差异）

缺点与注意点：

- **实现更容易出错**：shape 对齐与广播错误很常见
- **base 等超参数会影响外推**：不同 base 会改变旋转频率分布
- **低精度稳定性**：长序列时必须关注 NaN/Inf 与数值范围

#### 5.3.1 形状对齐检查表（先画再写）

常见注意力输入：

- `q`: `[batch, num_heads, seq_len, head_dim]`
- `k`: `[batch, num_kv_heads, seq_len, head_dim]`

位置相关张量的目标形态：

- `cos/sin`（基础）: `[batch, seq_len, head_dim]`
- `cos/sin`（广播到 heads）: 在 heads 维插入大小为 1 的维度后变为 `[batch, 1, seq_len, head_dim]`

实现前你必须明确：

- 你的 `head_dim` 来自哪里（通常是 q 的最后一维）
- `position_ids` 是 `[batch, seq_len]` 还是 `[seq_len]`，你如何统一处理
- 哪些维度允许广播，哪些维度必须严格相等

#### 5.3.2 数值稳定性要求

在长序列（例如几千 token）和低精度（fp16/bf16）下，RoPE 容易出现数值问题。你需要在实现中确保：

- 不产生 NaN/Inf
- 旋转不会系统性改变向量范数（理论上是正交旋转，范数应保持不变，允许微小误差）

---

## 练习（对实验报告的要求）

### 实验报告格式要求

- 使用 Markdown 完成，以文本说明为主
- 回答所有练习问题
- 列出你认为本实验中重要的知识点，并说明其与 LLM 原理知识点的关系
- 记录你的实现取舍（例如某个边界条件你选择了报错还是自动修正）及理由
- 至少包含一次完整调试记录（失败 -> 定位 -> 修复 -> 验证）

### 练习 1：理解“接口契约”（Tokenizer/Embedding 的协议）

请说明：

- 为什么 `BaseTokenizer` 的存在是“协议层”而不是“代码复用小技巧”
- pad/eos/unk/bos 这四类特殊 token 各自的语义是什么
- 如果不同模块对 eos/pad 的约定不一致，会导致什么训练/推理错误

### 练习 2：手算一次 BPE 合并（拆解 merges 的作用机制）

请用一个极小语料（你自己构造即可）完成 2-3 次 merge 的手算过程，并回答：

- 你统计的最高频 pair 是什么，为什么
- `min_frequency` 提前停止会带来什么影响
- merges 顺序改变会如何影响 encode 结果（给出一个反例）

### 练习 3：Byte-Level 的可逆性

请说明：

- 为什么 `_create_bytes_to_unicode` 必须构造双射
- 你如何验证 `_bytes_to_unicode` 与 `_unicode_to_bytes` 互为逆过程
- Byte-Level 相比字符级 BPE 的主要代价是什么（从序列长度、可读性、预处理等角度回答）

### 练习 4：TokenEmbedding 的本质与梯度路径

请说明：

- 为什么 embedding 是“查表”，而不是普通的线性层
- 为什么梯度会回到 embedding 权重矩阵的“某些行”
- weight tying（embedding 与 lm_head 共享权重）为什么能工作，它依赖什么接口约定

### 练习 5：RoPE 的形状与广播

请说明：

- 你认为 q/k/cos/sin 的目标 shape 分别是什么
- 哪些维度是通过广播匹配的，哪些必须严格相等
- 你如何验证“旋转不改变范数”（允许数值误差）

### 练习 6：验证你的实现

请在报告中附上：

- 你运行的验证命令（pytest 或你自建的最小验证）
- 关键验证结果摘要
- 一个你遇到的 bug：现象、定位思路、修复点、修复后证据

---

## 延伸阅读

- Sennrich et al., 2015: Neural Machine Translation of Rare Words with Subword Units（BPE）
- Vaswani et al., 2017: Attention Is All You Need（Sinusoidal）
- Su et al., 2021: RoFormer: Enhanced Transformer with Rotary Position Embedding（RoPE）
- GPT-2 / RoBERTa 的 Byte-Level BPE 实践资料（理解字节映射动机）

---

## 常见问题 FAQ

### Q1：decode(encode(text)) 为什么不一定完全等于原文？

字符级 BPE 通常以“可读性”和“稳定 tokenization”为主，会引入空格规范化或分词标记；Byte-Level 才更强调强可逆。

### Q2：BPE 训练为什么这么慢？

合并迭代需要反复统计相邻 pair 频率。可以先保证正确性，再考虑用更高效的数据结构或缓存优化（但不能改变行为）。

### Q3：RoPE 最常见的 bug 是什么？

几乎都是 shape/广播对齐错误。解决办法不是“盲改”，而是先写清目标 shape，再逐维对照实际张量。
