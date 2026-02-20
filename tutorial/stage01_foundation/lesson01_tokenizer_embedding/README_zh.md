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

### 本章你要推进的项目进度

- **完成** `tokenizer/base_tokenizer.py`、`tokenizer/bpe_tokenizer.py`、`tokenizer/byte_level_tokenizer.py` 的完整实现
- **完成** `model/embedding.py` 中 `TokenEmbedding` 的从 0 实现（理解查表本质）
- **完成** `model/embedding.py` 中 `PositionalEncoding`（正余弦）与 `RoPE`（旋转位置编码）的实现

---

## 实验要求与约束

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

#### 1.2.1 `BaseTokenizer.__init__` 初始化方法

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
   - 如果不在：为它分配一个新的 ID，写入 `vocab`
4. 基于更新后的 `vocab` 构建 `inverse_vocab`（确保一一对应）
5. 写入并缓存各特殊 token 的 ID（pad/eos/unk/bos）

补充：如何给“新加入的 token”分配 ID

- 你不能假设传入的 `vocab` 的 ID 一定是 `0..N-1` 连续整数（虽然大多数情况下是）。
- 因此，“新 token 的 ID”要满足两个要求：
  - 不与现有任何 ID 冲突（否则 `inverse_vocab` 会覆盖，decode 会不稳定）。
  - 在同一份 `vocab` 上重复初始化时结果可复现（不要依赖不稳定的遍历顺序）。
- 一种常见且稳定的策略是：先收集当前 `vocab` 中已经被占用的 ID 集合；然后选择一个“未被占用的新 ID”（例如大于当前最大 ID 的下一个整数）。你的报告里应说明你选用的规则与理由。

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

#### 1.2.3 `save` / `load`

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

#### 1.2.4 `vocab_size` 访问tokenizer的词表大小

目标：`vocab_size` 必须与 `vocab` 的内容一致，而不是与某个缓存数字一致。

自然语言伪代码：

1. 取出当前 `vocab` 映射中 token 的数量
2. 将其作为词表大小返回

边界提醒：

- 如果你允许 `vocab` 的 ID 不连续，那么 `vocab_size` 的定义仍应是“token 的数量”，而不是“最大 ID + 1”。否则会出现“词表里只有 100 个 token，但最大 ID=10000”的异常行为。

#### 1.2.5 `tokenize`

目标：`BaseTokenizer.tokenize` 在语义上是“把文本切成 token 字符串列表”，但基类通常不知道具体切分策略。

自然语言伪代码（两种合理选择，二选一并在报告中说明）：

- 选择 A（更严格）：在基类中明确抛出“未实现/需子类实现”的错误，避免误用导致 silent bug。
- 选择 B（更宽松）：提供一个“最小可用默认策略”，例如把输入视为一个整体 token（或按空白切分）。这种做法便于写 demo，但要明确它只是兜底，不等价于 BPE/Byte-Level。

工程提醒：

- lesson1 的测试里会通过“写一个最小子类覆盖 tokenize”来验证 `encode_batch` 等基类逻辑，因此你要确保：子类重写 `tokenize` 后，`convert_tokens_to_ids/convert_ids_to_tokens` 等通用函数能稳定工作。

#### 1.2.6 `convert_tokens_to_ids`（OOV 回退到 unk）

目标：把 token 字符串列表映射为 ID 列表，并对 OOV 做一致处理。

自然语言伪代码：

1. 初始化空的输出 ID 列表
2. 遍历输入 tokens：
   - 若 token 在 `vocab` 中：追加对应 ID
   - 否则：追加 `unk_token_id`
3. 返回 ID 列表

边界提醒：

- 不要在这里自动添加 bos/eos/pad；这些是 `encode/encode_batch` 的职责。
- 不要在这里做任何“字符串清理”（例如 strip 或 lower）；清理应属于 tokenize/预处理阶段，否则会造成“同一个 token 在不同路径被不同地改写”。

#### 1.2.7 `convert_ids_to_tokens`（越界/未知 ID 回退到 unk）

目标：把 ID 列表映射为 token 字符串列表，确保 decode 路径永不崩溃。

自然语言伪代码：

1. 初始化空的输出 token 列表
2. 遍历输入 token_ids：
   - 若该 ID 在 `inverse_vocab` 中：追加对应 token 字符串
   - 否则：追加 `unk_token` 的字符串表示
3. 返回 token 列表

边界提醒：

- 这里的“未知 ID 回退”是一个重要的鲁棒性约定：即便上游张量里出现了异常 ID，你也能得到可解释输出，而不是直接报错中断调试。

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

### 2.5 `BPETokenizer._pretokenize`（把文本切成“训练/编码单元”）

目标：为 `train` 与 `encode` 提供一个稳定的“基本单元”序列（常见是单词/标点/空白段）。

自然语言伪代码（推荐实现一个“最小但稳定”的版本）：

1. 对输入文本做最轻量的规范化（可选）：例如把连续空白折叠为单个空格，或保持原样；关键是要在报告中说明你的选择会如何影响 decode 的可读性。
2. 将文本切分成单元列表，要求：
   - 单元切分对同一输入是确定的（deterministic）
   - 结果不产生空串单元（或产生后会被过滤掉），避免训练/编码出现“空 token”
3. 返回单元列表

工程建议：

- lesson1 的基础测试为了避免空格复杂度，会使用“无空格字符串”来测试 BPE 的核心 merge 行为。因此 `_pretokenize` 的一个安全选择是：当文本不含空白时，直接把整段作为一个单元；当含空白时，再按空白切分并丢弃空串。

### 2.6 `BPETokenizer._postprocess`（把 token 拼接结果恢复可读文本）

目标：把 decode 阶段拼接出来的中间文本，按你在 `_pretokenize` 中的约定恢复成更自然的可读形式。

自然语言伪代码：

1. 接收中间文本（它来自 token 字符串的直接拼接）
2. 根据你的分词标记约定做清理（如果你引入了“词首标记/子词标记”，在这里移除它们）
3. 进行空白规范化（可选）：例如把多空格折叠、去掉首尾空格
4. 返回清理后的文本

边界提醒：

- 如果你在 encode 侧没有引入任何额外标记（例如完全按字符/子词直接拼接），那么 `_postprocess` 可以非常简单，但仍应保证它不会把有效字符误删。

### 2.7 `BPETokenizer.get_vocab`（暴露词表用于检查与保存）

目标：返回当前词表映射，供外部检查/调试/保存使用。

自然语言伪代码：

1. 返回 `vocab`（你可以选择返回原对象或返回一个拷贝）
2. 若返回原对象，请在报告中说明：外部修改词表可能破坏分词器一致性；你如何在项目中避免这种误用

### 2.8 `BPETokenizer.tokenize`（只做合并，不做 ID 映射）

目标：复用与 `encode` 相同的 BPE 合并逻辑，但输出 token 字符串列表，而不是 ID。

自然语言伪代码：

1. 用 `_pretokenize` 将文本切成基本单元列表
2. 对每个单元执行与 `encode` 一致的合并过程，得到该单元的 token 字符串序列
3. 将所有单元的 token 序列按顺序拼接成一个总 token 列表
4. 返回 token 列表（通常不在这里添加 bos/eos；是否添加要在报告中给出一致约定）

### 2.9 `BPETokenizer.save` / `BPETokenizer.load`（保存 merges 的唯一真相）

目标：保存后加载必须恢复同样的 merges 顺序与同样的词表，从而保证 encode 行为可复现。

自然语言伪代码：

- `save`：
  1. 若目录不存在则创建
  2. 写出 `vocab.json`（token -> id）
  3. 写出 `special_tokens.json`
  4. 写出 `merges.txt`（按顺序逐行保存每条 merge 的左右 token；要求读取后能无歧义还原 pair）
  5. （可选）写出 `tokenizer_config.json` 记录 tokenizer 类型与关键超参
- `load`：
  1. 读入上述文件并校验存在性与字段完整性
  2. 恢复 `vocab` 与 `merges`，并用它们构造分词器实例
  3. 重建 `merges_dict`（pair -> 顺序索引），并验证其与 `merges` 一致
  4. （建议）做一次最小自检：随机取几条 token/id 做正反向映射检查，确保 `vocab` 与 `inverse_vocab` 一致

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

### 3.5 `ByteLevelTokenizer.__init__`（初始化字节映射，保证全局一致）

目标：确保任意实例都具备相同的“字节值(0..255) <-> Unicode 单字符”的双射映射，并且这个映射在 encode/decode 中被一致使用。

自然语言伪代码：

1. 先完成父类初始化（让 vocab/merges/special_tokens 等协议就位）
2. 若类变量 `BYTES_TO_UNICODE` 尚未创建：
   - 调用 `_create_bytes_to_unicode` 创建映射
   - 将其缓存到类变量，避免每个实例重复构造（保证一致性与性能）
3. 在实例上准备反向映射（Unicode 字符 -> byte 值），供 `_unicode_to_bytes` 使用

边界提醒：

- 这个映射必须是“全局固定”的：同一个字节值在不同实例里映射到不同字符会导致保存/加载后 decode 不可逆。

### 3.6 `ByteLevelTokenizer._create_bytes_to_unicode`（构造 256 个字节的可逆映射）

目标：构造一个大小恰好为 256 的映射：每个字节值映射到一个“单字符”Unicode 字符；所有输出字符互不相同，且可构造反向映射实现严格可逆。

自然语言伪代码（GPT-2 风格的思路，描述到可实现粒度）：

1. 先选出一批“天然可显示/不容易出问题”的字节值集合，把它们直接映射到同码点字符：
   - 典型做法是包含大部分可打印 ASCII（不含控制字符）以及一段较安全的扩展拉丁字符区间
2. 对剩余的字节值（未被包含的那些）：
   - 按字节值从小到大遍历
   - 为每个字节依次分配一个“新的 Unicode 码点”，从某个起始点开始递增
   - 关键要求是：这些新分配的字符不会与步骤 1 的字符冲突
3. 最终验证三件事：
   - 映射表大小为 256
   - 每个 value 都是长度为 1 的字符串
   - values 去重后仍为 256（保证双射）

### 3.7 `ByteLevelTokenizer._bytes_to_unicode`（UTF-8 字节序列 -> “可打印字符序列”）

目标：把任意 Unicode 文本转换为“字节级表示字符串”，使后续 BPE 可以只在字符串层面运算，但仍保持信息不丢失。

自然语言伪代码：

1. 将输入文本用 UTF-8 编码成字节序列
2. 对每个字节值：
   - 查 `BYTES_TO_UNICODE` 映射，得到一个单字符
3. 将这些字符按原顺序拼接成新字符串并返回

### 3.8 `ByteLevelTokenizer._unicode_to_bytes`（“可打印字符序列” -> UTF-8 字节序列 -> 原文本）

目标：把字节级表示字符串严格还原为原始文本，确保 decode(encode(text)) 可逆（至少在不引入额外后处理破坏的前提下）。

自然语言伪代码：

1. 准备反向映射 `UNICODE_TO_BYTES`（由 `BYTES_TO_UNICODE` 反转得到）
2. 遍历输入字符串的每个字符：
   - 查反向映射得到对应字节值
   - 依序收集为字节序列
3. 将字节序列按 UTF-8 解码回文本：
   - 如果你选择“替换策略”处理异常字节序列，要在报告里说明；但在本实验中，若输入来自 `_bytes_to_unicode`，正常情况下不应出现无法解码的情况
4. 返回解码后的文本

### 3.9 `ByteLevelTokenizer.train`（在字节表示上训练 merges）

目标：把语料先映射到字节级表示，再在该表示上运行 BPE 的 merges 学习逻辑；初始基础符号应覆盖 256 个字节字符。

自然语言伪代码：

1. 对训练语料逐条执行 `_bytes_to_unicode`，得到“字节级语料”
2. 构造/确认初始基础词表包含 256 个基础符号（每个符号对应一个字节映射字符）
3. 在字节级语料上执行 BPE 的 `train` 流程，得到 merges 与扩展后的 vocab
4. 训练结束后，确保 `merges_dict` 与 `inverse_vocab` 等结构已与最终结果一致

### 3.10 `ByteLevelTokenizer.encode` / `decode`（前后各加一层可逆变换）

目标：在“字节级表示字符串”上进行 BPE 合并与词表映射，但对用户暴露的接口仍是原始文本。

自然语言伪代码：

- `encode`：
  1. 原文本 `text` -> `_bytes_to_unicode` 得到 `byte_text`
  2. 对 `byte_text` 执行 BPE 的 tokenize/合并逻辑，得到 token 字符串序列
  3. 将 token 字符串序列映射到 ID（OOV 回退到 unk）
  4. 按项目约定可选地添加 bos/eos，并做 truncation
- `decode`：
  1. ID 序列 -> token 字符串序列（可选跳过特殊 token）
  2. 将 token 字符串直接拼接成 `byte_text`（注意：这里的拼接是在“字节级表示空间”）
  3. `byte_text` -> `_unicode_to_bytes` 还原得到原文本
  4. 返回文本

与测试契约对齐的提醒：

- 当 merges 为空且 vocab 覆盖 256 基础符号时，encode/decode 应对任意 Unicode 文本保持可逆（测试包含中文与 emoji）。

### 3.11 `ByteLevelTokenizer.save` / `ByteLevelTokenizer.load`

目标：保存/加载后，encode/decode 的行为与可逆性保持不变。

自然语言伪代码：

1. 保存的核心内容与 BPE 类似（vocab、merges、special_tokens、必要的 config）
2. 不需要把 `BYTES_TO_UNICODE` 直接落盘（因为它可由 `_create_bytes_to_unicode` 决定性重建），但你必须保证重建算法与版本一致
3. load 时在完成父类恢复后，重新初始化字节映射与反向映射，确保 `_bytes_to_unicode/_unicode_to_bytes` 可用

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

自然语言伪代码（把“查表”说清楚）：

1. 输入 `input_ids`，形状通常为 `[batch, seq_len]`，并且应为整数 ID（不是 one-hot）
2. 对 `input_ids` 中的每个位置：
   - 将该位置的 ID 视为“选取 embedding_table 的第 ID 行”的指令
   - 得到一个长度为 `hidden_size` 的向量
3. 将所有位置的向量按原 batch/seq 结构堆叠回去，得到输出张量：
   - 形状为 `[batch, seq_len, hidden_size]`
4. 返回该张量

边界提醒：

- 如果 `input_ids` 出现越界 ID，你可以选择让框架报错（更早暴露数据/分词器错误），也可以选择回退到 unk（更鲁棒）。在本项目中更常见的是“越界即错误”，因为 tokenizer 应该保证 ID 合法；但你需要在报告里说明你的选择。

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

#### 5.2.3 `PositionalEncoding.__init__`（预计算并注册为 buffer）

目标：在初始化阶段把位置表一次性算好，并保证它不会被优化器更新。

自然语言伪代码：

1. 调用 `_create_pe` 生成形状为 `[max_len, d_model]` 的位置表
2. 将该位置表注册为 module 的 buffer（即：随模型一起移动到 device、会被保存到 state_dict，但不参与梯度）
3. （可选）在报告中解释：为什么用 buffer 而不是参数（参数会被训练改变，破坏“固定公式”的性质）

#### 5.2.4 `PositionalEncoding._create_pe`（按公式生成位置表）

目标：严格实现正余弦公式，并保证数值与维度对齐正确。

自然语言伪代码：

1. 创建位置索引向量：包含 `0..max_len-1`，并把它组织成“列向量”以便与频率项做广播乘法
2. 创建频率缩放项向量（只对应偶数维）：
   - 频率随维度指数衰减：低维变化慢、高维变化快
   - 该向量长度应为 `d_model/2`
3. 生成位置表矩阵：
   - 初始化全零矩阵，形状 `[max_len, d_model]`
   - 偶数维填入 `sin(位置 * 频率项)`
   - 奇数维填入 `cos(位置 * 频率项)`
4. 返回该矩阵

边界提醒：

- `d_model` 通常要求为偶数；如果不是偶数，你需要在实现中明确你的处理策略（报错或允许最后一维缺失某个 sin/cos 对）。

#### 5.2.5 `PositionalEncoding.forward`（按 seq_len 截取并广播相加）

目标：不改变输入 shape，只做加法注入。

自然语言伪代码：

1. 读取输入 `x` 的序列长度 `seq_len`（来自 `x` 的第二维）
2. 从位置表中取出前 `seq_len` 行，得到形状 `[seq_len, d_model]` 的片段
3. 将该片段广播到 batch 维，与 `x` 做逐元素相加
4. 返回相加后的张量

工程提醒：

- 若输入 `x` 的 dtype 不是 float32（例如 fp16/bf16），你需要明确位置表与输入相加时的 dtype 策略（例如在相加前把位置表转换到与 `x` 相同 dtype），否则可能出现隐式类型提升或精度差异。

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

#### 5.3.4 `RoPE.__init__`（构造 inv_freq，并约束 dim）

目标：准备好每个“二维维度对”的频率倒数向量 `inv_freq`，用于把位置 ID 映射到旋转角度。

自然语言伪代码：

1. 检查 `dim` 是否为偶数（因为 RoPE 以“每两维为一组”进行旋转）；若不是，应明确报错或给出一致约定
2. 构造维度对索引 `i = 0..(dim/2 - 1)`
3. 对每个 `i` 计算该维度对的频率倒数：
   - 频率随 `i` 增大而变化（指数形式），`base` 控制整体频率尺度
4. 将得到的 `inv_freq` 存为 module 的 buffer（不参与梯度），以便跟随模型迁移 device

#### 5.3.5 `RoPE._compute_cos_sin`（位置 -> 角度 -> cos/sin，形状要可广播）

目标：根据 `position_ids` 与 `inv_freq` 计算出可广播到 Q/K 的 `cos` 与 `sin` 张量。

自然语言伪代码：

1. 统一 `position_ids` 的形状：
   - 若输入是一维 `[seq_len]`：视为所有 batch 共用同一位置序列，并在前面补一个 batch 维
   - 若输入是二维 `[batch, seq_len]`：直接使用
2. 计算角度矩阵 `angles`：
   - 对每个位置 `pos` 与每个维度对频率 `inv_freq` 做“外积式组合”
   - 结果形状应为 `[batch, seq_len, dim/2]`
3. 将角度扩展到 `dim`：
   - 因为每个维度对对应两维，你需要把 `[dim/2]` 的角度扩展/复制成 `[dim]` 的角度排列，使最后一维与 Q/K 的 `head_dim` 对齐
4. 计算 `cos` 与 `sin`：
   - 对扩展后的角度逐元素计算余弦与正弦
   - 输出形状均为 `[batch, seq_len, dim]`
5. 数值稳定性策略（建议）：
   - 在低精度输入（fp16/bf16）下，优先在更高精度（例如 float32）里计算角度与 sin/cos，再按需要转换回输入 dtype

#### 5.3.6 `RoPE.rotate_half`（构造“90 度旋转”的辅助向量）

目标：把最后一维拆成两半 `(x1, x2)`，返回 `(-x2, x1)`，用于实现二维旋转公式。

自然语言伪代码：

1. 将输入张量 `x` 在最后一维均分成前半 `x1` 与后半 `x2`（每半大小为 `dim/2`）
2. 构造新张量：
   - 前半取 `-x2`
   - 后半取 `x1`
3. 沿最后一维拼接回去，返回形状与 `x` 相同的张量

#### 5.3.7 `RoPE.apply_rotary_pos_emb`（把 cos/sin 广播到 heads 维并应用旋转）

目标：把旋转位置编码应用到 `q` 与 `k`，并保持输入输出形状一致。

自然语言伪代码：

1. 明确输入形状：
   - `q`: `[batch, num_heads, seq_len, head_dim]`
   - `k`: `[batch, num_kv_heads, seq_len, head_dim]`
   - `cos/sin`: `[batch, seq_len, head_dim]`
2. 让 `cos/sin` 可广播到 heads 维：
   - 在 `cos/sin` 的第二维插入一个大小为 1 的维度，使其变成 `[batch, 1, seq_len, head_dim]`
3. 应用旋转公式（逐元素）：
   - `q_rot` = `q` 与 `cos` 的逐元素乘积，加上 `rotate_half(q)` 与 `sin` 的逐元素乘积
   - `k_rot` 同理
4. 返回 `(q_rot, k_rot)`，形状分别与输入 `q/k` 相同

验证建议（写进报告也很好）：

- 检查范数守恒：旋转前后 `q` 的最后一维范数应几乎不变（允许小数值误差）。

#### 5.3.8 `RoPE.forward`（把 position_ids 串起来用）

目标：在 forward 中完成“计算 cos/sin -> 应用到 q/k”的完整路径。

自然语言伪代码：

1. 从 `q` 读取 `seq_len`、`head_dim` 与 device/dtype 信息
2. 调用 `_compute_cos_sin` 得到 `cos/sin`
3. 调用 `apply_rotary_pos_emb` 得到旋转后的 `q/k`
4. 返回旋转后的 `(q, k)`

与测试契约相关的提醒：

- 如果把 `position_ids` 整体加上一个常数偏移（所有位置一起平移），RoPE 的注意力分数应保持不变（体现其相对位置性质）。这是你实现正确性的一个强信号。

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
