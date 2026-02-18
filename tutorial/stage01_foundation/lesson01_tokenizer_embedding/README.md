# L01: Tokenizer 与 Embedding

## 学习目标

1. **理解** BPE (Byte Pair Encoding) 分词算法的原理
2. **掌握** 字节级分词 (Byte-level Tokenization) 的优势
3. **理解** 位置编码的作用：为什么 Transformer 需要位置信息
4. **掌握** RoPE (旋转位置编码) 的数学原理和实现
5. **能够** 实现 TokenEmbedding 和 RoPE 模块

---

## 理论背景

### 1. BPE 分词算法

**核心思想**：从字符级开始，迭代合并频率最高的字符对，构建子词词表。

**算法步骤**：
```
初始: 词表 = 所有字符 (如 'a', 'b', 'c', ...)
迭代:
    1. 统计所有相邻字符对频率
    2. 找到频率最高的字符对 (如 "e" + "r" = "er")
    3. 将 "er" 加入词表
    4. 语料中所有 "e r" 替换为 "er"
```

**BPE 的优势**：
- **处理 OOV**：未登录词可以通过子词表示
- **词表可控**：通过 vocab_size 参数控制词表大小
- **平衡粒度**：避免过细（字符级序列太长）或过粗（词级 OOV 多）

### 2. 字节级分词

**核心思想**：直接对原始字节进行分词，而非 Unicode 字符。

**优势**：
- **语言无关**：任何语言的文本都可以用 256 个字节表示
- **无需字符集限制**：避免 OOV 问题
- **更稳定**：处理罕见字符、表情符号等更鲁棒

### 3. 位置编码

**为什么需要位置编码？**
- Self-Attention 是置换等变的：不考虑 token 的顺序
- 位置编码为模型提供序列顺序信息

**两种主要方案**：

#### Sinusoidal 位置编码 (原始 Transformer)
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

特点：固定编码，可外推到更长序列

#### RoPE (旋转位置编码)

**核心思想**：通过旋转矩阵将位置信息注入到 Q、K 向量中，使得注意力结果只与相对位置有关。

**数学原理**：

对于二维向量 (x₁, x₂)，位置 m 的旋转编码：
```
[m] = [cos(mθ)  -sin(mθ)] [x₁]
      [sin(mθ)   cos(mθ)] [x₂]
```

**高维推广**：每两个维度组成一对进行旋转

**优势**：
1. **相对位置编码**：`<f(q,m), f(k,n)>` 只依赖于 (m-n)
2. **长度外推性好**：适合更长上下文
3. **现代 LLM 标准**：LLaMA、Qwen、Mistral 等都采用 RoPE

---

## 代码实现

### 项目结构

```
tokenizer/
├── base_tokenizer.py      # 分词器基类
├── bpe_tokenizer.py       # BPE 分词器
└── byte_level_tokenizer.py # 字节级分词器

model/
├── embedding.py           # 嵌入层和 RoPE
└── config.py              # 模型配置
```

---

## 实践练习

### 练习 1: 实现 TokenEmbedding

打开 `model/embedding.py`，完成 `TokenEmbedding` 类的实现：

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        """
        初始化 Token 嵌入层，将 token ID 映射到隐藏维度空间。

        Args:
            vocab_size: 词表大小
            hidden_size: 嵌入维度
        """
        super().__init__()
        # 实现: 创建可学习的嵌入查找表
        pass

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        将 token ID 转换为嵌入向量。

        Args:
            input_ids: token ID，形状 [batch_size, seq_len]

        Returns:
            嵌入向量，形状 [batch_size, seq_len, hidden_size]
        """
        # 实现: 通过嵌入表将 input_ids 映射为嵌入向量
        pass
```

### 练习 2: 实现 RoPE

完成 `RoPE` 类的核心方法：

```python
class RoPE(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        初始化旋转位置编码模块。

        Args:
            dim: 注意力头维度
            max_position_embeddings: 最大位置数
            base: 基础频率
        """
        super().__init__()
        # 实现: 预计算旋转角度频率 inv_freq，形状为 [dim // 2]
        # 公式: inv_freq = base^(-2i/dim)，其中 i = 0, 1, ..., dim/2-1
        pass

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        实现旋转矩阵的后半部分取负操作。

        输入: [..., dim]
        输出: [..., dim]
        """
        # 实现: 将张量分为前后两半，前半部分与后半部分取负后交换位置
        pass

    def forward(self, q, k, position_ids):
        """
        应用旋转位置编码到查询和键向量。

        Args:
            q: 查询向量 [..., seq_len, dim]
            k: 键向量 [..., seq_len, dim]
            position_ids: 位置索引

        实现思路:
        1. 根据 position_ids 计算旋转角度
        2. 将角度转换为 cos 和 sin 形式
        3. 使用旋转矩阵对 Q 和 K 进行变换
        """
        # 实现: 完成 RoPE 编码的前向传播
        pass
```

### 练习 3: 验证 RoPE 的相对位置特性

编写代码验证 RoPE 的相对位置编码特性：

```python
import torch
from model.embedding import RoPE

# 创建 RoPE 模块
rope = RoPE(dim=64)

# 测试: 验证 <f(q,m), f(k,n)> 只依赖于 (m-n)
# ... 你的验证代码
```

---

## 测试验证

运行测试用例验证实现正确性：

```bash
# 基本功能测试
pytest tutorial/stage01_foundation/lesson01_tokenizer_embedding/testcases/basic_test.py -v

# 进阶测试 (可选)
pytest tutorial/stage01_foundation/lesson01_tokenizer_embedding/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **BPE 原始论文**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015)
- **RoPE 论文**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- **字节级分词**: 参考 GPT-2/GPT-3 的 byte-level BPE 实现
