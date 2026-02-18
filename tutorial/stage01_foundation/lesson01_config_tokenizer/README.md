# 课时1：项目概述、配置系统与分词器

## 学习目标

1. 理解Transformer Decoder-only架构的整体设计
2. 掌握ModelConfig配置类的设计与使用
3. 理解BPE分词算法的核心原理
4. 实现BPE训练和编码

---

## 1. Transformer Decoder-only 架构概览

### 1.1 架构演进

```
原始Transformer (2017)
    ├── Encoder: 双向注意力，适合理解任务
    └── Decoder: 因果注意力，适合生成任务

GPT系列 (Decoder-only)
    └── 仅保留Decoder，堆叠多层Transformer Block
        ├── GPT-1 (2018): 预训练 + 微调范式
        ├── GPT-2 (2019): Zero-shot能力
        ├── GPT-3 (2020): In-context learning
        └── LLaMA/Qwen (2023+): 开源高效架构
```

### 1.2 训练流程 Pipeline

```
Pretrain (无监督预训练)
    ├── 数据: 海量无标注文本
    ├── 任务: 下一词预测 (Next Token Prediction)
    └── 输出: Base模型 (如 Qwen2.5-7B-Base)
            │
            ▼
SFT (监督微调)
    ├── 数据: 指令-回答对 (如Alpaca格式)
    ├── 任务: 学习遵循指令
    └── 输出: Instruct模型 (如 Qwen2.5-7B-Instruct)
            │
            ▼
Alignment (对齐训练)
    ├── DPO: 直接偏好优化
    ├── GRPO: 组相对策略优化
    └── PPO: 近端策略优化 (传统RLHF)
```

---

## 2. ModelConfig 配置系统

### 2.1 配置参数详解

```python
@dataclass
class ModelConfig:
    # 词汇表与维度
    vocab_size: int = 32000        # 词表大小
    hidden_size: int = 4096        # 隐藏层维度 (d_model)
    num_hidden_layers: int = 32    # Transformer层数

    # 注意力配置
    num_attention_heads: int = 32  # 注意力头数
    num_key_value_heads: int = 32  # KV头数 (GQA: 设为小于num_attention_heads)
    max_position_embeddings: int = 4096  # 最大序列长度

    # FFN配置
    intermediate_size: int = 11008 # FFN中间层维度
    hidden_act: str = "silu"       # 激活函数: silu/gelu/relu

    # 归一化配置
    rms_norm_eps: float = 1e-6     # RMSNorm epsilon
    use_rms_norm: bool = True      # True=RMSNorm, False=LayerNorm

    # 位置编码
    rope_theta: float = 10000.0    # RoPE基频
    use_rope: bool = True          # True=RoPE, False=Sinusoidal

    # 正则化
    dropout_rate: float = 0.0      # Dropout概率
    attention_dropout: float = 0.0 # 注意力Dropout

    # 杂项
    pad_token_id: int = 0          # 填充token ID
    bos_token_id: int = 1          # 起始token ID
    eos_token_id: int = 2          # 结束token ID
    tie_word_embeddings: bool = False  # 共享输入输出embedding
```

### 2.2 GQA (Grouped Query Attention) 配置

```
标准MHA: num_attention_heads = 32, num_key_value_heads = 32
    └── 每个Query头对应独立的K、V头

GQA: num_attention_heads = 32, num_key_value_heads = 8
    └── 每4个Query头共享1组K、V头 (32/8=4)
    └── KV缓存减少为原来的 1/4，推理更快

MQA: num_attention_heads = 32, num_key_value_heads = 1
    └── 所有Query头共享1组K、V头
    └── 极致压缩，但可能损失质量
```

### 2.3 不同规模模型配置

| 模型 | 参数量 | hidden_size | num_layers | num_heads | intermediate_size |
|------|--------|-------------|------------|-----------|-------------------|
| Tiny | 10M | 128 | 4 | 4 | 512 |
| Small | 100M | 512 | 8 | 8 | 2048 |
| Base | 1B | 2048 | 16 | 16 | 8192 |
| Large | 7B | 4096 | 32 | 32 | 11008 |
| XLarge | 70B | 8192 | 80 | 64 | 28672 |

---

## 3. BPE (Byte Pair Encoding) 分词算法

### 3.1 子词分词原理

**为什么需要子词分词？**

```
问题1: 词级分词
    词表: ["猫", "狗", "猫咪", "小狗", ...]
    问题: 词汇量爆炸，无法处理新词，"猫咪"和"猫"语义关联丢失

问题2: 字符级分词
    词表: ["a", "b", "c", ...]
    问题: 序列过长，语义粒度太细

子词分词 (BPE) 的解决方案:
    词表: ["猫", "咪", "狗", "小", ...]
    优势:
    - "猫" + "咪" → "猫咪" (可组合)
    - 词表大小可控 (通常32k-100k)
    - 能处理未登录词 (OOV)
```

### 3.2 BPE 训练算法

**核心思想**: 从字符开始，迭代合并频率最高的字符对

```
初始语料:
    low: 5次, lower: 2次, newest: 6次, widest: 3次

Step 1: 初始化为字符序列
    l o w </w>: 5
    l o w e r </w>: 2
    n e w e s t </w>: 6
    w i d e s t </w>: 3

Step 2: 统计所有相邻字符对频率
    (l, o): 7    (o, w): 7    (w, </w>): 5
    (w, e): 2    (e, r): 2    (n, e): 6
    (e, w): 6    (w, e): 8    (e, s): 9
    (s, t): 9    ...

Step 3: 合并频率最高的对 (e, s) → "es"
    更新语料...

Step 4: 重复合并，直到达到目标词表大小
```

### 3.3 BPE 编码算法

**贪心最长匹配**:
```
输入: "lowest"
词表: ["low", "est", "lo", "w", "e", "s", "t", ...]

编码过程:
    1. 尝试最长匹配: "low" ✓ 匹配成功，剩余 "est"
    2. 尝试最长匹配: "est" ✓ 匹配成功，剩余 ""
    3. 编码完成: ["low", "est"]

Token IDs: [low_id, est_id]
```

---

## 4. 实现指引

### 4.1 model/config.py

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """
    LLM模型配置类

    属性说明:
        vocab_size: 词汇表大小，决定embedding矩阵的第一维
        hidden_size: 隐藏层维度，影响模型容量和计算量
        num_hidden_layers: Transformer层数，深层网络学习能力更强
        num_attention_heads: 注意力头数，必须整除hidden_size
        num_key_value_heads: GQA的KV头数，<= num_attention_heads
        max_position_embeddings: 最大序列长度，影响位置编码和缓存
        intermediate_size: FFN中间层维度，通常为2-4倍hidden_size
    """

    # Step 1: 定义所有配置字段及其默认值
    # 参考上面的参数详解表格

    # Step 2: 实现__post_init__进行参数验证
    # 验证1: num_attention_heads必须整除hidden_size
    #    使用取模运算验证整除关系，若余数不为0则抛出异常
    # 验证2: num_key_value_heads必须整除num_attention_heads
    #    使用取模运算验证整除关系
    # 验证3: 计算head_dim
    #    通过 hidden_size 除以 num_attention_heads 得到每个头的维度

    # Step 3: 提供辅助属性
    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        pass

    def to_dict(self) -> dict:
        """转换为字典，便于序列化"""
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """从字典创建配置"""
        pass
```

### 4.2 tokenizer/bpe_tokenizer.py

```python
class BPETokenizer:
    """
    BPE分词器实现

    核心数据结构:
        vocab: Dict[int, str] - id到token的映射
        merges: List[Tuple[str, str]] - 合并规则列表(按优先级排序)
        special_tokens: 特殊token处理
    """

    def __init__(self, vocab_size: int = 32000):
        # Step 1: 初始化基础字符词表
        # 通常包含: 字节级字符(0-255)、特殊token(<pad>, <s>, </s>等)
        pass

    def train(self, texts: List[str], num_merges: int):
        """
        在语料上训练BPE

        Args:
            texts: 训练文本列表
            num_merges: 执行的合并次数

        算法步骤:
        Step 1: 预处理文本，添加</w>标记词尾
        Step 2: 将文本拆分为字符序列
        Step 3: 统计所有词及其频率
        Step 4: 迭代num_merges次:
            4.1 统计所有相邻字符对及其频率
            4.2 找到频率最高的字符对(bigram)
            4.3 在所有词中合并该字符对
            4.4 将合并规则加入merges列表
            4.5 更新词表，加入新token
        """
        pass

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token IDs

        算法步骤:
        Step 1: 预处理文本(统一编码、小写化等)
        Step 2: 将文本拆分为字符序列
        Step 3: 迭代应用合并规则:
            3.1 找出当前序列中所有可合并的相邻对
            3.2 按合并规则优先级选择最先定义的合并
            3.3 执行合并操作，将相邻字符对组合成新的token
        Step 4: 将最终token序列转换为IDs

        Returns:
            token_ids: List[int] - token ID列表
        """
        pass

    def decode(self, token_ids: List[int]) -> str:
        """
        将token IDs解码为文本

        算法步骤:
        Step 1: 将每个ID转换为对应的token字符串
        Step 2: 拼接所有token
        Step 3: 移除</w>等标记，恢复原始格式
        """
        pass

    def save(self, path: str):
        """保存词表和合并规则到文件"""
        pass

    def load(self, path: str):
        """从文件加载词表和合并规则"""
        pass
```

### 4.3 tokenizer/byte_level_tokenizer.py

```python
class ByteLevelTokenizer:
    """
    字节级BPE分词器 (类似GPT-2使用)

    特点:
    1. 基础词表直接对应256个字节值
    2. 无需预处理处理Unicode，任何文本都可以编码
    3. 不会遇到OOV问题
    """

    def __init__(self, vocab_size: int = 50000):
        # Step 1: 初始化字节级基础词表
        # 词表0-255直接映射到字节值0-255
        pass

    def train(self, texts: List[str], num_merges: int):
        """
        训练过程与BPE类似，但操作的是字节序列

        Step 1: 将文本转换为UTF-8字节序列
        Step 2: 在字节序列上执行BPE合并算法
        """
        pass

    def encode(self, text: str) -> List[int]:
        """
        Step 1: 将文本编码为UTF-8字节序列
        Step 2: 应用BPE合并规则
        Step 3: 返回token IDs
        """
        pass

    def decode(self, token_ids: List[int]) -> str:
        """
        Step 1: 将token IDs转换为字节序列
        Step 2: 使用UTF-8解码为文本
        """
        pass
```

---

## 5. 关键公式总结

### GQA 头数计算

```
num_query_groups = num_attention_heads / num_key_value_heads
kv_cache_size_ratio = num_key_value_heads / num_attention_heads
```

### 模型参数量估算

```
总参数量 ≈ V * H + L * (
    # Attention
    4 * H^2 +  # Q, K, V, O projections
    # FFN
    2 * H * I  # gate_proj/up_proj + down_proj
)

其中:
    V = vocab_size
    H = hidden_size
    I = intermediate_size
    L = num_hidden_layers
```

---

## 6. 常见陷阱与注意事项

1. **注意力头数必须整除hidden_size**: 否则无法均匀划分头维度
2. **GQA的KV头数必须整除Q头数**: 保证每组Query能均匀共享KV
3. **BPE合并顺序影响编码结果**: 优先应用更早学习的合并规则
4. **特殊token处理**: 确保pad/eos/bos token在词表中有明确定义
5. **词表大小与显存关系**: Embedding层占显存 = vocab_size * hidden_size * 4 bytes

---

## 7. 课后练习

1. 计算一个7B模型的配置：给定hidden_size=4096, num_layers=32，反推合适的intermediate_size
2. 手动执行BPE训练：给定语料 ["aaab", "aaba", "abaa", "baaa"]，执行3次合并
3. 思考：为什么GQA可以减少KV缓存但MHA不行？
4. 对比：ByteLevel BPE与普通BPE的优缺点
