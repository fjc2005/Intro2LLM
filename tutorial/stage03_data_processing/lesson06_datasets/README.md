# 课时6：数据集与数据清洗

## 学习目标

1. 理解预训练数据集的构建与采样
2. 掌握SFT数据集的instruction掩码处理
3. 理解DPO偏好数据集的格式
4. 掌握数据清洗的关键技术：过滤、去重、质量评估

---

## 1. 预训练数据集 (PretrainDataset)

### 1.1 预训练的任务定义

```
任务: 因果语言建模 (Causal Language Modeling)
      也称为下一词预测 (Next Token Prediction)

给定:   "今天 天气 很 好"
预测:   "今天" → "天气"
        "今天 天气" → "很"
        "今天 天气 很" → "好"

数学形式:
    L = -Σ log P(x_t | x_{<t})
```

### 1.2 数据采样策略

```
问题: 文档长度差异大，如何高效批处理?

策略1: 固定长度采样 (Fixed Length)
    - 将长文档切分为固定长度片段
    - 短文档用特殊token拼接或填充
    - 简单但可能切断语义边界

策略2: 动态填充 (Dynamic Padding)
    - 按实际长度批处理
    - 同批次内用padding对齐
    - 需要attention mask处理padding

策略3: Packing (推荐)
    - 将多个短文档打包到一个序列
    - 用特殊token分隔
    - 最大化GPU利用率
```

### 1.3 注意力掩码处理

```python
# 情况1: 普通因果掩码 (无padding)
attention_mask = None  # 或使用因果掩码

# 情况2: 有padding的批次
# input_ids:    [1, 2, 3, 4, 0, 0]  (0是pad)
# attention_mask: [1, 1, 1, 1, 0, 0]
# 在注意力计算中，padding位置设为-∞

# 情况3: 打包多个文档
# input_ids:    [<s>, A, B, </s>, <s>, C, D, E, </s>]
# 需要防止文档间注意力 (文档B不应attend到文档A)
# 使用segment_ids或特殊attention mask
```

---

## 2. SFT数据集 (SFTDataset)

### 2.1 Instruction Tuning格式

```python
# Alpaca格式
{
    "instruction": "将以下中文翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is nice today."
}

# 转换为对话格式
template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# 简化的对话格式 (类似ShareGPT)
{
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "讲个笑话"},
        {"role": "assistant", "content": "好的，这是一个笑话..."}
    ]
}
```

### 2.2 Loss Masking原理

```
关键问题: 只对assistant的回复计算损失，不对用户的prompt计算

原因:
    - Prompt是输入，不是预测目标
    - 学习预测prompt会降低学习效率
    - 模型应该学习如何回复，而不是复述问题

实现方法:
    labels = input_ids.copy()
    labels[prompt_positions] = -100  # ignore_index

示例:
    tokens:   [<s>, 你, 好, 我, 是, AI]
    roles:    [     user     assistant]
    labels:   [-100, -100, -100, 我, 是, AI]
              ↑ prompt部分mask为-100
```

### 2.3 对话模板 (Chat Template)

```
不同模型的对话格式:

LLaMA-2:
    <s>[INST] {user_message} [/INST] {assistant_message} </s>
    <s>[INST] {user_message} [/INST]

Qwen:
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>

ChatGLM:
    [Round 1]
    问：{user_message}
    答：{assistant_message}
```

---

## 3. DPO数据集 (DPODataset)

### 3.1 偏好数据格式

```python
# DPO数据格式
{
    "prompt": "人类需要帮助解决数学问题: 2+2=",
    "chosen": "2+2等于4。这是基础的加法运算。",
    "rejected": "我不知道答案。"
}

# 完整格式 (带对话历史)
{
    "conversations": [
        {"from": "human", "value": "你好"},
        {"from": "gpt", "value": "你好！"},
        {"from": "human", "value": "2+2=?"}
    ],
    "chosen": {"from": "gpt", "value": "等于4"},
    "rejected": {"from": "gpt", "value": "我不知道"}
}
```

### 3.2 DPO数据处理

```python
# 将preference数据转换为模型输入
def process_dpo_example(example):
    # 1. 编码prompt
    prompt_ids = tokenizer.encode(example["prompt"])

    # 2. 编码chosen和rejected
    chosen_ids = tokenizer.encode(example["chosen"])
    rejected_ids = tokenizer.encode(example["rejected"])

    # 3. 构建完整序列
    chosen_full = prompt_ids + chosen_ids
    rejected_full = prompt_ids + rejected_ids

    # 4. 计算prompt长度 (用于mask)
    prompt_len = len(prompt_ids)

    return {
        "chosen_input_ids": chosen_full,
        "rejected_input_ids": rejected_full,
        "prompt_length": prompt_len,
    }
```

---

## 4. 数据清洗与过滤

### 4.1 长度过滤

```python
def length_filter(examples, min_length=10, max_length=100000):
    """过滤过短或过长的文档"""
    return [
        ex for ex in examples
        if min_length <= len(ex["text"]) <= max_length
    ]
```

### 4.2 重复过滤

```python
# 精确重复检测
def exact_deduplication(examples):
    seen = set()
    unique = []
    for ex in examples:
        text_hash = hash(ex["text"])
        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(ex)
    return unique

# 模糊重复检测 (MinHash + LSH)
from datasketch import MinHash, MinHashLSH

def minhash_deduplication(examples, threshold=0.9):
    """
    MinHash算法步骤:
    1. 将文档分词为n-grams
    2. 对每个n-gram计算多个hash值
    3. 取每个hash函数的最小值作为签名
    4. 相似文档的MinHash签名相似
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    for i, ex in enumerate(examples):
        m = MinHash(num_perm=128)
        # 添加n-grams
        for ngram in get_ngrams(ex["text"], n=5):
            m.update(ngram.encode('utf8'))

        # 检查是否有相似文档
        result = lsh.query(m)
        if not result:  # 无相似文档
            lsh.insert(i, m)
            yield ex
```

### 4.3 质量评分

```python
def quality_score(text):
    """多维度质量评分"""
    scores = {}

    # 1. 语言模型困惑度 (Perplexity)
    scores["perplexity"] = compute_ppl(text)

    # 2. 可读性评分
    scores["readability"] = flesch_reading_ease(text)

    # 3. 符号比例
    scores["symbol_ratio"] = count_symbols(text) / len(text)

    # 4. 行长度方差 (检测表格/代码)
    lines = text.split('\n')
    scores["line_variance"] = variance([len(l) for l in lines])

    # 综合评分
    final_score = weighted_average(scores)
    return final_score

def filter_by_quality(examples, min_score=0.5):
    return [ex for ex in examples if quality_score(ex["text"]) >= min_score]
```

### 4.4 敏感内容过滤

```python
def toxic_content_filter(examples):
    """基于规则或模型的有害内容过滤"""
    # 规则匹配
    blocked_patterns = [
        r"(暴力|色情|歧视)",
        # ...
    ]

    # 分类器检测
    toxicity_classifier = load_classifier()

    def is_safe(text):
        # 规则检查
        for pattern in blocked_patterns:
            if re.search(pattern, text):
                return False

        # 模型预测
        toxicity_score = toxicity_classifier.predict(text)
        return toxicity_score < 0.5

    return [ex for ex in examples if is_safe(ex["text"])]
```

---

## 5. 实现指引

### 5.1 data/pretrain_dataset.py

```python
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional

class PretrainDataset(Dataset):
    """
    预训练数据集

    支持:
    - 从文本文件加载
    - 动态长度采样
    - Packing多个文档
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        packing: bool = True,
    ):
        # Step 1: 加载和预处理数据
        # 读取文本文件或jsonl
        # 进行基础清洗

        # Step 2: Tokenize所有文本
        # 保存tokenized结果

        # Step 3: 如果packing=True，将短文档打包
        # 用eos_token分隔，构建长序列

        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回:
            {
                "input_ids": [seq_len],
                "attention_mask": [seq_len],
                "labels": [seq_len],  # 与input_ids相同(因果LM)
            }
        """
        # Step 1: 获取对应样本

        # Step 2: 构建attention_mask (处理padding)

        # Step 3: labels = input_ids (下一个token预测)

        pass
```

### 5.2 data/sft_dataset.py

```python
class SFTDataset(Dataset):
    """
    监督微调数据集

    关键功能: 对prompt部分进行mask，只对response计算loss
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        chat_template: Optional[str] = None,
    ):
        # Step 1: 加载数据 (json/jsonl格式)

        # Step 2: 应用对话模板
        # 将instruction/input/output或messages转换为统一格式

        # Step 3: Tokenize并确定prompt/response边界

        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回:
            {
                "input_ids": [seq_len],
                "attention_mask": [seq_len],
                "labels": [seq_len],  # prompt部分为-100
            }
        """
        # Step 1: 完整序列的token IDs

        # Step 2: 确定prompt长度
        # prompt_len = len(tokenized_prompt)

        # Step 3: 构建labels
        # labels = input_ids.clone()
        # labels[:prompt_len] = -100  # ignore_index

        pass
```

### 5.3 data/dpo_dataset.py

```python
class DPODataset(Dataset):
    """
    DPO偏好数据集

    每个样本包含: prompt, chosen(偏好回答), rejected(非偏好回答)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
    ):
        # Step 1: 加载偏好数据

        # Step 2: 分别编码prompt+chosen和prompt+rejected

        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回:
            {
                "chosen_input_ids": [chosen_len],
                "chosen_attention_mask": [chosen_len],
                "rejected_input_ids": [rejected_len],
                "rejected_attention_mask": [rejected_len],
                "prompt_length": int,
            }
        """
        # 编码chosen和rejected序列
        # 记录prompt长度用于loss masking
        pass
```

### 5.4 data/filtering.py

```python
"""
数据清洗和过滤工具
"""

def length_filter(examples: List[Dict], min_len: int, max_len: int) -> List[Dict]:
    """长度过滤"""
    pass

def exact_deduplication(examples: List[Dict]) -> List[Dict]:
    """精确去重"""
    pass

def minhash_deduplication(
    examples: List[Dict],
    threshold: float = 0.85,
    ngrams: int = 5,
) -> List[Dict]:
    """MinHash模糊去重"""
    pass

def quality_filter(
    examples: List[Dict],
    min_perplexity: float = 20,
    max_perplexity: float = 1000,
) -> List[Dict]:
    """基于困惑度的质量过滤"""
    pass

def toxic_filter(examples: List[Dict], toxicity_threshold: float = 0.5) -> List[Dict]:
    """有害内容过滤"""
    pass
```

---

## 6. 关键概念总结

### 数据集对比

| 特性 | Pretrain | SFT | DPO |
|------|----------|-----|-----|
| 输入格式 | 纯文本 | Instruction + Output | Prompt + Chosen/Rejected |
| Loss计算 | 全序列 | 仅Response | 对比偏好对 |
| 数据量 | 大 (TB级) | 小 (MB-GB级) | 小 (MB级) |
| 数据质量 | 一般 | 高 | 高 |
| 主要目的 | 学习语言 | 学习指令遵循 | 学习人类偏好 |

### 清洗流程

```
原始数据
    ↓
长度过滤 → 去除过短/过长文档
    ↓
去重 → 精确去重 + MinHash模糊去重
    ↓
质量过滤 → 困惑度、可读性、格式检查
    ↓
安全过滤 → 有害内容、PII检测
    ↓
清洗后数据
```

---

## 7. 常见陷阱与注意事项

1. **Packing时的跨文档注意力**: 使用segment_ids或特殊mask防止
2. **Loss Masking边界**: 确保-100不覆盖response的第一个token
3. **Tokenization一致性**: 训练和推理使用相同的tokenizer配置
4. **DPO长度不匹配**: chosen和rejected长度不同时的处理
5. **数据泄露**: 确保训练集和测试集没有重叠
6. **重复惩罚**: 训练数据中的重复会影响模型生成多样性

---

## 8. 课后练习

1. **手动构建样本**: 给出一个instruction，手动写出SFT格式的完整样本
2. **Loss Masking**: 计算一个对话样本的labels，标出哪些位置是-100
3. **MinHash理解**: 解释为什么MinHash能高效估计Jaccard相似度
4. **数据配比**: 设计一个多领域数据的采样策略
5. **质量评分**: 给出一个你认为"高质量"和"低质量"的文本例子
