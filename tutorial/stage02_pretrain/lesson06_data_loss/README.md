# L06: 数据处理与损失函数

## 学习目标

1. **理解** 数据过滤和预处理方法
2. **掌握** Cross Entropy 损失函数的数学原理
3. **能够** 实现数据管道和损失计算

---

## 理论背景

### 1. 数据预处理

训练大规模语言模型需要高质量的数据预处理。预处理流程通常包括以下几个关键步骤:

#### 1.1 重复数据删除 (Deduplication)

**目的**: 去除重复或近似重复的文本，防止模型过拟合到重复模式。

**方法**:
- **精确去重**: 使用哈希或精确匹配删除完全相同的文档
- **近似去重**: 使用 MinHash、SimHash 等算法检测近似重复内容
- **子序列去重**: 检测长文档中的重复段落

**经验法则**: 当重复率超过一定阈值时，模型会出现明显的重复生成问题。

#### 1.2 低质量数据过滤

**目的**: 移除噪声数据，提升训练数据质量。

**常用方法**:
- **语言识别**: 过滤非目标语言文本
- **长度过滤**: 移除过短或过长的文档
- **统计特征过滤**:
  - 特殊字符比例
  - 重复字符/词比例
  - perplexity 异常值
- **规则过滤**: 过滤垃圾文本、敏感内容等
- **质量分类器**: 训练文本质量分类模型

#### 1.3 长度截断 (Truncation)

**目的**: 处理过长序列，避免显存爆炸。

**方法**:
- **固定长度截断**: 将所有序列截断到固定长度
- **滑动窗口**: 使用滑动窗口生成多个训练样本
- **分层截断**: 对不同层级的数据使用不同长度阈值

### 2. Cross Entropy 损失函数

#### 2.1 信息论基础

**信息量**: 事件 x 发生的信息量为 $I(x) = -\log P(x)$

**熵 (Entropy)**: 概率分布的不确定性度量
$$H(P) = -\sum_x P(x) \log P(x)$$

**交叉熵 (Cross Entropy)**: 真实分布 P 与预测分布 Q 的差异
$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

#### 2.2 语言模型中的 Cross Entropy

给定上下文 $x_{1:t-1}$，语言模型预测下一个 token $x_t$ 的概率为 $P(x_t | x_{1:t-1})$。

**训练目标**: 最大化对数似然，等价于最小化交叉熵损失。

**序列损失**:
$$\mathcal{L}_{LM} = -\frac{1}{T} \sum_{t=1}^{T} \log P_{\theta}(x_t | x_{1:t-1})$$

**逐 token 损失**:
对于位置 t，假设真实 token 为 $y_t$，模型预测的 logit 为 $z_t$，则:
$$L_t = -\log \text{softmax}(z_t)[y_t] = -\left(z_t[y_t] - \log \sum_j \exp(z_t[j])\right)$$

#### 2.3 交叉熵与 Perplexity 的关系

Perplexity 是评估语言模型的常用指标，与交叉熵的关系:
$$\text{PPL} = \exp\left(\frac{1}{T} \sum_t L_t\right) = \exp(H(P, Q))$$

---

## 代码实现

### 项目结构

```
data/
├── filtering.py      # 数据过滤
├── truncation.py     # 长度截断
└── dataset.py        # 数据集实现

loss/
└── cross_entropy.py  # 损失函数实现
```

---

## 实践练习

### 练习 1: 实现数据过滤器类

打开 `data/filtering.py`，实现以下过滤器类:

```python
class LengthFilter(DataFilter):
    def __init__(self, min_length: int = 10, max_length: int = 10000):
        """
        长度过滤器。

        Args:
            min_length: 最小长度阈值
            max_length: 最大长度阈值
        """
        super().__init__(min_length, max_length)

    def filter(self, text: str) -> bool:
        """
        根据长度过滤。

        实现:
        1. 计算文本长度
        2. 检查是否在 [min_length, max_length] 范围内
        """
        pass


class QualityFilter(DataFilter):
    def __init__(self, max_special_char_ratio: float = 0.5):
        """
        质量过滤器。

        Args:
            max_special_char_ratio: 最大特殊字符比例
        """
        pass

    def filter(self, text: str) -> bool:
        """
        过滤低质量文本。

        实现:
        1. 检查特殊字符比例
        2. 检查重复率
        3. 综合判断
        """
        pass
```

### 练习 2: 实现 CrossEntropyLoss 类

打开 `loss/cross_entropy.py`，实现语言模型损失计算:

```python
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        交叉熵损失。

        Args:
            ignore_index: 忽略的索引 (用于 padding)
            label_smoothing: 标签平滑系数
        """
        pass

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失。

        公式: L = -log P(target | context)

        Args:
            logits: 模型输出的 logits [batch, seq_len, vocab_size]
            labels: 目标 token IDs [batch, seq_len]

        Returns:
            标量损失值
        """
        pass
```

### 练习 3: 使用数据处理管道

```python
# 使用过滤器
from data.filtering import LengthFilter, QualityFilter, Pipeline

length_filter = LengthFilter(min_length=50, max_length=5000)
quality_filter = QualityFilter(max_special_char_ratio=0.3)

pipeline = Pipeline()
pipeline.add_filter(length_filter)
pipeline.add_filter(quality_filter)

# 处理数据
texts = ["...", "..."]
filtered_texts = pipeline.process(texts)
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage02_pretrain/lesson06_data_loss/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage02_pretrain/lesson06_data_loss/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **数据处理**: 了解 LLaMA、Common Crawl 等数据集的处理流程
- **质量过滤**: 探索可用于文本质量评估的多种方法
- **损失函数**: 深入理解 softmax 和交叉熵的数值稳定性问题
