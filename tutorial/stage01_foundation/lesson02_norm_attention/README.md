# L02: Normalization 与 Attention

## 学习目标

1. **理解** LayerNorm 和 RMSNorm 的区别与联系
2. **掌握** Multi-Head Attention (MHA) 的工作机制
3. **能够** 实现完整的注意力模块

---

## 理论背景

### 1. 归一化层

#### LayerNorm (层归一化)

**论文**: "Layer Normalization" (Ba et al., 2016)

**计算公式**:
```
y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
```

**特点**:
- 减去均值，除以标准差
- 有可学习的缩放 (gamma) 和平移 (beta) 参数
- 对每个样本独立归一化，不依赖 batch

#### RMSNorm (均方根层归一化)

**论文**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

**计算公式**:
```
RMS(x) = sqrt(mean(x^2) + eps)
y = x / RMS(x) * weight
```

**特点**:
- 只使用均方根，不需要计算均值
- 只有缩放参数，没有平移参数
- 计算更简单，更快

**为什么现代 LLM 使用 RMSNorm**:
1. 计算更高效 (无需计算均值)
2. Pre-LN 结构不需要 bias 来重新定位激活分布
3. LLaMA、Qwen、Mistral 等都采用 RMSNorm

### 2. Multi-Head Attention (MHA)

**论文**: "Attention Is All You Need" (Vaswani et al., 2017)

**核心公式**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**多头版本**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)
```

**特点**:
- 每个头有独立的 Q、K、V 投影
- num_heads * head_dim = hidden_size
- 适用于需要精细注意力的场景
- KV 缓存占用较大

---

## 代码实现

### 项目结构

```
model/
├── norm.py           # LayerNorm, RMSNorm
├── attention.py       # MultiHeadAttention, GroupedQueryAttention
└── embedding.py       # RoPE
```

---

## 实践练习

### 练习 1：实现 LayerNorm

打开 `model/norm.py`，完成 `LayerNorm` 类：

```python
class LayerNorm(nn.Module):
    """
    Layer Normalization (层归一化)

    论文: "Layer Normalization" (Ba et al., 2016)
    链接: https://arxiv.org/abs/1607.06450

    计算公式:
        y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

    其中:
        - E[x] 是对最后一个维度求均值
        - Var[x] 是对最后一个维度求方差
        - gamma (weight) 是可学习的缩放参数
        - beta (bias) 是可学习的平移参数
        - eps 是防止除零的小常数

    相比 BatchNorm，LayerNorm 对每个样本独立归一化，
    不依赖 batch 统计量，更适合序列建模任务。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """
        初始化 LayerNorm 层。

        Args:
            normalized_shape: 要归一化的维度大小，通常是 hidden_size
            eps: 数值稳定常数，防止除以零，默认 1e-6

        需要初始化的参数:
            - weight (gamma): 形状 [normalized_shape]，初始化为 1
            - bias (beta): 形状 [normalized_shape]，初始化为 0
        """
        super().__init__()
        # 创建可学习参数 weight 和 bias
        # weight 初始化为全 1，bias 初始化为全 0
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，对输入进行层归一化。

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_size]
               或其他任何形状，只要最后一维是 normalized_shape

        Returns:
            归一化后的张量，形状与输入相同

        计算步骤:
            Step 1: 保存原始数据类型，将输入转为 float32 以提高数值稳定性
            Step 2: 计算最后一维的均值
                    对输入张量沿最后一个维度计算均值
                    结果形状: [..., 1]
            Step 3: 计算方差
                    沿最后一个维度计算方差 (使用无偏估计=False)
                    结果形状: [..., 1]
            Step 4: 标准化
                    使用公式: (x - 均值) / sqrt(方差 + eps)
                    即减去均值后除以标准差
            Step 5: 应用可学习参数
                    使用缩放系数 (gamma/weight) 乘以标准化后的值
                    加上平移系数 (beta/bias)
            Step 6: 恢复原始数据类型
        """
        pass
```

### 练习 2: 实现 RMSNorm

打开 `model/norm.py`，完成 `RMSNorm` 类：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        初始化 RMSNorm (均方根层归一化)。

        Args:
            hidden_size: 隐藏层维度
            eps: 数值稳定常数，防止除零
        """
        super().__init__()
        # 实现: 创建可学习的缩放参数 weight
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm 前向传播。

        公式: y = x / RMS(x) * weight，其中 RMS(x) = sqrt(mean(x^2) + eps)

        Args:
            x: 输入张量 [..., hidden_size]

        Returns:
            归一化后的张量
        """
        # 实现: 按照上述公式计算归一化输出
        pass
```

### 练习 3: 实现 MultiHead Attention

完成 `model/attention.py` 中的 `MultiHeadAttention` 类：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        """
        初始化 MHA 层。

        Args:
            config: ModelConfig 实例，包含:
                - hidden_size: 隐藏层维度
                - num_attention_heads: 注意力头数
                - attention_dropout: dropout 概率

        需要创建的参数:
            - q_proj: Query 投影，[hidden_size, hidden_size]
            - k_proj: Key 投影，[hidden_size, hidden_size]
            - v_proj: Value 投影，[hidden_size, hidden_size]
            - o_proj: 输出投影，[hidden_size, hidden_size]
        """
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        前向传播，实现缩放点积注意力。

        核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            position_ids: 位置 ID，用于 RoPE
            past_key_value: 缓存的 KV，用于生成阶段加速
            attention_mask: 注意力掩码
            use_cache: 是否返回 KV 缓存

        Returns:
            attn_output: 注意力输出
            present_key_value: 新的 KV 缓存 (如果 use_cache=True)
        """
        pass
```

**实现要点**:
1. 线性投影得到 Q、K、V
2. reshape 为多头形式 [batch, num_heads, seq_len, head_dim]
3. 应用 RoPE (如果配置使用)
4. 处理 KV 缓存
5. 计算注意力分数并应用掩码
6. softmax 归一化并计算输出

---

## 测试验证

```bash
# 基础测试
pytest tutorial/stage01_foundation/lesson02_norm_attention/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage01_foundation/lesson02_norm_attention/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **LayerNorm 论文**: https://arxiv.org/abs/1607.06450
- **RMSNorm 论文**: https://arxiv.org/abs/1910.07467
- **GQA 论文**: https://arxiv.org/abs/2305.13245
- **Flash Attention**: 了解 GPU 优化的注意力实现
