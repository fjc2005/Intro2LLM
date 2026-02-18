# L15: 高效注意力机制

## 学习目标

1. **理解** Flash Attention 原理
2. **掌握** MQA 和 GQA
3. **了解** 各种注意力优化技术

---

## 理论背景

### 1. 标准注意力的计算复杂度

标准 Self-Attention 的计算:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**计算复杂度**:
- $QK^T$: $O(N^2 d)$ (N = 序列长度)
- Softmax: $O(N^2)$
- 加权求和: $O(N^2 d)$

**显存占用**:
- $Q, K, V$: $O(N^2 d)$ (需要存储完整的注意力矩阵)

### 2. Flash Attention

#### 2.1 IO-Aware 算法

Flash Attention 的核心思想是将注意力计算重新组织为分块计算，减少 HBM (High Bandwidth Memory) 访问。

#### 2.2 分块计算

```python
# 标准实现
S = Q @ K^T  # 完整的 NxN 矩阵，存储在显存
P = softmax(S)  # 完整的 NxN 矩阵
O = P @ V  # 完整的 NxN 矩阵

# Flash Attention 实现
# 分块计算，避免存储完整矩阵
for block in blocks:
    S_block = Q_block @ K^T  # 只计算当前块
    P_block = softmax(S_block)
    O += P_block @ V_block
```

#### 2.3 Online Softmax

标准的 softmax 需要两次遍历:
1. 计算最大值和指数和
2. 归一化

Flash Attention 使用 online softmax，只需一次遍历:

$$\text{softmax}_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$$

其中 $m = \max(x)$ 是当前最大值。

#### 2.4 优势

- **显存**: 从 $O(N^2)$ 降到 $O(N)$
- **计算量**: 保持不变
- **速度**: 显著提升 (减少显存访问)

### 3. MQA (Multi-Query Attention)

#### 3.1 背景

标准 MHA (Multi-Head Attention) 每个头有独立的 Q、K、V。

#### 3.2 MQA 核心思想

所有 Q 头共享同一对 K、V:

$$K_{shared} = K_1 = K_2 = ... = K_h$$
$$V_{shared} = V_1 = V_2 = ... = V_h$$

#### 3.3 优势

- **显存**: KV Cache 大幅减少
- **推理速度**: 更快的预填充阶段

#### 3.4 问题

共享 K/V 可能导致质量下降。

### 4. GQA (Grouped-Query Attention)

#### 4.1 核心思想

将 Q 头分组，每组共享 K、V:

$$num\_groups = \frac{num\_heads}{num\_kv\_heads}$$

#### 4.2 对比

| 特性 | MHA | MQA | GQA |
|------|-----|-----|-----|
| K/V 头数 | h | 1 | g (g << h) |
| KV Cache | O(Nh) | O(N) | O(Ng) |
| 质量 | 最高 | 较低 | 接近 MHA |
| 速度 | 较慢 | 最快 | 较快 |

### 5. 其他优化技术

#### 5.1 稀疏注意力 (Sparse Attention)

只计算部分 token 之间的注意力:
- 局部窗口注意力
- 随机注意力
- 块稀疏注意力

#### 5.2 线性注意力

使用核函数近似注意力:
$$Attention(Q, K, V) = \phi(Q) (\phi(K)^T V)$$

#### 5.3 滑动窗口注意力

使用固定大小的滑动窗口:

```
位置 0: 看 [0]
位置 1: 看 [0, 1]
位置 2: 看 [0, 1, 2]
位置 3: 看 [0, 1, 2, 3]
位置 4: 看 [1, 2, 3, 4]
位置 5: 看 [2, 3, 4, 5]
...
```

---

## 代码实现

### 项目结构

```
model/
├── attention.py       # 注意力实现
├── flash_attention.py # Flash Attention
└── gqa.py            # GQA 实现
```

---

## 实践练习

### 练习 1: 实现 GroupedQueryAttention

打开 `model/attention.py`，实现 `GroupedQueryAttention` 类:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        """
        初始化 GQA 层。

        Args:
            config: ModelConfig 实例，包含:
                - hidden_size: 隐藏层维度
                - num_attention_heads: 注意力头数
                - num_key_value_heads: KV 头数
                - attention_dropout: dropout 概率
        """
        pass

    def repeat_kv(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        复制 KV 头以匹配 Q 头数量。

        Args:
            x: KV 张量 [batch, num_kv_heads, seq_len, head_dim]
            num_groups: 复制次数

        Returns:
            复制后的张量 [batch, num_heads, seq_len, head_dim]
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
        前向传播。

        实现要点:
        1. Q 投影: 完整 num_heads
        2. K/V 投影: 只有 num_kv_heads
        3. 使用 repeat_kv 扩展 K/V
        4. 计算标准注意力
        """
        pass
```

### 练习 2: 分析 GQA 显存优势

比较 MHA 和 GQA 的 KV Cache 大小:

```python
# MHA: num_kv_heads = num_attention_heads
# GQA: num_kv_heads < num_attention_heads

# KV Cache 大小对比
# MHA: O(batch * num_heads * seq * head_dim)
# GQA: O(batch * num_kv_heads * seq * head_dim)
```

### 练习 3: 实现滑动窗口注意力

在 `MultiHeadAttention` 或 `GroupedQueryAttention` 中添加滑动窗口支持:

```python
def forward_with_sliding_window(
    self,
    hidden_states,
    attention_mask,
    window_size=512,
    ...
):
    """
    实现滑动窗口注意力。

    每个位置只关注 window_size 范围内的 token。
    """
    # 实现:
    # 1. 创建滑动窗口掩码
    # 2. 组合因果掩码和窗口掩码
    # 3. 应用掩码进行注意力计算
    pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage05_advanced/lesson15_efficient_attention/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage05_advanced/lesson15_efficient_attention/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **Flash Attention 论文**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **GQA 论文**: "GQA: Training Generalized Multi-Query Transformer Models"
- **GPU 优化**: 了解 CUDA 内存层次结构和优化技巧
