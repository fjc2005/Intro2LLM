# 课时3：注意力机制 - MHA、GQA与RoPE集成

## 学习目标

1. 深入理解Scaled Dot-Product Attention的数学原理
2. 掌握Multi-Head Attention的并行计算机制
3. 理解并实现因果掩码(Causal Mask)
4. 掌握GQA(Grouped Query Attention)的实现
5. 集成RoPE到注意力计算中

---

## 1. Scaled Dot-Product Attention

### 1.1 核心公式

**论文**: [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

其中:
    Q: Query矩阵, 形状 [batch, seq_len, dim]
    K: Key矩阵, 形状 [batch, seq_len, dim]
    V: Value矩阵, 形状 [batch, seq_len, dim]
    d_k: Key的维度 (dim / num_heads)
```

### 1.2 完整计算流程

```
输入: Q, K, V ∈ R^{batch × seq_len × dim}

Step 1: 计算注意力分数 (Attention Scores)
    scores = Q @ K^T
    形状: [batch, seq_len, seq_len]
    含义: 每个位置对其他所有位置的注意力强度

Step 2: 缩放 (Scaling)
    scaled_scores = scores / √d_k
    目的: 防止softmax进入梯度饱和区

Step 3: 应用掩码 (Masking, 可选但Decoder中必需)
    masked_scores = scaled_scores + mask
    因果掩码: 下三角为0，上三角为-∞

Step 4: Softmax归一化
    attn_weights = softmax(masked_scores, dim=-1)
    形状: [batch, seq_len, seq_len]
    性质: 每行和为1

Step 5: 加权求和
    output = attn_weights @ V
    形状: [batch, seq_len, dim]
```

### 1.3 为什么要缩放？(Why Scale?)

```
问题: 当d_k较大时，QK^T的点积值会很大
    - Q, K的元素是随机初始化的，方差≈1
    - QK^T的元素数量级 ≈ d_k
    - 方差会随d_k增大而增大

后果:
    - softmax输入值很大 → 梯度很小
    - 梯度消失，难以训练

解决方案: 除以√d_k
    - 将点积值缩放到单位方差
    - 保持softmax输入在合理范围
```

**数学证明**:
```
假设 Q, K 的元素独立同分布，均值为0，方差为1

E[QK^T] = Σ E[q_i * k_i] = Σ E[q_i] * E[k_i] = 0

Var(QK^T) = Σ Var(q_i * k_i)
          = Σ (E[q_i²] * E[k_i²] - E[q_i]² * E[k_i]²)
          = Σ (1 * 1 - 0)
          = d_k

缩放后 Var(QK^T / √d_k) = Var(QK^T) / d_k = 1 ✓
```

---

## 2. Multi-Head Attention (MHA)

### 2.1 核心思想

```
单头注意力的局限:
    - 只能捕捉一种注意力模式
    - 不同语义关系需要不同注意力

多头注意力的优势:
    - 并行计算多组注意力
    - 每组学习不同的注意力模式
    - 例如: 语法关系、指代关系、语义关系等
```

### 2.2 计算流程

```
输入: hidden_states ∈ R^{batch × seq_len × hidden_size}

Step 1: 线性投影生成Q、K、V
    Q = hidden_states @ W_q  (W_q: [hidden_size, hidden_size])
    K = hidden_states @ W_k  (W_k: [hidden_size, hidden_size])
    V = hidden_states @ W_v  (W_v: [hidden_size, hidden_size])

Step 2: 分割为多头
    Q: [batch, seq_len, num_heads, head_dim]
    K: [batch, seq_len, num_heads, head_dim]
    V: [batch, seq_len, num_heads, head_dim]
    其中 head_dim = hidden_size / num_heads

    转置后:
    Q: [batch, num_heads, seq_len, head_dim]
    K: [batch, num_heads, seq_len, head_dim]
    V: [batch, num_heads, seq_len, head_dim]

Step 3: 并行计算多头注意力
    attn_output = softmax(Q @ K^T / √head_dim) @ V
    形状: [batch, num_heads, seq_len, head_dim]

Step 4: 拼接多头
    concat: [batch, seq_len, num_heads, head_dim]
    → [batch, seq_len, hidden_size]

Step 5: 输出投影
    output = concat @ W_o  (W_o: [hidden_size, hidden_size])
```

### 2.3 多头并行计算示意

```
输入X: [batch, seq_len, hidden_size]

          ┌──→ Linear → Split ──→ Head 1 ──┐
          ├──→ Linear → Split ──→ Head 2 ──┤
X ──→ Linear Projection                ├──→ Concat ──→ Linear ──→ Output
          ├──→ Linear → Split ──→ Head 3 ──┤
          └──→ Linear → Split ──→ Head 4 ──┘

每个Head独立计算Attention:
    Head_i = Attention(Q_i, K_i, V_i)
```

---

## 3. 因果掩码 (Causal Mask)

### 3.1 为什么需要因果掩码？

```
语言模型的核心任务: 预测下一个token

约束: 预测位置i时，只能看到位置0到i-1的信息
      不能"偷看"未来的token

无掩码的注意力矩阵 (自注意力):
              pos0  pos1  pos2  pos3
    pos0      1     0.3   0.2   0.1    ← pos0可以看所有位置 (错误!)
    pos1      0.2   1     0.4   0.3    ← pos1可以看pos2,3 (错误!)
    pos2      0.1   0.2   1     0.5
    pos3      0.3   0.1   0.2   1

有因果掩码的注意力矩阵 (下三角):
              pos0  pos1  pos2  pos3
    pos0      1     0     0     0      ← pos0只能看自己
    pos1      0.5   1     0     0      ← pos1能看pos0,1
    pos2      0.3   0.4   1     0      ← pos2能看pos0,1,2
    pos3      0.2   0.3   0.5   1      ← pos3能看pos0,1,2,3
```

### 3.2 因果掩码实现

```python
# 方法1: 使用torch.triu创建上三角掩码
def create_causal_mask(seq_len):
    # 创建上三角矩阵(不含对角线)为True
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # 将True位置填充为-∞
    causal_mask = torch.zeros(seq_len, seq_len)
    causal_mask.masked_fill_(mask, float('-inf'))
    return causal_mask

# 方法2: 使用torch.full直接创建
seq_len = 4
causal_mask = torch.full((seq_len, seq_len), float('-inf'))
causal_mask = torch.triu(causal_mask, diagonal=1)
# 结果: [[0, -inf, -inf, -inf],
#        [0, 0, -inf, -inf],
#        [0, 0, 0, -inf],
#        [0, 0, 0, 0]]
```

### 3.3 掩码在注意力中的应用

```
scores: [batch, num_heads, seq_len, seq_len]
mask:   [seq_len, seq_len] 或 [batch, 1, seq_len, seq_len]

masked_scores = scores + mask
# -inf加任何数还是-inf，softmax后变为0

attn_weights = softmax(masked_scores, dim=-1)
# 上三角位置的注意力权重变为0
```

---

## 4. Grouped Query Attention (GQA)

### 4.1 从MHA到GQA的演进

```
MHA (Multi-Head Attention):
    Q头数: 32, K头数: 32, V头数: 32
    每个Q有独立的K、V
    KV缓存: 32 * seq_len * head_dim

GQA (Grouped Query Attention):
    Q头数: 32, K头数: 8, V头数: 8
    每4个Q共享1组K、V
    KV缓存: 8 * seq_len * head_dim (减少75%)

MQA (Multi-Query Attention):
    Q头数: 32, K头数: 1, V头数: 1
    所有Q共享1组K、V
    KV缓存: 1 * seq_len * head_dim (减少97%)
```

### 4.2 GQA计算流程

```
输入: hidden_states ∈ R^{batch × seq_len × hidden_size}

Step 1: 投影 (注意K、V头数减少)
    Q = hidden_states @ W_q  → [batch, seq_len, hidden_size]
    K = hidden_states @ W_k  → [batch, seq_len, num_kv_heads * head_dim]
    V = hidden_states @ W_v  → [batch, seq_len, num_kv_heads * head_dim]

Step 2: 分割为多头
    Q: [batch, seq_len, num_heads, head_dim]
    K: [batch, seq_len, num_kv_heads, head_dim]
    V: [batch, seq_len, num_kv_heads, head_dim]

Step 3: 扩展K、V以匹配Q头数
    # K, V形状: [batch, num_kv_heads, seq_len, head_dim]
    # 需要重复每个KV头 num_heads/num_kv_heads 次
    K = K.repeat_interleave(num_heads // num_kv_heads, dim=1)
    V = V.repeat_interleave(num_heads // num_kv_heads, dim=1)
    # 现在K, V形状: [batch, num_heads, seq_len, head_dim]

Step 4: 正常计算注意力
    attn_output = softmax(Q @ K^T / √head_dim) @ V
```

### 4.3 GQA优势分析

```
内存节省:
    MHA:  2 * batch * num_heads * seq_len * head_dim
    GQA:  2 * batch * num_kv_heads * seq_len * head_dim
    节省比例: num_kv_heads / num_heads

计算量:
    相同: QK^T和softmax的计算量不变
    减少: K、V的投影参数量和计算量

质量:
    MQA: 质量下降较明显
    GQA: 质量接近MHA，内存效率接近MQA
    推荐: num_kv_heads = num_heads / 4
```

---

## 5. 实现指引

### 5.1 model/attention.py

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力

    公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_heads, seq_len, head_dim]
            v: [batch, num_heads, seq_len, head_dim]
            mask: [batch, 1, seq_len, seq_len] 或兼容形状
            dropout: 可选的dropout层

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Step 1: 获取维度信息
        # batch_size, num_heads, seq_len, head_dim = q.shape

        # Step 2: 计算注意力分数 Q @ K^T
        # scores = torch.matmul(q, k.transpose(-2, -1))
        # scores shape: [batch, num_heads, seq_len, seq_len]

        # Step 3: 缩放
        # scale = math.sqrt(head_dim)
        # scores = scores / scale

        # Step 4: 应用掩码 (如果提供)
        # if mask is not None:
        #     scores = scores + mask
        # mask中的-inf会使softmax后对应位置为0

        # Step 5: Softmax
        # attn_weights = F.softmax(scores, dim=-1)

        # Step 6: Dropout (如果提供)
        # if dropout is not None:
        #     attn_weights = dropout(attn_weights)

        # Step 7: 加权求和
        # output = torch.matmul(attn_weights, v)

        pass


class MultiHeadAttention(nn.Module):
    """
    多头注意力 (MHA)

    支持可选的GQA和RoPE
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        # Step 1: 保存配置
        # self.hidden_size = hidden_size
        # self.num_heads = num_attention_heads
        # self.num_kv_heads = num_key_value_heads or num_attention_heads
        # self.head_dim = hidden_size // num_attention_heads
        # self.num_kv_groups = num_attention_heads // self.num_kv_heads

        # Step 2: 创建Q、K、V投影层
        # q_proj: [hidden_size, num_heads * head_dim] = [hidden_size, hidden_size]
        # k_proj: [hidden_size, num_kv_heads * head_dim]
        # v_proj: [hidden_size, num_kv_heads * head_dim]
        # o_proj: [hidden_size, hidden_size]

        # Step 3: 可选的RoPE
        # if use_rope:
        #     self.rope = RoPE(self.head_dim, max_position_embeddings, rope_base)

        # Step 4: Dropout
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            past_key_value: 用于KV缓存的先前K、V
            use_cache: 是否返回KV用于缓存

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            present_key_value: 可选的(K, V)元组用于缓存
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: 线性投影得到Q、K、V
        # q = self.q_proj(hidden_states)
        # k = self.k_proj(hidden_states)
        # v = self.v_proj(hidden_states)
        # Shape: [batch, seq_len, num_heads*head_dim]或[num_kv_heads*head_dim]

        # Step 2: 重塑为多头形式
        # q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, head_dim] 或 [batch, num_kv_heads, ...]

        # Step 3: 应用RoPE (如果启用)
        # if hasattr(self, 'rope'):
        #     q, k = self.rope(q, k)

        # Step 4: 处理KV缓存
        # if past_key_value is not None:
        #     past_k, past_v = past_key_value
        #     k = torch.cat([past_k, k], dim=2)  # 在seq_len维度拼接
        #     v = torch.cat([past_v, v], dim=2)
        # kv_seq_len = k.shape[2]

        # Step 5: GQA - 扩展K、V以匹配Q头数
        # if self.num_kv_groups > 1:
        #     k = k.repeat_interleave(self.num_kv_groups, dim=1)
        #     v = v.repeat_interleave(self.num_kv_groups, dim=1)
        # Now k, v shape: [batch, num_heads, kv_seq_len, head_dim]

        # Step 6: 创建因果掩码 (如果是训练或生成阶段)
        # if attention_mask is None and self.training:
        #     causal_mask = torch.triu(
        #         torch.full((seq_len, kv_seq_len), float('-inf')),
        #         diagonal=1
        #     ).to(hidden_states.device)
        #     attention_mask = causal_mask

        # Step 7: 计算注意力
        # attn_output, attn_weights = scaled_dot_product_attention(
        #     q, k, v, mask=attention_mask
        # )
        # attn_output shape: [batch, num_heads, seq_len, head_dim]

        # Step 8: 合并多头
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Step 9: 输出投影
        # output = self.o_proj(attn_output)

        # Step 10: 返回结果和KV缓存(如果需要)
        pass
```

---

## 6. 关键公式总结

### Scaled Dot-Product Attention
```
scores = Q @ K^T / √d_k
attn = softmax(scores + mask)
output = attn @ V
```

### Multi-Head Attention
```
Q = X @ W_q    # [batch, seq, hidden] @ [hidden, hidden]
K = X @ W_k
V = X @ W_v

split → num_heads → Attention → concat → X @ W_o
```

### GQA重复因子
```
num_kv_groups = num_heads / num_kv_heads
K_repeated = K.repeat_interleave(num_kv_groups, dim=1)
V_repeated = V.repeat_interleave(num_kv_groups, dim=1)
```

### 因果掩码
```
mask[i, j] = 0           if j <= i (下三角和对角线)
mask[i, j] = -inf        if j > i  (上三角)
```

---

## 7. 常见陷阱与注意事项

1. **缩放因子的位置**: 务必在softmax前缩放，不要在后
2. **因果掩码的维度**: 确保掩码可以广播到[batch, heads, seq, seq]
3. **GQA的repeat次数**: num_heads必须整除num_kv_heads
4. **KV缓存的拼接维度**: 在seq_len维度拼接(dim=2)
5. **RoPE的应用位置**: 在分割多头后、注意力计算前应用
6. **softmax的dim**: 确保dim=-1(最后一个维度)
7. **transpose后的contiguous**: view前调用contiguous避免内存布局问题

---

## 8. 课后练习

1. **手动计算Attention**: 给定Q=[[1,0],[0,1]], K=[[1,1],[0,1]], V=[[1,2],[3,4]]，手动计算输出
2. **复杂度分析**: 计算MHA的时间复杂度和空间复杂度
3. **GQA内存对比**: 计算7B模型在seq_len=4096时，MHA/GQA/MQA的KV缓存大小
4. **掩码设计**: 设计一个Padding掩码(非因果)，用于处理变长序列
5. **多头可视化**: 训练一个小模型，可视化不同头的注意力模式差异
