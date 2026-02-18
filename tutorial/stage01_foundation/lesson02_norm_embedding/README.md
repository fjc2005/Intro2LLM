# 课时2：归一化层与嵌入层

## 学习目标

1. 理解LayerNorm和RMSNorm的数学原理与实现差异
2. 掌握Pre-LN与Post-LN架构的优劣对比
3. 理解Token Embedding的作用与实现
4. 掌握RoPE旋转位置编码的完整算法

---

## 1. 归一化层 (Normalization)

### 1.1 为什么需要归一化？

**深度网络训练中的问题**:
```
Internal Covariate Shift (内部协变量偏移)
    └── 深层网络中，每一层的输入分布随前层参数更新而变化
    └── 导致训练困难，需要较低学习率

梯度问题
    └── 深层网络梯度可能消失(vanishing)或爆炸(exploding)
    └── 梯度经过多层传播呈指数级变化
```

**归一化的作用**:
- 稳定每层的输入分布
- 加速收敛
- 允许使用更大学习率
- 提供一定程度的正则化

### 1.2 LayerNorm (层归一化)

**论文**: [Layer Normalization (2016)](https://arxiv.org/abs/1607.06450)

**数学公式**:

```
输入: x ∈ R^{d}  (d为特征维度)

Step 1: 计算均值
    μ = (1/d) * Σ_{i=1}^{d} x_i

Step 2: 计算方差
    σ² = (1/d) * Σ_{i=1}^{d} (x_i - μ)²

Step 3: 归一化
    x̂_i = (x_i - μ) / √(σ² + ε)

Step 4: 缩放和平移 (可学习参数)
    y_i = γ * x̂_i + β

其中:
    ε: 数值稳定性小常数 (如1e-6)
    γ (gain): 可学习缩放参数，形状[d]
    β (bias): 可学习偏移参数，形状[d]
```

**张量形状变换**:
```
输入x: [batch_size, seq_len, hidden_size]

计算均值/方差维度: last dimension (hidden_size)
μ, σ²: [batch_size, seq_len, 1]

归一化后: [batch_size, seq_len, hidden_size]
γ, β: [hidden_size]

输出y: [batch_size, seq_len, hidden_size]
```

### 1.3 RMSNorm (Root Mean Square Layer Normalization)

**论文**: [Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467)

**核心洞察**: LayerNorm中的re-centering (减均值)对性能影响不大

**数学公式**:

```
输入: x ∈ R^{d}

Step 1: 计算均方根 (不计算均值，仅计算幅值)
    RMS(x) = √( (1/d) * Σ_{i=1}^{d} x_i² )

Step 2: 归一化 (无偏移，仅缩放)
    x̂_i = x_i / RMS(x)

Step 3: 缩放 (可学习参数)
    y_i = γ * x̂_i

简化公式:
    y = x / √(mean(x²) + ε) * γ
```

**与LayerNorm对比**:

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算均值 | ✓ | ✗ |
| 计算方差 | ✓ (标准差) | ✓ (均方根) |
| 偏移参数(β) | ✓ | ✗ |
| 参数量 | 2 * hidden_size | hidden_size |
| 速度 | 较慢 | 较快 |
| 使用模型 | BERT, GPT-2 | LLaMA, Qwen, Mistral |

### 1.4 Pre-LN vs Post-LN

**架构对比**:

```python
# Post-LN (原始Transformer)
x = x + Sublayer(x)      # 先执行子层(Attention/FFN)
x = LayerNorm(x)         # 再归一化

# Pre-LN (现代LLM标准)
x = x + Sublayer(LayerNorm(x))  # 先归一化，再执行子层
```

**差异可视化**:

```
Post-LN Transformer Block:
    Input ────┬──→ [LayerNorm] ──→ [Attention] ──┐
              │                                  └──→ Add ───┬──→ [LayerNorm] ──→ [FFN] ──┐
              │                                             │                             └──→ Add ──→ Output
              └─────────────────────────────────────────────┘

Pre-LN Transformer Block:
    Input ────┬──→ [LayerNorm] ──→ [Attention] ──┐
              │                                  └──→ Add ───┬──→ [LayerNorm] ──→ [FFN] ──┐
              │                                             │                             └──→ Add ──→ Output
              └─────────────────────────────────────────────┘
```

**优劣对比**:

| 特性 | Post-LN | Pre-LN |
|------|---------|--------|
| 训练稳定性 | 较差，需要学习率warmup | 更好，可以使用更大学习率 |
| 残差连接传播 | 经过归一化，梯度可能消失 | 更直接，梯度流动更好 |
| 初始化敏感度 | 高 | 低 |
| 下游任务性能 | 略好(原始论文) | 相当或更好 |
| 现代使用 | 较少 | 主流(LLaMA, GPT-3, etc.) |

**数值稳定性问题**:

```
Post-LN的问题:
    深层网络中，归一化在残差连接之后
    梯度反向传播时需要穿过归一化层
    可能导致梯度消失

Pre-LN的优势:
    残差连接 "x + Sublayer(...)" 中的x是干净的
    梯度可以直接通过x回流，不受归一化影响
```

---

## 2. 嵌入层 (Embedding)

### 2.1 Token Embedding

**作用**: 将离散的token ID映射为连续的向量表示

**数学表示**:
```
输入: token_id ∈ {0, 1, ..., vocab_size-1}

Embedding矩阵: W ∈ R^{vocab_size × hidden_size}

输出: embedding = W[token_id] ∈ R^{hidden_size}

批量形式:
    输入: [batch_size, seq_len] 的token IDs
    输出: [batch_size, seq_len, hidden_size] 的向量
```

**PyTorch实现要点**:
```python
# 核心操作: 索引查找 (Index Lookup)
# 等价于: W[input_ids] 或 F.embedding(input_ids, W)
```

### 2.2 权重共享 (Weight Tying)

**概念**: 输入Embedding与输出投影共享同一组权重

```
标准设置:
    Input Embedding:  vocab_size × hidden_size
    Output Projection: hidden_size × vocab_size
    总计: 2 * vocab_size * hidden_size

权重共享:
    Input = Output^T
    总计: vocab_size * hidden_size
    节省约50%参数量 (当vocab_size较大时显著)
```

**使用场景**:
- GPT-2, LLaMA等使用权重共享
- BERT不使用(因为BERT不是生成模型)

### 2.3 位置编码

**为什么需要位置编码？**

```
问题: 自注意力机制是位置无关的(permutation equivariant)
    Attention(Q, K, V) 对输入序列的顺序不敏感
    "我爱猫"和"猫爱我"会产生相同的注意力模式

解决: 为每个位置引入唯一的位置信息
```

**位置编码类型对比**:

| 类型 | 公式/描述 | 特点 |
|------|-----------|------|
| Sinusoidal | PE(pos, 2i) = sin(pos/10000^{2i/d}) | 固定，可外推 |
| Learned | 可学习的位置嵌入 | 灵活，不能外推 |
| RoPE | 旋转位置编码 | 相对位置感知，可外推 |
| ALiBi | 基于距离的装饰偏置 | 训练稳定，外推强 |

### 2.4 RoPE (Rotary Position Embedding) 旋转位置编码

**论文**: [RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)

**核心思想**: 通过旋转矩阵将位置信息编码到查询和键向量中

**二维旋转矩阵**:
```
对于二维向量 (x, y)，旋转θ角度:
    [cosθ  -sinθ] [x]   [x*cosθ - y*sinθ]
    [sinθ   cosθ] [y] = [x*sinθ + y*cosθ]
```

**扩展到多维 (成对旋转)**:
```
对于d维向量，每两个维度一组进行旋转:
    维度(0,1)旋转θ
    维度(2,3)旋转2θ
    维度(4,5)旋转3θ
    ...

旋转角度θ与位置pos成正比:
    θ_i = pos * 10000^{-2i/d}
```

**完整RoPE公式**:

```
对于位置pos的向量x ∈ R^d:

Step 1: 将x分解为d/2对
    x = [x_0, x_1, x_2, x_3, ..., x_{d-2}, x_{d-1}]

Step 2: 为每对计算旋转角度
    θ_i = pos * 10000^{-2i/d},  i ∈ [0, d/2)

Step 3: 应用旋转
    [x'_{2i}  ]   [cos(θ_i)  -sin(θ_i)] [x_{2i}  ]
    [x'_{2i+1}] = [sin(θ_i)   cos(θ_i)] [x_{2i+1}]

简化形式:
    x'_{2i}   = x_{2i} * cos(θ_i) - x_{2i+1} * sin(θ_i)
    x'_{2i+1} = x_{2i} * sin(θ_i) + x_{2i+1} * cos(θ_i)

其中cos(θ_i), sin(θ_i)称为旋转位置编码
```

**高效的复数/极坐标视角**:
```
将每对(x_{2i}, x_{2i+1})看作复数: z = x_{2i} + i*x_{2i+1}
旋转等价于乘以复数: e^{iθ} = cos(θ) + i*sin(θ)

z' = z * e^{iθ}
```

**RoPE的关键性质**:
```
1. 相对位置编码: RoPE(q_m)^T · RoPE(k_n) 只依赖于(m-n)
   → 模型自然学习相对位置关系

2. 长序列外推: 训练时max_len=2048，推理时可扩展到更长
   (效果逐渐下降，但比绝对位置编码好)

3. 与Attention兼容: 直接在Q、K上应用，不修改Attention计算
```

---

## 3. 实现指引

### 3.1 model/norm.py

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization 实现

    公式: y = (x - μ) / √(σ² + ε) * γ + β

    输入形状: [..., hidden_size]
    输出形状: [..., hidden_size]
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Step 1: 初始化可学习参数
        # self.weight (γ): 形状[hidden_size]，初始化为1
        # self.bias (β): 形状[hidden_size]，初始化为0
        # Step 2: 保存eps
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size]
        Returns:
            normalized: [..., hidden_size]
        """
        # Step 1: 计算输入的原始数据类型(用于最后转换回来)
        # original_dtype = x.dtype

        # Step 2: 转换为float32进行数值稳定计算
        # x = x.float()

        # Step 3: 计算均值 (沿最后一个维度)
        # mean = x.mean(dim=-1, keepdim=True)
        # Shape: [..., 1]

        # Step 4: 计算方差 (沿最后一个维度)
        # variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        # Shape: [..., 1]

        # Step 5: 归一化
        # x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        # Step 6: 缩放和平移
        # output = x_norm * self.weight + self.bias

        # Step 7: 转回原始数据类型
        pass


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    公式: y = x / √(mean(x²) + ε) * weight

    输入形状: [..., hidden_size]
    输出形状: [..., hidden_size]

    LLaMA/Qwen等现代LLM使用
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Step 1: 初始化可学习参数
        # self.weight: 形状[hidden_size]，初始化为1
        # Step 2: 保存eps
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., hidden_size]
        Returns:
            normalized: [..., hidden_size]
        """
        # Step 1: 保存原始数据类型
        # original_dtype = x.dtype

        # Step 2: 转换为float32
        # x = x.float()

        # Step 3: 计算均方值 (mean of x^2)
        # variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Shape: [..., 1]

        # Step 4: 计算归一化因子
        # x_norm = x * torch.rsqrt(variance + self.eps)
        # rsqrt = 1 / sqrt，数值更稳定

        # Step 5: 应用可学习权重
        # output = x_norm * self.weight

        # Step 6: 转回原始数据类型
        pass
```

### 3.2 model/embedding.py

```python
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Token Embedding层

    将token IDs转换为密集向量表示
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # Step 1: 创建Embedding层
        # nn.Embedding(vocab_size, hidden_size)
        pass

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - token IDs
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        # Step 1: 直接通过embedding层查找
        # embeddings = self.embedding(input_ids)
        pass


class RoPE(nn.Module):
    """
    Rotary Position Embedding (旋转位置编码)

    核心: 通过旋转矩阵将位置信息编码到查询和键中
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # Step 1: 保存配置
        # self.head_dim = head_dim
        # self.max_seq_len = max_seq_len
        # self.base = base

        # Step 2: 预计算旋转角度θ
        # 为每对维度计算: θ_i = 1 / (base^(2i/head_dim))
        # i ∈ [0, head_dim/2)
        # inv_freq形状: [head_dim/2]

        # Step 3: 预计算所有位置的sin/cos缓存
        # positions = torch.arange(max_seq_len)
        # angles = positions[:, None] * inv_freq[None, :]  # [max_seq_len, head_dim/2]
        # 扩展为成对形式: [max_seq_len, head_dim]
        # 缓存cos和sin值
        pass

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        应用RoPE到输入张量

        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
               通常是Q或K矩阵
            seq_len: 当前序列长度
        Returns:
            rotated_x: 应用位置编码后的张量，形状相同
        """
        # Step 1: 从缓存获取对应位置的cos, sin
        # cos = self.cos_cached[:seq_len]  # [seq_len, head_dim]
        # sin = self.sin_cached[:seq_len]

        # Step 2: 将x分解为两两一组
        # x shape: [..., head_dim]
        # 转换为: [..., head_dim/2, 2]
        # x1 = x[..., 0::2]  # 偶数维度
        # x2 = x[..., 1::2]  # 奇数维度

        # Step 3: 应用旋转
        # rotated_x1 = x1 * cos - x2 * sin
        # rotated_x2 = x1 * sin + x2 * cos

        # Step 4: 交错合并回原始形状
        # output[..., 0::2] = rotated_x1
        # output[..., 1::2] = rotated_x2
        pass

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        辅助函数: 旋转张量的一半维度

        用于另一种等效的RoPE实现方式
        """
        # Step 1: 分割最后维度
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]

        # Step 2: 交错重排: [-x2, x1]
        pass


class RotaryEmbedding(nn.Module):
    """
    完整的旋转位置编码模块

    通常用于在Attention中同时编码Q和K
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # 初始化RoPE实例
        pass

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对Q和K应用旋转位置编码

        Args:
            q: [batch_size, num_heads, seq_len, head_dim]
            k: [batch_size, num_heads, seq_len, head_dim]
        Returns:
            q_rotated, k_rotated: 应用位置编码后的Q和K
        """
        # 获取seq_len
        # 对q和k分别应用rope
        pass
```

---

## 4. 关键公式总结

### LayerNorm
```
μ = mean(x, dim=-1, keepdim=True)
σ² = var(x, dim=-1, keepdim=True)
y = (x - μ) / √(σ² + ε) * γ + β
```

### RMSNorm (LLaMA/Qwen风格)
```
rms = √(mean(x², dim=-1, keepdim=True) + ε)
y = x / rms * weight
```

### RoPE 旋转
```
θ_i = pos * base^{-2i/d}

[x'_{2i}  ]   [cos(θ_i)  -sin(θ_i)] [x_{2i}  ]
[x'_{2i+1}] = [sin(θ_i)   cos(θ_i)] [x_{2i+1}]
```

### 参数量对比
```
LayerNorm: 2 * hidden_size (γ + β)
RMSNorm: 1 * hidden_size (仅weight)
TokenEmbedding: vocab_size * hidden_size
RoPE: 0 (无可学习参数，纯计算)
```

---

## 5. 常见陷阱与注意事项

1. **数值稳定性**: 始终使用float32计算归一化，再转回原类型
2. **keepdim=True**: 均值/方差计算时必须保持维度用于广播
3. **RMSNorm无偏移**: 不要添加bias参数
4. **RoPE缓存**: 预计算sin/cos避免重复计算，但要注意显存占用
5. **head_dim必须偶数**: RoPE要求维度可以成对旋转
6. **position偏移**: 注意处理KV cache时的位置偏移计算

---

## 6. 课后练习

1. **手动计算LayerNorm**: 给定x = [1.0, 2.0, 3.0, 4.0], γ=1, β=0, ε=1e-6，手动计算输出
2. **RMSNorm简化**: 证明RMSNorm在均值为0时等价于LayerNorm(γ=1, β=0)
3. **RoPE可视化**: 实现并可视化不同位置的旋转效果
4. **性能对比**: 测试LayerNorm和RMSNorm的推理速度差异
5. **位置编码外推**: 在seq_len=512上训练RoPE，测试在seq_len=1024上的效果
