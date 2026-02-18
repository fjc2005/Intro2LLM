# 课时4：前馈网络与Transformer块

## 学习目标

1. 理解FFN(前馈网络)在Transformer中的作用
2. 掌握SwiGLU和GeGLU门控激活函数的数学原理
3. 理解并实现Pre-LN Transformer块
4. 掌握残差连接与梯度流动的关系
5. 理解Dropout在Transformer中的应用

---

## 1. 前馈网络 (Feed-Forward Network)

### 1.1 FFN的作用

```
FFN在Transformer中的角色:
    - 对每个位置的表示进行独立变换
    - 提供非线性能力，增强模型表达能力
    - 将注意力输出映射到更高维空间，再投影回来

为什么需要FFN?
    - 注意力机制是线性组合(加权求和)
    - FFN引入非线性，使网络能学习更复杂的函数
    - "Attention is all you need" 但FFN也很重要
```

### 1.2 标准FFN结构

**原始Transformer FFN**:
```
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

维度变换:
    x:      [batch, seq_len, hidden_size]
    W1:     [hidden_size, intermediate_size]
    b1:     [intermediate_size]
    W2:     [intermediate_size, hidden_size]
    b2:     [hidden_size]

其中 intermediate_size 通常为 2-4 * hidden_size
```

### 1.3 SwiGLU 激活函数

**论文**: [GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202)

**核心思想**: 使用门控机制控制信息流动

```
标准GLU:
    GLU(x) = σ(xW + b) ⊗ (xV + c)

SwiGLU (Swish + GLU):
    SwiGLU(x) = Swish(xW) ⊗ (xV)

其中:
    Swish(x) = x · σ(x)    (SiLU激活)
    σ(x) = sigmoid(x)
    ⊗ 表示逐元素乘法
```

**SwiGLU的FFN结构**:
```
SwiGLU_FFN(x) = (Swish(x @ W_gate) ⊗ (x @ W_up)) @ W_down

维度:
    x:          [batch, seq_len, hidden_size]
    W_gate:     [hidden_size, intermediate_size]
    W_up:       [hidden_size, intermediate_size]
    W_down:     [intermediate_size, hidden_size]

Note: SwiGLU没有偏置项(bias=False)，这是LLaMA等模型的标准做法
```

**为什么SwiGLU更好？**
```
1. 门控机制: 动态控制每个维度的信息通过量
2. Swish激活: 平滑、非单调，在负区间也有小梯度
3. 表达能力: 实验证明比ReLU/GELU有更好的下游任务性能
4. 现代LLM标准: LLaMA, Qwen, Mistral等都使用SwiGLU
```

### 1.4 GeGLU 激活函数

```
GeGLU(x) = GELU(xW) ⊗ (xV)

GELU(x) = x · Φ(x)    其中Φ是标准正态分布的CDF

简化近似:
    GELU(x) ≈ 0.5 · x · (1 + tanh[√(2/π) · (x + 0.044715 · x³)])
```

**SwiGLU vs GeGLU**:

| 特性 | SwiGLU | GeGLU |
|------|--------|-------|
| 门控激活 | Swish/SiLU | GELU |
| 平滑性 | 非常平滑 | 平滑 |
| 负区间行为 | 小负值 | 小负值 |
| 使用模型 | LLaMA, Qwen, PaLM | GPT-4(传闻), GLM |

---

## 2. Transformer块

### 2.1 Pre-LN Transformer Block

**现代LLM的标准架构**:
```
输入: x

# 注意力子层
residual = x
x = LayerNorm(x)
x = Attention(x)
x = residual + x  # 残差连接

# FFN子层
residual = x
x = LayerNorm(x)
x = FFN(x)
x = residual + x  # 残差连接

输出: x
```

### 2.2 Post-LN vs Pre-LN 深度对比

```
Post-LN (原始Transformer):
    x = LayerNorm(x + Attention(x))
    x = LayerNorm(x + FFN(x))

    问题:
    - 残差连接在LayerNorm之后
    - 梯度反向传播时需经过LayerNorm
    - 可能导致梯度消失

Pre-LN (现代LLM):
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

    优势:
    - 残差连接中的x是"干净"的
    - 梯度可以直接通过x回流
    - 训练更稳定，可以使用更大学习率
```

### 2.3 残差连接 (Residual Connection)

**数学表示**:
```
y = F(x) + x

其中F(x)是要学习的残差函数
```

**梯度流动分析**:
```
反向传播时:
    ∂y/∂x = ∂F(x)/∂x + 1

优势:
    - 即使∂F(x)/∂x很小(梯度消失)，+1保证梯度仍能流动
    - 深层网络中，梯度可以直接通过残差连接回传
    - 使得训练100+层的网络成为可能
```

**可视化 - 100层网络的梯度**:
```
无残差连接:
    gradient ≈ (0.1)^100 ≈ 0 (完全消失)

有残差连接:
    gradient ≈ 1 + small_terms ≈ 1 (健康流动)
```

### 2.4 Dropout正则化

**在Transformer中的应用位置**:
```
1. Embedding层后: 防止过度依赖特定token
2. Attention后: 防止过度依赖特定位置关系
3. FFN后: 防止过度依赖特定特征组合
4. 残差连接前: 一些实现选择在这里

标准位置 (HuggingFace Transformers):
    - Attention dropout: 在softmax后
    - Hidden dropout: 在FFN后和残差前
```

**Dropout在推理时**:
```
- 训练时: 随机置零p比例的元素，其余乘以1/(1-p)
- 推理时: 关闭dropout，使用所有神经元
- PyTorch自动处理: model.train() vs model.eval()
```

---

## 3. 实现指引

### 3.1 model/feedforward.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU激活函数

    SwiGLU(x) = Swish(xW) ⊗ (xV)

    其中 Swish(x) = x · σ(x) = SiLU(x)
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        # Step 1: 创建门控投影层 W_gate
        # nn.Linear(hidden_size, intermediate_size, bias=False)

        # Step 2: 创建上投影层 W_up
        # nn.Linear(hidden_size, intermediate_size, bias=False)

        # Step 3: 创建下投影层 W_down
        # nn.Linear(intermediate_size, hidden_size, bias=False)

        # Step 4: Dropout层
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Step 1: 计算门控值
        # gate = F.silu(self.gate_proj(x))  # Swish激活
        # Shape: [batch, seq_len, intermediate_size]

        # Step 2: 计算上投影
        # up = self.up_proj(x)
        # Shape: [batch, seq_len, intermediate_size]

        # Step 3: 门控乘法
        # gated = gate * up  # 逐元素乘法
        # Shape: [batch, seq_len, intermediate_size]

        # Step 4: 下投影
        # output = self.down_proj(gated)
        # Shape: [batch, seq_len, hidden_size]

        # Step 5: Dropout
        pass


class GeGLU(nn.Module):
    """
    GeGLU激活函数

    GeGLU(x) = GELU(xW) ⊗ (xV)
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        # 与SwiGLU类似，但使用GELU代替Swish
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Step 1: 计算门控值
        # gate = F.gelu(self.gate_proj(x))

        # Step 2: 计算上投影并相乘
        # up = self.up_proj(x)
        # gated = gate * up

        # Step 3: 下投影和dropout
        pass


class FeedForward(nn.Module):
    """
    统一的前馈网络接口

    支持不同激活函数: SwiGLU, GeGLU, ReLU, GELU
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "swiglu",
        dropout: float = 0.0,
    ):
        super().__init__()
        # Step 1: 根据hidden_act选择实现
        # if hidden_act == "swiglu":
        #     self.act = SwiGLU(hidden_size, intermediate_size, dropout)
        # elif hidden_act == "geglu":
        #     self.act = GeGLU(hidden_size, intermediate_size, dropout)
        # ...其他激活函数
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

### 3.2 model/transformer_block.py

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer块

    结构:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    特点:
        - 使用Pre-LayerNorm
        - 支持GQA和RoPE
        - 支持KV缓存
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        hidden_act: str = "swiglu",
        rms_norm_eps: float = 1e-6,
        use_rms_norm: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        # Step 1: 初始化输入归一化层
        # if use_rms_norm:
        #     self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        #     self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # else:
        #     self.input_layernorm = LayerNorm(hidden_size, eps=rms_norm_eps)
        #     self.post_attention_layernorm = LayerNorm(hidden_size, eps=rms_norm_eps)

        # Step 2: 初始化自注意力层
        # self.self_attn = MultiHeadAttention(
        #     hidden_size=hidden_size,
        #     num_attention_heads=num_attention_heads,
        #     num_key_value_heads=num_key_value_heads,
        #     dropout=attention_dropout,
        #     use_rope=True,
        #     rope_base=rope_base,
        #     max_position_embeddings=max_position_embeddings,
        # )

        # Step 3: 初始化FFN
        # self.mlp = FeedForward(
        #     hidden_size=hidden_size,
        #     intermediate_size=intermediate_size,
        #     hidden_act=hidden_act,
        #     dropout=dropout,
        # )

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
            hidden_states: [batch, seq_len, hidden_size]
            present_key_value: 可选的(K, V)元组
        """
        # Step 1: 保存残差
        # residual = hidden_states

        # Step 2: 输入归一化
        # hidden_states = self.input_layernorm(hidden_states)

        # Step 3: 自注意力 + 残差连接
        # hidden_states, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     past_key_value=past_key_value,
        #     use_cache=use_cache,
        # )
        # hidden_states = residual + hidden_states

        # Step 4: 保存残差
        # residual = hidden_states

        # Step 5: FFN前归一化
        # hidden_states = self.post_attention_layernorm(hidden_states)

        # Step 6: FFN + 残差连接
        # hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        # Step 7: 返回结果和KV缓存
        pass
```

---

## 4. 关键公式总结

### SwiGLU
```
gate = Swish(x @ W_gate) = SiLU(x @ W_gate)
up = x @ W_up
hidden = gate ⊗ up
output = hidden @ W_down
```

### Pre-LN Transformer Block
```
# Attention子层
x = x + Attention(LayerNorm(x))

# FFN子层
x = x + SwiGLU(LayerNorm(x))
```

### 参数量计算
```
Attention参数:
    4 * hidden_size²  (Q, K, V, O投影)

FFN参数 (SwiGLU):
    3 * hidden_size * intermediate_size  (gate, up, down)

总参数量 (每层):
    4 * hidden_size² + 3 * hidden_size * intermediate_size + 2 * hidden_size
```

---

## 5. 常见陷阱与注意事项

1. **SwiGLU没有bias**: 现代实现通常设置bias=False
2. **中间维度选择**: intermediate_size通常是2-4倍hidden_size，2.67倍常见
3. **Dropout位置**: 通常在残差连接前或FFN输出后
4. **残差连接顺序**: Pre-LN是先norm再sublayer，Post-LN相反
5. **数值稳定性**: 归一化层使用float32计算
6. **激活函数选择**: 现代LLM首选SwiGLU，GPT系列多用GELU

---

## 6. 课后练习

1. **手动计算SwiGLU**: 给定小维度输入，手动计算SwiGLU输出
2. **参数量对比**: 计算SwiGLU、GeGLU、ReLU-FFN的参数量差异
3. **残差连接可视化**: 实现一个深层网络，可视化有/无残差连接的梯度分布
4. **激活函数对比**: 在同一模型上对比Swish和ReLU的训练动态
5. **FFN裁剪**: 思考如何对FFN进行结构化剪枝
