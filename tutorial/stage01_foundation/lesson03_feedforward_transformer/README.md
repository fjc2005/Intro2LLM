# L03: FeedForward 与 Transformer

## 学习目标

1. **理解** GeGLU 和 SwiGLU 的区别与优势
2. **掌握** Transformer Block 的 Pre-LN 结构
3. **能够** 实现完整的 Transformer Block

---

## 理论背景

### 1. 门控线性单元 (GLU)

**论文**: "GLU Variants Improve Transformer" (Shazeer, 2020)

**核心思想**: 通过门控机制控制信息流动。

#### GeGLU

```
GeGLU(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down
```

- 使用 GELU 作为门控激活函数
- PaLM 使用 GeGLU

#### SwiGLU

```
SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
```

- 使用 SiLU/Swish 作为门控激活函数
- LLaMA、Qwen、Mistral 使用 SwiGLU

**对比**:
- GeGLU: GELU(x) = x * Φ(x)
- SwiGLU: SiLU(x) = x * sigmoid(x)
- 两者性能相当，SwiGLU 更流行

### 2. Transformer Block 结构

**Pre-LN (现代标准)**:
```
Input -> Norm -> Attention -> + -> Norm -> FFN -> + -> Output
```

vs

**Post-LN (原始)**:
```
Input -> Attention -> Norm -> FFN -> Norm -> Output
```

**为什么选择 Pre-LN**:
- Post-LN 深层网络训练不稳定
- Pre-LN 在极深网络 (96+ 层) 上训练更稳定
- 学习率可以设置更大

---

## 代码实现

### 项目结构

```
model/
├── feedforward.py       # GeGLU, SwiGLU
├── transformer_block.py # TransformerBlock
└── attention.py         # Attention
```

---

## 实践练习

### 练习 1: 实现 SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, config):
        """
        初始化 SwiGLU (Swish-Gated Linear Unit) 前馈网络。

        SwiGLU 公式: SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

        Args:
            config: 模型配置，包含 hidden_size 和 intermediate_size
        """
        # 实现: 创建三个线性投影层
        # - gate_proj: hidden_size -> intermediate_size
        # - up_proj: hidden_size -> intermediate_size
        # - down_proj: intermediate_size -> hidden_size
        pass

    def forward(self, x):
        """
        SwiGLU 前向传播。

        实现思路:
        1. 通过 gate_proj 获取门控信号，应用 SiLU 激活函数
        2. 通过 up_proj 获取上投影
        3. 门控信号与上投影逐元素相乘
        4. 通过 down_proj 降维输出
        """
        pass
```

### 练习 2: 实现 TransformerBlock

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        """
        初始化 Pre-LN 结构的 Transformer 块。

        Pre-LN 结构特点: 每个子层前放置归一化层
        Input -> Norm -> Attention -> + -> Norm -> FFN -> + -> Output

        Args:
            config: 模型配置
        """
        # 实现: 创建以下子模块
        # - input_layernorm: 注意力前的归一化
        # - self_attn: 自注意力模块
        # - post_attention_layernorm: FFN 前的归一化
        # - mlp: 前馈网络模块
        pass

    def forward(self, x, attention_mask=None, use_cache=False):
        """
        Pre-LN Transformer 块的前向传播。

        实现思路:
        1. 输入先经过 Pre-Attention 归一化，然后通过注意力模块
        2. 注意力输出与原始输入相加（残差连接）
        3. 结果再经过 Pre-FFN 归一化，然后通过 FFN 模块
        4. FFN 输出与注意力后的结果相加（残差连接）
        """
        pass
```

---

## 测试验证

```bash
pytest tutorial/stage01_foundation/lesson03_feedforward_transformer/testcases/basic_test.py -v
pytest tutorial/stage01_foundation/lesson03_feedforward_transformer/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **GLU 论文**: https://arxiv.org/abs/2002.05202
- **Pre-LN Transformer**: 了解深层 Transformer 训练稳定性
