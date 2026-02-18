# L05: 模型配置与效率分析

## 学习目标

1. **理解** 模型配置的各个参数含义
2. **掌握** FLOPs 计算方法
3. **掌握** 显存估算方法
4. **能够** 根据硬件选择合适的模型配置

---

## 理论背景

### 1. 模型配置参数

```python
@dataclass
class ModelConfig:
    vocab_size: int          # 词表大小
    hidden_size: int         # 隐藏层维度 (d_model)
    intermediate_size: int   # FFN 中间层维度
    num_hidden_layers: int   # Transformer 层数
    num_attention_heads: int # 注意力头数
    num_key_value_heads: int # KV 头数 (GQA)
    max_position_embeddings: int  # 最大序列长度
    rope_theta: float        # RoPE 基础频率
    rms_norm_eps: float      # RMSNorm epsilon
    attention_dropout: float # 注意力 dropout
    hidden_act: str          # 激活函数
    use_rms_norm: bool       # 使用 RMSNorm
    use_rope: bool           # 使用 RoPE
    use_swiglu: bool         # 使用 SwiGLU
```

### 2. 参数量计算

**Attention 参数量**:
```
Q_proj: hidden_size * hidden_size
K_proj: hidden_size * (num_kv_heads * head_dim)
V_proj: hidden_size * (num_kv_heads * head_dim)
O_proj: hidden_size * hidden_size
总计: ~3 * hidden_size^2 (MHA) 或 ~2 * hidden_size^2 + hidden_size * num_kv_heads * head_dim (GQA)
```

**FFN 参数量**:
```
gate_proj: hidden_size * intermediate_size
up_proj: hidden_size * intermediate_size
down_proj: intermediate_size * hidden_size
总计: 2 * hidden_size * intermediate_size (GeGLU/SwiGLU)
```

**Embedding 参数量**:
```
vocab_size * hidden_size
```

**总参数量**:
```
总参数量 ≈
  vocab_size * hidden_size  (embedding)
+ num_layers * (
    2 * hidden_size * (num_kv_heads * head_dim)  # K, V (GQA)
  + 3 * hidden_size^2  # Q, O
  + 2 * hidden_size * intermediate_size  # FFN
  + 2 * hidden_size  # 归一化
)
```

### 3. FLOPs 计算

**前向传播 FLOPs** (每 token):
```
Attention: 4 * hidden_size^2 * num_layers
FFN: 8 * hidden_size * intermediate_size * num_layers
总计: ~8 * hidden_size * (hidden_size + 2 * intermediate_size) * num_layers
```

**训练 FLOPs** (前向 + 反向):
```
训练 FLOPs ≈ 3 * 前向 FLOPs (简化估计)
```

### 4. 显存估算

**模型参数显存**:
```
参数量 * 4 bytes (FP32) / 2 bytes (FP16) / 2 bytes (BF16)
```

**KV Cache 显存**:
```
2 * batch_size * num_layers * num_kv_heads * max_seq_len * head_dim * 2 (K+V) * dtype_bytes
```

**Activation 显存**:
```
序列长度 * batch_size * hidden_size * num_layers * 各种系数
```

---

## 代码实现

### configs/model_config.py

```python
@dataclass
class ModelConfig:
    vocab_size: int          # 词表大小
    hidden_size: int         # 隐藏层维度 (d_model)
    intermediate_size: int   # FFN 中间层维度
    num_hidden_layers: int   # Transformer 层数
    num_attention_heads: int # 注意力头数
    num_key_value_heads: int # KV 头数 (GQA)
    max_position_embeddings: int  # 最大序列长度
    rope_theta: float        # RoPE 基础频率
    rms_norm_eps: float      # RMSNorm epsilon
    attention_dropout: float # 注意力 dropout
    hidden_act: str          # 激活函数
    use_rms_norm: bool       # 使用 RMSNorm
    use_rope: bool           # 使用 RoPE
    use_swiglu: bool         # 使用 SwiGLU
```

---

## 实践练习

### 练习 1: 分析 ModelConfig

查看 `configs/tiny_config.py` 或 `model/config.py`，理解各配置参数的含义。

```python
# 示例: 创建配置并查看参数
from configs.tiny_config import tiny_config

config = tiny_config()
print(f"模型参数量: {config.vocab_size * config.hidden_size}")
print(f"隐藏层维度: {config.hidden_size}")
print(f"层数: {config.num_hidden_layers}")
```

### 练习 2: 手动计算模型参数量

根据配置，手动计算 LLM 的总参数量：

```
模型参数量构成:
- Embedding 层: vocab_size * hidden_size
- 每层 Transformer:
  - Attention: Q, K, V, O 四个投影矩阵
  - FFN: gate, up, down 三个投影矩阵
  - Norm: 归一化层参数 (2 * hidden_size)
- LM Head: hidden_size * vocab_size
```

**示例计算**: 使用 tiny_config (约 10M 参数) 验证计算结果。

### 练习 3: 估算显存

基于配置估算推理和训练时的显存占用：

```
显存占用分析:
- 模型参数 (FP16): 参数量 * 2 bytes
- KV Cache: 2 * batch * num_layers * num_kv_heads * seq_len * head_dim * 2 (K+V)
- 训练时额外: 梯度 + 优化器状态 (Adam: 4x 参数显存)
```

---

## 测试验证

```bash
pytest tutorial/stage01_foundation/lesson05_model_config_efficiency/testcases/basic_test.py -v
pytest tutorial/stage01_foundation/lesson05_model_config_efficiency/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **FLOPs 优化**: 了解不同精度训练的影响
- **KV Cache**: 了解更多推理优化技术
