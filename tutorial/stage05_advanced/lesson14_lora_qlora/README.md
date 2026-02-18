# L14: LoRA 与 QLoRA

## 学习目标

1. **理解** LoRA 原理和实现
2. **理解** QLoRA 量化技术
3. **掌握** 混合精度训练

---

## 理论背景

### 1. LoRA (Low-Rank Adaptation)

#### 1.1 背景

全参数微调大模型需要大量显存和计算资源。LoRA 通过低秩矩阵分解实现参数高效微调。

#### 1.2 核心思想

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 添加一个低秩更新:
$$W = W_0 + \Delta W = W_0 + BA$$

其中:
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ 是秩

#### 1.3 前向传播

$$h = W_0 x + BAx$$

实现时只需要:
1. 计算 $A^T x$ (降维)
2. 计算 $B (A^T x)$ (升维)

#### 1.4 参数量分析

原始: $d \times k$ 参数

LoRA: $d \times r + r \times k = r(d + k)$ 参数

当 $r \ll d, k$ 时，参数量大幅减少。

**例如**: $d=4096, k=4096, r=8$
- 原始: $4096^2 \approx 16M$
- LoRA: $8 \times (4096 + 4096) \approx 65K$
- 压缩比: ~250x

### 2. QLoRA (Quantized LoRA)

#### 2.1 量化基础

**量化**: 将高精度权重映射到低精度表示

**NF4 (Normalized Float 4)**: 专为 LLM 权重设计的 4 位量化格式

- 使用分块归一化
- 动态范围更适合 LLM 权重分布

#### 2.2 双量化

QLoRA 使用双量化进一步减少显存:

1. **权重量化**: 将模型权重从 FP16 量化为 NF4
2. **量化常数**: 对量化常数再次量化

#### 2.3 QLoRA 流程

```
1. 加载预训练模型 (NF4 量化)
2. LoRA 训练:
   - 量化权重用于前向传播 (需要时反量化)
   - LoRA 参数使用 FP16/BF16 训练
3. 合并权重: W = W_0 + BA
```

### 3. 混合精度训练

#### 3.1 精度类型

- **FP32**: 32 位浮点
- **FP16**: 16 位浮点
- **BF16**: 16 位浮点 (更宽的动态范围)
- **INT8/INT4**: 整数量化

#### 3.2 混合精度策略

```python
# 前向传播: 使用低精度加速
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)

# 损失计算和反向传播: 使用高精度
loss = loss_fn(output, target)
loss.backward()

# 参数更新: 优化器状态使用高精度
optimizer.step()
```

---

## 代码实现

### 项目结构

```
model/
├── lora.py        # LoRA 实现
└── quant.py       # 量化工具

training/
└── qlora_trainer.py # QLoRA 训练器
```

---

## 实践练习

### 练习 1: 实现 LoRA 层

打开 `model/lora.py`，实现 LoRA 模块:

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r: int = 8, lora_alpha: float = 16, lora_dropout: float = 0.0):
        """
        LoRA 层的实现。

        对于输入 x:
        output = W_0 @ x + (B @ A) @ x

        Args:
            in_features: 输入维度
            out_features: 输出维度
            r: 低秩维度 (rank)
            lora_alpha: 缩放因子，通常设为 r 的倍数
            lora_dropout: Dropout 概率
        """
        # 实现:
        # 1. 冻结原始权重 W_0
        # 2. 初始化可训练的低秩矩阵 A 和 B
        # A 使用随机初始化，B 初始化为零 (训练初期不影响模型)
        pass

    def forward(self, x):
        """
        前向传播。

        实现:
        output = W_0 @ x + (lora_alpha / r) * (B @ A) @ x
        """
        pass

    def merge(self):
        """
        将 LoRA 权重合并到原始权重。
        """
        pass
```

### 练习 2: 在模型中应用 LoRA

```python
def get_lora_model(model, r: int = 8, lora_alpha: float = 16, lora_dropout: float = 0.0, target_modules=None):
    """
    为模型添加 LoRA。

    通常对以下模块应用 LoRA:
    - q_proj, k_proj, v_proj (注意力模块)
    - o_proj (输出投影)
    - gate_proj, up_proj, down_proj (FFN 模块)

    Args:
        model: 预训练模型
        r: LoRA 秩
        lora_alpha: 缩放参数
        lora_dropout: Dropout 概率
        target_modules: 要应用 LoRA 的模块名列表

    流程:
    1. 遍历模型的所有模块
    2. 找到目标模块 (如 Linear)
    3. 替换为 LoRA 版本
    """
    pass
```

### 练习 3: 实现 QLoRA 量化

```python
def quantize_tensor(x, num_bits=4):
    """
    将张量量化为低比特表示。

    NF4 量化步骤:
    1. 将权重分块 (通常每块 64 个元素)
    2. 计算每块的最小值和最大值
    3. 使用量化代码本映射到 num_bits 位

    Args:
        x: 输入张量
        num_bits: 量化位数 (4, 8 等)

    返回:
        量化后的张量和量化常数
    """
    pass


def dequantize_tensor(x_quant, scale, zero_point):
    """
    从量化表示恢复浮点张量。

    Args:
        x_quant: 量化张量
        scale: 量化缩放因子
        zero_point: 零点

    返回:
        恢复的浮点张量
    """
    pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage05_advanced/lesson14_lora_qlora/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage05_advanced/lesson14_lora_qlora/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **LoRA 论文**: "LoRA: Low-Rank Adaptation of Large Language Models"
- **QLoRA 论文**: "QLoRA: Efficient Finetuning of Quantized LLMs"
- **PEFT 库**: 了解 Hugging Face PEFT 库中的 LoRA 实现
