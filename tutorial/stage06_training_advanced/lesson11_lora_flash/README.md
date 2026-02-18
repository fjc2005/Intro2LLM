# 课时11：参数高效微调与性能优化

## 学习目标

1. 深入理解LoRA低秩适配的原理与实现
2. 掌握QLoRA量化技术的核心思想
3. 理解Flash Attention的IO-Aware算法
4. 实现内存优化与计算效率权衡

---

## 1. LoRA (Low-Rank Adaptation)

### 1.1 核心思想

**论文**: [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)

```
问题: 全参数微调LLM成本高昂
    - 7B模型需要28GB显存存储参数
    - 优化器状态需要额外显存
    - 梯度计算耗时

解决方案: 低秩适配
    - 冻结预训练权重
    - 训练低秩分解矩阵
    - 大幅减少可训练参数量
```

### 1.2 LoRA数学原理

```
原始前向传播:
    h = W_0 * x

LoRA前向传播:
    h = W_0 * x + B * A * x

其中:
    W_0 ∈ R^{d×k}: 预训练权重 (冻结)
    A ∈ R^{r×k}: 可学习低秩矩阵 (随机初始化)
    B ∈ R^{d×r}: 可学习低秩矩阵 (零初始化)
    r << min(d, k): 秩 (通常8, 16, 32, 64)

参数量对比:
    全参数: d × k
    LoRA:  r × (d + k)
    节省:  (d × k) / (r × (d + k)) 倍
```

### 1.3 LoRA初始化策略

```
A矩阵: 随机高斯初始化
B矩阵: 零初始化

原因:
    - 初始时 B*A = 0
    - 初始输出与预训练模型相同
    - 训练从预训练能力开始渐进适应
```

### 1.4 缩放因子 (Alpha)

```
h = W_0 * x + (alpha / r) * B * A * x

default: alpha = 2r 或 r

作用:
    - 控制LoRA适应的强度
    - 与秩解耦，方便调参
```

---

## 2. QLoRA (Quantized LoRA)

### 1.1 核心思想

**论文**: [QLoRA: Efficient Finetuning of Quantized LLMs (2023)](https://arxiv.org/abs/2303.15693)

```
QLoRA = 4-bit量化 + LoRA + 分页优化器

特点:
    - 基模型用4-bit NormalFloat量化
    - LoRA参数保持16-bit或32-bit
    - 分页优化器处理显存峰值
```

### 1.2 4-bit NormalFloat (NF4)

```
NormalFloat:
    - 信息论最优的4-bit数据类型
    - 假设权重服从正态分布
    - 量化值根据正态分布分位数设置

双量化:
    - 量化常数本身也量化
    - 进一步减少显存占用
```

---

## 3. Flash Attention

### 1.1 核心问题

```
标准注意力的内存瓶颈:

HBM (High Bandwidth Memory) 容量大但速度慢
SRAM (Static RAM) 容量小但速度快 (A100: 192KB)

标准Attention:
    1. 从HBM加载Q, K
    2. 计算S = QK^T, 写入HBM
    3. 从HBM加载S, 计算P = softmax(S), 写入HBM
    4. 从HBM加载P, V, 计算O = PV, 写入HBM

问题: 多次HBM读写，IO成为瓶颈
```

### 1.2 Flash Attention算法

```
核心思想: Tiling + Online Softmax

步骤:
    1. 将Q, K, V分块(tile)到SRAM
    2. 在SRAM中完成所有计算
    3. 只将最终结果写回HBM

Online Softmax:
    - 避免存储完整的attention矩阵
    - 迭代更新softmax结果
    - 数值稳定的分块计算

复杂度:
    时间: O(N^2) (与标准attention相同)
    内存: O(N) (从O(N^2)降低)
```

### 1.3 Flash Attention优势

```
优势:
    - 内存高效: 无需存储N×N attention矩阵
    - IO感知: 减少HBM访问
    - 精确: 没有近似，输出与标准attention相同

局限:
    - 需要特定硬件支持
    - 序列越长优势越明显
```

---

## 4. 实现指引

### 4.1 model/lora.py

```python
class LoRALayer(nn.Module):
    """
    LoRA低秩适配层

    h = W_0 * x + (alpha/r) * B * A * x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        # Step 1: 保存配置
        # 保存秩r、alpha参数和缩放因子(alpha/r)

        # Step 2: 创建低秩矩阵
        # 创建A矩阵作为可学习参数，使用随机初始化，形状为[r, in_features]
        # 创建B矩阵作为可学习参数，使用零初始化，形状为[out_features, r]

        # Step 3: Dropout
        # 初始化dropout层用于正则化
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: 原始输出 (W_0 * x)
        # 将输入传入基础线性层，获取原始输出

        # Step 2: LoRA分支
        # 将输入与A矩阵转置相乘，再与B矩阵转置相乘
        # 将结果乘以缩放因子

        # Step 3: 合并输出
        # 将原始输出与LoRA分支输出相加

        pass


class LinearWithLoRA(nn.Module):
    """包装器: 在nn.Linear上添加LoRA"""

    def __init__(self, base_layer: nn.Linear, r: int = 8, ...):
        # 保存base_layer
        # 创建LoRALayer
        pass

    def forward(self, x):
        # return base_layer(x) + lora_layer(x)
        pass
```

### 4.2 model/qlora.py

```python
class QLoRAModel:
    """
    QLoRA模型

    使用bitsandbytes进行4-bit量化 + LoRA
    """

    def __init__(self, model, quantization_config):
        # Step 1: 量化基础模型到4-bit
        # 调用量化函数将模型的基础权重压缩到4位精度

        # Step 2: 添加LoRA层
        # 遍历模型的所有命名模块
        # 如果模块是线性层，使用LinearWithLoRA包装器为其添加LoRA能力
        # 用包装后的模块替换原模块

        pass

    def prepare_for_training(self):
        # 冻结所有非LoRA参数
        # 只训练LoRA参数
        pass
```

### 4.3 utils/flash_attention.py

```python
def flash_attention_forward(
    q: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Flash Attention前向传播

    使用PyTorch 2.0+内置的scaled_dot_product_attention
    """
    # 实现思路:

    # Step 1: 检查支持
    # 检查PyTorch版本是否支持flash attention
    # PyTorch 2.0+提供了torch.nn.functional.scaled_dot_product_attention

    # Step 2: 调用高效实现
    # 使用F.scaled_dot_product_attention函数
    # 传入参数: q, k, v, causal_mask(如果需要因果)
    # 该函数会自动选择最优后端:
    #   - Flash Attention (最快)
    #   - Memory-Efficient Attention
    #   - 标准实现(回退方案)

    # 核心公式:
    #   output = softmax(QK^T / sqrt(d_k)) @ V
    #   但以分块方式计算，避免O(N^2)内存

    # Step 3: 回退处理
    # 如果当前环境不支持高效实现，手动实现标准attention
    pass
```

---

## 5. 常见陷阱与注意事项

1. **LoRA秩选择**: 小任务r=8，大任务r=64，不是越大越好
2. **目标模块**: 通常只对Attention的Q、V投影添加LoRA
3. **QLoRA内存**: 虽然参数少，但激活值仍占显存
4. **Flash Attention兼容性**: 需要PyTorch 2.0+或flash-attn库
5. **梯度累积**: LoRA训练时仍需注意batch size

---

## 6. 课后练习

1. 计算LoRA的参数量节省比例
2. 对比LoRA不同秩的效果
3. 在消费级GPU上用QLoRA微调模型
4. 测量Flash Attention的加速比
