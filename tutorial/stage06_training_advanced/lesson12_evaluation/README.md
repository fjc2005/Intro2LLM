# 课时12：模型评估与部署

## 学习目标

1. 掌握困惑度(Perplexity)的计算与理解
2. 理解MMLU评估方法与实现
3. 了解HumanEval代码生成评估
4. 掌握检查点管理与模型序列化
5. 理解FLOPs与显存估算方法

---

## 1. 困惑度 (Perplexity)

### 1.1 数学定义

```
困惑度衡量语言模型预测下一个token的能力

PPL = exp(-1/N * Σ log P(x_i | x_<i))

其中:
    N: token总数
    P(x_i | x_<i): 模型预测第i个token的概率

直观理解:
    - PPL = 100: 相当于从100个等概率选择中预测
    - PPL越低，模型预测能力越强
    - PPL = vocab_size: 与随机猜测相当
```

### 1.2 滑动窗口困惑度

```
问题: 长序列可能超出模型最大长度

解决方案: 滑动窗口
    PPL = exp(-1/N * Σ_{i=1}^{N} log P(x_i | x_{max(1,i-stride):i-1}))

其中stride是滑动步长

优点:
    - 可以评估任意长度文本
    - 考虑长程依赖
```

---

## 2. MMLU评估

### 2.1 MMLU简介

**MMLU (Massive Multitask Language Understanding)**:
```
- 57个学科的多项选择题
- 涵盖STEM、人文、社科等领域
- 测试知识和推理能力

格式:
    问题: "光合作用的产物是什么?"
    选项: A. 氧气 B. 二氧化碳 C. 氮气 D. 氢气
    答案: A
```

### 2.2 MMLU评估方法

```
Prompt构建:
    "以下是一道选择题，请选出正确答案。\n"
    "问题: {question}\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n"
    "答案: "

评估:
    1. 模型生成答案token (A, B, C, D)
    2. 或者直接比较选项的log概率
    3. 计算准确率
```

---

## 3. HumanEval评估

### 3.1 代码生成评估

```
HumanEval:
    - 164个编程问题
    - 函数签名 + docstring
    - 测试用例验证正确性

格式:
    def factorial(n):
        '''Return the factorial of n'''
        # 模型需要完成此函数

评估指标: Pass@k
    - 生成k个候选答案
    - 只要有1个通过测试即算正确
```

### 3.2 Pass@k计算

```
Pass@k = E[1 - C(n-c, k) / C(n, k)]

其中:
    n: 生成样本数 (通常n=200)
    c: 通过测试的样本数
    k: 评估参数 (通常k=1, 10, 100)
```

---

## 4. 检查点管理

### 4.1 保存检查点

```python
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    loss,
    output_dir,
):
    """
    保存训练检查点

    检查点内容:
    - epoch: 当前训练轮数
    - step: 当前训练步数
    - model_state_dict: 模型参数
    - optimizer_state_dict: 优化器状态(动量、学习率等)
    - scheduler_state_dict: 调度器状态
    - loss: 当前损失值

    实现思路:
    Step 1: 创建包含所有状态的字典
    Step 2: 构造保存路径 (如 outputs/checkpoint-1000)
    Step 3: 使用torch.save保存到文件
    Step 4: (可选)保存模型配置和tokenizer
    """
    pass
```

### 4.2 加载检查点

```python
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    加载训练检查点

    实现思路:
    Step 1: 使用torch.load加载检查点文件
    Step 2: 恢复模型参数
    Step 3: (可选)恢复优化器状态，继续训练
    Step 4: (可选)恢复调度器状态
    Step 5: 返回epoch、step、loss用于日志
    """
    pass
```

---

## 5. FLOPs与显存估算

### 5.1 FLOPs计算

```
Transformer FLOPs估算:

前向传播 (per layer):
    Attention: 4 * batch * seq_len^2 * hidden_size
    FFN: 2 * batch * seq_len * hidden_size * intermediate_size

总FLOPs:
    FLOPs ≈ num_layers * 2 * batch * seq_len * hidden_size
            * (4 * hidden_size + 2 * intermediate_size + seq_len)

简化估算 (1 token):
    FLOPs_per_token ≈ 2 * params
    (每个参数需要2次浮点运算: 乘和加)
```

### 5.2 显存估算

```
模型参数显存:
    FP32: 4 bytes/param
    FP16/BF16: 2 bytes/param
    INT8: 1 byte/param
    INT4: 0.5 bytes/param

优化器状态 (Adam):
    FP32: 8 bytes/param (一阶矩 + 二阶矩)
    8-bit: 2 bytes/param

激活值显存:
    与batch_size, seq_len, hidden_size成正比

总显存估算:
    训练: params * (dtype_size + optimizer_size) + activation_memory
    推理: params * dtype_size + kv_cache_memory
```

---

## 6. 实现指引

### 6.1 evaluation/perplexity.py

```python
def compute_perplexity(
    model,
    dataloader,
    device='cuda',
) -> float:
    """
    计算困惑度

    PPL = exp(-mean(log P(x_i)))
    """
    # Step 1: 累计损失
    # 初始化总损失为0
    # 初始化总token数为0

    # Step 2: 遍历数据
    # 对每个批次:
    #   获取模型输出的logits
    #   计算交叉熵损失，使用sum reduction累加所有token的损失
    #   将批次损失累加到总损失
    #   统计批次中非padding token的数量(标签不等于-100的位置)

    # Step 3: 计算PPL
    # 计算平均损失: 总损失除以总token数
    # 计算困惑度: 对平均损失取指数

    pass
```

### 6.2 evaluation/mmlu.py

```python
def evaluate_mmlu(
    model,
    tokenizer,
    data_path,
    num_few_shot=0,
) -> Dict[str, float]:
    """
    MMLU评估
    """
    # Step 1: 加载MMLU数据

    # Step 2: 对每个样本:
    #   构建prompt
    #   模型生成或计算选项概率
    #   比较预测答案和真实答案

    # Step 3: 按学科和总体统计准确率

    pass
```

### 6.3 utils/compute.py

```python
def estimate_flops(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> int:
    """
    估算FLOPs
    """
    # 基于配置计算前向传播FLOPs
    # 考虑Attention、FFN各部分
    pass

def estimate_memory(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
    dtype: str = 'fp16',
    training: bool = True,
) -> int:
    """
    估算显存需求 (bytes)
    """
    # 计算参数显存
    # 计算激活值显存
    # 计算优化器状态显存
    pass
```

---

## 7. 常见陷阱与注意事项

1. **PPL计算范围**: 确保排除padding位置
2. **MMLU few-shot**: 注意prompt构建的一致性
3. **检查点版本**: 加载时检查模型架构兼容性
4. **显存估算误差**: 实际显存使用可能因实现而异
5. **评估指标选择**: 根据任务选择合适的评估指标

---

## 8. 课后练习

1. 手动计算一个小模型的PPL
2. 实现MMLU评估的few-shot learning
3. 估算7B模型训练和推理的显存需求
4. 对比不同精度(FP32/FP16/INT8)的PPL差异
