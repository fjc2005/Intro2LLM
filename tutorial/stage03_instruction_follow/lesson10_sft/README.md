# L10: 监督微调 SFT

## 学习目标

1. **理解** 监督微调 (SFT) 的原理
2. **掌握** 损失掩码技术
3. **能够** 实现 SFT 数据集和训练器

---

## 理论背景

### 1. 监督微调 (SFT) 原理

#### 1.1 从预训练到微调

**预训练阶段**: 在大规模无标注数据上学习通用语言表示
- 目标: 预测下一个 token
- 数据: 互联网文本、书籍、代码等

**SFT 阶段**: 在有标注的指令数据上学习遵循指令
- 目标: 给定指令，生成正确的响应
- 数据: (指令, 响应) 对

#### 1.2 SFT 数据格式

典型的 SFT 数据包含:

```
{
    "instruction": "请写一首关于春天的诗",
    "input": "",
    "output": "春风拂面，万物复苏..."
}
```

或者使用聊天格式:

```
[
    {"role": "user", "content": "请写一首关于春天的诗"},
    {"role": "assistant", "content": "春风拂面，万物复苏..."}
]
```

#### 1.3 训练目标

SFT 本质上仍然是语言建模任务:
$$\mathcal{L}_{SFT} = -\sum_t \log P_{\theta}(y_t | \text{instruction}, y_{<t})$$

### 2. 损失掩码 (Loss Mask)

#### 2.1 目的

在 SFT 训练中，我们只希望模型学习如何生成响应 (response)，而不关心指令 (instruction) 部分的预测准确性。

#### 2.2 原理

对于序列:
```
[INST] 请写诗 [/INST] 春风拂面...
```

我们希望:
- 计算 response 部分的损失
- 忽略 instruction 部分的损失

#### 2.3 实现

```python
tokens:    [INST] 请 写 诗 [/INST] 春 风 拂 面 ...
mask:      [ 0   0   0   0      0    1   1   1  1  ...
```

其中:
- mask = 0: 不计算损失 (指令部分)
- mask = 1: 计算损失 (响应部分)

#### 2.4 位置编码的影响

SFT 数据通常需要特殊处理:
- **对话模板**: 使用特殊 token 标记角色 (如 [INST], [/INST])
- **连续性**: 确保模型学习在指令后开始生成

### 3. SFT 训练技巧

#### 3.1 数据质量

- 指令多样性: 覆盖各种任务类型
- 响应质量: 确保响应正确且格式良好
- 数据清洗: 过滤低质量数据

#### 3.2 训练策略

- **学习率**: 通常比预训练小 (如 1e-5 ~ 5e-6)
- **Epochs**: 通常 2-3 个 epoch
- **早停**: 监控验证损失，防止过拟合

---

## 代码实现

### 项目结构

```
data/
├── sft_dataset.py  # SFT 数据集
└── collate.py     # 批处理处理

training/
├── sft_trainer.py # SFT 训练器
```

---

## 实践练习

### 练习 1: 实现 SFT 数据集

打开 `data/sft_dataset.py`，实现 SFT 数据集:

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        """
        监督微调数据集。

        Args:
            data_path: SFT 数据文件路径 (JSONL 格式)
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        # 实现: 加载数据并预处理
        pass

    def __getitem__(self, idx):
        """
        获取单个样本。

        数据格式:
        {
            "instruction": "...",
            "input": "...",  # 可选
            "output": "..."
        }

        需要返回:
        - input_ids: 完整序列的 token IDs
        - attention_mask: 注意力掩码
        - labels: 用于计算损失的标签 (instruction 部分为 -100)
        """
        # 实现: 构建序列和应用损失掩码
        pass
```

### 练习 2: 实现损失掩码

```python
def create_sft_labels(input_ids, tokenizer, instruction_template):
    """
    为 SFT 创建带掩码的标签。

    目标:
    - instruction 部分: 标签为 -100 (不计算损失)
    - output 部分: 标签为实际的 token IDs

    Args:
        input_ids: 输入序列
        tokenizer: 分词器
        instruction_template: 指令模板

    实现思路:
    1. 找到 instruction 结束和 output 开始的位置
    2. 将 instruction 部分的标签设为 -100
    3. 保留 output 部分的标签
    """
    pass
```

### 练习 3: 实现 SFT 训练器

```python
class SFTTrainer(Trainer):
    def __init__(self, model, train_dataset, eval_dataset, config):
        """
        SFT 训练器。

        特点:
        - 使用带有损失掩码的标签
        - 可能需要特殊的对话模板
        """
        pass

    def compute_loss(self, batch):
        """
        计算 SFT 损失。

        实现:
        - 使用 labels 中的 -100 掩码
        - 只在 response 部分计算交叉熵损失
        """
        pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage03_instruction_follow/lesson10_sft/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage03_instruction_follow/lesson10_sft/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **SFT 论文**: 了解 InstructGPT、FLAN 等 SFT 方法
- **数据格式**: 了解 Alpaca、ShareGPT 等数据集格式
- **聊天模板**: 了解各类模型的对话模板差异
