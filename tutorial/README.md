# Intro2LLM 教程总览

欢迎来到 Intro2LLM 完整教程！本教程采用**循序渐进的教学方式**，将大语言模型(LLM)的实现划分为**6大阶段、12个课时**，帮助学习者从零开始构建完整的LLM训练 pipeline。

## 项目特色

### 文档优先教学风格

本项目采用独特的**注释即实现**教学理念：

- **函数体极简**：仅使用 `pass` 或 `return ...`，不包含实际执行代码
- **注释即算法**：通过详细的中文注释描述完整实现步骤
- **学生自主实现**：学习者根据自然语言描述自行编写 PyTorch 代码
- **双重测试验证**：每个课时包含 `basic` 和 `advanced` 两层测试用例

示例风格：
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm 前向传播
    Args:
        x: [batch, seq_len, hidden_size]
    Returns:
        normalized: [batch, seq_len, hidden_size]
    """
    # Step 1: 转换为float32以保证数值稳定性
    # Step 2: 计算均方值 (mean of x^2)
    #    variance = x.pow(2).mean(dim=-1, keepdim=True)
    #    Shape: [batch, seq_len, 1]
    # Step 3: 计算RMS并应用归一化
    #    Formula: output = x / sqrt(variance + eps) * self.weight
    # Step 4: 转回原始数据类型
    pass
```

## 学习路径

### 路径一：最小可行路径 (MVP)
适合快速了解LLM全貌的学习者：
```
课时1 → 课时2 → 课时3 → 课时4 → 课时5 → 课时7 → 课时10
```

### 路径二：标准完整路径
按顺序完成全部12课时，深入理解每个组件。

### 路径三：研究导向路径
重点学习对齐训练与高级主题：
```
课时8 (DPO) → 课时9 (GRPO/PPO) → 课时11 (LoRA/FlashAttention)
```

## 阶段划分与课时安排

### 第一阶段：基础组件 (Foundation Components)

#### 课时1：项目概述、配置系统与分词器
- **目标文件**: `model/config.py`, `tokenizer/bpe_tokenizer.py`, `tokenizer/byte_level_tokenizer.py`
- **核心内容**:
  - Transformer Decoder-only 架构概览与训练流程
  - ModelConfig dataclass 设计：vocab_size, hidden_size, num_layers, num_heads等
  - GQA配置原理与架构开关选项
  - 子词分词原理与BPE算法
- **[进入学习](stage01_foundation/lesson01_config_tokenizer/README.md)**

#### 课时2：归一化层与嵌入层
- **目标文件**: `model/norm.py`, `model/embedding.py`
- **核心内容**:
  - LayerNorm: `y = (x - mean) / sqrt(variance + eps) * gamma + beta`
  - RMSNorm: `y = x / sqrt(mean(x^2) + eps) * weight`
  - Token Embedding: 从离散token到连续向量
  - RoPE旋转位置编码：旋转矩阵计算、频率维度配对、外推性
- **[进入学习](stage01_foundation/lesson02_norm_embedding/README.md)**

#### 课时3：注意力机制 - MHA、GQA与RoPE集成
- **目标文件**: `model/attention.py`
- **核心内容**:
  - Scaled Dot-Product Attention: `Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V`
  - Multi-Head Attention并行计算与注意力头划分
  - 因果掩码(Causal Mask)实现：上三角矩阵置-inf
  - GQA: 多个Query头共享KV头，KV缓存优化
- **[进入学习](stage01_foundation/lesson03_attention/README.md)**

---

### 第二阶段：模型组装 (Model Assembly)

#### 课时4：前馈网络与Transformer块
- **目标文件**: `model/feedforward.py`, `model/transformer_block.py`
- **核心内容**:
  - FFN结构：扩展维度→激活→投影
  - SwiGLU: `SwiGLU(x) = Swish(xW) ⊗ (xV)`
  - GeGLU: `GeGLU(x) = GELU(xW) ⊗ (xV)`
  - Pre-LN结构: `x + Sublayer(Norm(x))`
  - 残差连接与梯度流动、Dropout正则化
- **[进入学习](stage02_model_assembly/lesson04_ffn_transformer/README.md)**

#### 课时5：完整因果语言模型与生成策略
- **目标文件**: `model/causal_lm.py`
- **核心内容**:
  - Causal Language Model架构组装
  - KV缓存机制优化推理
  - 文本生成策略：Greedy、Temperature、Top-k、Top-p(Nucleus)
  - 生成终止条件与重复惩罚
- **[进入学习](stage02_model_assembly/lesson05_causal_lm/README.md)**

---

### 第三阶段：数据处理 (Data Processing)

#### 课时6：数据集与数据清洗
- **目标文件**: `data/pretrain_dataset.py`, `data/sft_dataset.py`, `data/dpo_dataset.py`, `data/filtering.py`
- **核心内容**:
  - 预训练数据集：因果语言建模、动态填充、Attention Mask
  - SFT数据集：instruction-input-output格式、Prompt掩码(-100)
  - DPO偏好数据集：prompt+chosen+rejected格式
  - 数据清洗：长度过滤、重复过滤、MinHash去重、质量评分
- **[进入学习](stage03_data_processing/lesson06_datasets/README.md)**

---

### 第四阶段：优化基础 (Optimization Fundamentals)

#### 课时7：损失函数与优化器
- **目标文件**: `loss/cross_entropy.py`, `loss/dpo_loss.py`, `optimizer/adamw.py`, `optimizer/lion.py`, `optimizer/scheduler.py`
- **核心内容**:
  - 交叉熵损失: `L = -sum(y_true * log(y_pred))`
  - AdamW优化器：一阶/二阶矩估计、解耦权重衰减
  - Lion优化器：符号动量更新
  - 学习率调度：Warmup + Cosine Decay
  - DPO损失初步：Bradley-Terry模型与对比损失概念
- **[进入学习](stage04_optimization/lesson07_loss_optimizer/README.md)**

---

### 第五阶段：对齐训练 (Alignment Training)

#### 课时8：DPO直接偏好优化
- **目标文件**: `loss/dpo_loss.py`, `training/dpo_trainer.py`
- **核心内容**:
  - RLHF流程回顾与DPO核心思想：无需显式奖励模型
  - Bradley-Terry模型详解
  - DPO损失完整推导：隐式奖励 `r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))`
  - Reference Model管理与Beta参数调优
- **[进入学习](stage05_alignment/lesson08_dpo/README.md)**

#### 课时9：GRPO与PPO强化学习
- **目标文件**: `loss/grpo_loss.py`, `loss/ppo_loss.py`, `training/grpo_trainer.py`, `training/ppo_trainer.py`
- **核心内容**:
  - GRPO：组内相对优势估计、无需Value Network、KL散度约束
  - PPO：Actor-Critic架构详解
  - 广义优势估计(GAE)：`A_t = sum((gamma*lambda)^l * delta_{t+l})`
  - Clipped Surrogate Objective
- **[进入学习](stage05_alignment/lesson09_grpo_ppo/README.md)**

---

### 第六阶段：训练体系与高级主题 (Training & Advanced)

#### 课时10：训练器体系实现
- **目标文件**: `training/*.py`, `utils/mixed_precision.py`
- **核心内容**:
  - 基础Trainer抽象类：训练循环结构 forward → backward → step
  - 梯度累积、混合精度训练(FP16/BF16)、梯度裁剪
  - 检查点保存与恢复、日志记录与监控
  - PretrainTrainer、SFTTrainer、DPOTrainer等完整实现
- **[进入学习](stage06_training_advanced/lesson10_trainers/README.md)**

#### 课时11：参数高效微调与性能优化
- **目标文件**: `model/lora.py`, `model/qlora.py`, `utils/flash_attention.py`
- **核心内容**:
  - LoRA: `h = Wx + BAx`、低秩分解、秩选择与缩放因子alpha
  - QLoRA: 4-bit NormalFloat量化、双量化、分页优化器
  - Flash Attention: Tiling分块计算、Online Softmax、IO-Aware算法
- **[进入学习](stage06_training_advanced/lesson11_lora_flash/README.md)**

#### 课时12：模型评估与部署
- **目标文件**: `evaluation/*.py`, `utils/checkpoint.py`, `utils/compute.py`
- **核心内容**:
  - 困惑度PPL计算：`PPL = exp(-1/N * sum(log P(x_i|x_<i)))`
  - MMLU评估：多项选择题格式、学科覆盖、提示词模板
  - HumanEval：代码生成评估、Pass@k指标
  - 检查点管理：保存、加载、模型序列化
  - FLOPs与显存估算
- **[进入学习](stage06_training_advanced/lesson12_evaluation/README.md)**

## 依赖关系图

```
课时1 (Config + Tokenizer)
└── 课时2 (Norm + Embedding)
    └── 课时3 (Attention)
        └── 课时4 (FFN + TransformerBlock)
            └── 课时5 (CausalLM)
                ├── 课时6 (Datasets)
                │   └── 课时7 (Loss + Optimizer)
                │       ├── 课时8 (DPO)
                │       │   └── 课时9 (GRPO + PPO)
                │       │       └── 课时10 (Trainers) ← 整合所有训练组件
                │       └── ──────────────────────────┘
                └── 课时11 (LoRA + Flash Attention)
                    └── 课时12 (Evaluation)
```

## 快速开始

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定课时测试
pytest tutorial/stage01_foundation/lesson02_norm_embedding/testcases/

# 运行特定测试
pytest tutorial/stage01_foundation/lesson02_norm_embedding/testcases/basic_test.py -v
```

### 训练流程

```bash
# 1. 预训练
python scripts/train_pretrain.py --config configs/tiny_config.py

# 2. 监督微调
python scripts/train_sft.py --model_path outputs/pretrain/final_model

# 3. DPO对齐
python scripts/train_dpo.py --model_path outputs/sft/final_model
```

## 项目结构

```
tutorial/
├── README.md                          # 本文件
├── stage01_foundation/                # 第一阶段：基础组件
│   ├── lesson01_config_tokenizer/
│   ├── lesson02_norm_embedding/
│   └── lesson03_attention/
├── stage02_model_assembly/            # 第二阶段：模型组装
│   ├── lesson04_ffn_transformer/
│   └── lesson05_causal_lm/
├── stage03_data_processing/           # 第三阶段：数据处理
│   └── lesson06_datasets/
├── stage04_optimization/              # 第四阶段：优化基础
│   └── lesson07_loss_optimizer/
├── stage05_alignment/                 # 第五阶段：对齐训练
│   ├── lesson08_dpo/
│   └── lesson09_grpo_ppo/
└── stage06_training_advanced/         # 第六阶段：训练与高级主题
    ├── lesson10_trainers/
    ├── lesson11_lora_flash/
    └── lesson12_evaluation/
```

## 推荐学习资源

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [LLaMA](https://arxiv.org/abs/2302.13971) - 开源LLM架构
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO算法
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - PPO算法

### 相关项目
- [minGPT](https://github.com/karpathy/minGPT) - Andrej Karpathy的简洁GPT实现
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 高效推理实现
- [transformers](https://github.com/huggingface/transformers) - HuggingFace Transformer库

---

Happy Learning! 祝学习愉快！
