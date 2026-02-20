# Intro2LLM 完整课程预览报告

## 项目概述

**Intro2LLM** 是一个从零实现大型语言模型的教育框架，系统覆盖从文本表示到对齐优化的完整 LLM 技术栈。课程分为 5 个阶段，共 15 课时，每课时包含：
- `README.md`：理论讲解与实现指导
- `testcases/basic_test.py` 和 `testcases/advanced_test.py`：行为契约测试

---

## 课程总览

| 阶段 | 课时 | 主题 | 核心能力 |
|------|------|------|----------|
| Stage 1: 基础 | 01–05 | 分词/嵌入/注意力/FFN/KV缓存/配置分析 | 构建完整 Transformer 推理管线 |
| Stage 2: 预训练 | 06–09 | 数据/损失/优化器/训练器/评估 | 搭建端到端预训练系统 |
| Stage 3: 指令跟随 | 10 | SFT 监督微调 | 使模型遵循指令 |
| Stage 4: 对齐 | 11–13 | DPO / PPO+GAE / GRPO | 将模型与人类偏好对齐 |
| Stage 5: 高级技术 | 14–15 | LoRA/QLoRA / 高效注意力 | 参数高效微调与推理加速 |

---

## STAGE 1: 基础架构（Lessons 01–05）

### Lesson 01 — Tokenizer 与 Embedding（文本表示基础）

**文件路径**: `tutorial/stage01_foundation/lesson01_tokenizer_embedding/`

**学习目标**:
- 掌握 BPE 和 Byte-Level BPE 分词算法
- 理解 Token Embedding 原理（离散 ID → 连续向量）
- 实现 Sinusoidal PE 和 RoPE 位置编码

**核心概念**:
- **BaseTokenizer 接口**: 批处理、save/load、特殊 token 管理的统一契约
- **BPE 算法**: 字符级合并规则，贪心合并策略
- **Byte-Level BPE**: UTF-8 字节序列 + 可逆字节到 Unicode 映射，覆盖所有语言
- **RoPE**: 通过 2D 向量旋转建模相对位置，现代 LLM 主流方案

**实现任务**:
- `tokenizer/base_tokenizer.py`: `encode_batch`, `save/load`, `vocab_size`
- `tokenizer/bpe_tokenizer.py`: `train`, `encode`, `decode`, `_pretokenize`
- `tokenizer/byte_level_tokenizer.py`: 字节-Unicode 双向映射
- `model/embedding.py`: TokenEmbedding, PE 构建与应用, RoPE 旋转

**关键设计**: 信息四级变换 → 符号 → 离散 ID → 密集向量 → 位置感知向量

---

### Lesson 02 — 归一化层与注意力机制（Transformer 核心）

**文件路径**: `tutorial/stage01_foundation/lesson02_normalization_attention/`

**学习目标**:
- 对比 LayerNorm vs RMSNorm 对训练稳定性的作用
- 掌握缩放点积注意力数学
- 实现带因果掩码的多头注意力

**核心概念**:
- **LayerNorm**: `(x - E[x]) / sqrt(Var + ε) ⊗ γ + β`，含 weight 和 bias
- **RMSNorm**: `x / RMS(x) ⊗ γ`，无 bias，计算快约 30%，现代 LLM 首选（LLaMA/Qwen/Mistral）
- **缩放点积注意力**: `Attention(Q,K,V) = softmax(QKᵀ/√dk)V`，√dk 防止梯度消失
- **Pre-LN vs Post-LN**: Pre-LN（Norm→子层→Add）梯度流更好，无需 warmup

**实现任务**:
- `model/norm.py`: LayerNorm, RMSNorm 的 `__init__` 和 `forward`
- `model/attention.py`: MultiHeadAttention，包含 Q/K/V 投影、注意力分数、因果掩码

---

### Lesson 03 — 前馈网络与 Transformer 块（非线性表达能力）

**文件路径**: `tutorial/stage01_foundation/lesson03_ffn_transformer_block/`

**学习目标**:
- 对比 BasicFFN / GeGLU / SwiGLU 架构差异
- 理解门控机制的信息流控制
- 实现完整 Transformer Block

**核心概念**:
- **BasicFFN**: `ReLU(xW₁)W₂`，简单双层
- **GeGLU**: `GELU(xW_gate) ⊙ (xW_up)`，GELU 激活门控
- **SwiGLU**: `SiLU(xW_gate) ⊙ (xW_up)`，SiLU = x·sigmoid(x)，现代 LLM 首选
- **参数量对比**: BasicFFN ~8d²，GLU 变体 ~12d²（多 50%，但效果更好）
- **残差连接**: Pre-LN 结构 → `Norm→Attention→Add` 和 `Norm→FFN→Add`

**实现任务**:
- `model/feedforward.py`: BasicFFN, FeedForward (GeGLU), SwiGLU
- `model/transformer_block.py`: 完整 Pre-LN Transformer Block 与残差连接

---

### Lesson 04 — KV 缓存与因果语言模型（推理优化）

**文件路径**: `tutorial/stage01_foundation/lesson04_kv_cache_causal_lm/`

**学习目标**:
- 理解自回归生成机制
- 掌握 KV Cache 优化原理
- 实现贪心/温度/top-k/top-p 采样策略

**核心概念**:
- **因果 LM**: 每个位置的预测仅依赖之前的 token，自监督训练目标：预测下一个 token
- **KV Cache**: 缓存并复用历史 K,V，避免每步重新计算，以内存换计算
- **采样策略**:
  - 贪心: 选最高概率 token
  - 温度: 缩放 logits，T越高分布越均匀
  - Top-k: 仅考虑最高 k 个概率的 token
  - Top-p (Nucleus): 选累积概率达 p 的 token 集合

**实现任务**:
- `model/causal_lm.py`: `__init__`, `forward`（含因果掩码）, `generate` 方法
- KV cache 处理逻辑和文本生成循环

---

### Lesson 05 — 模型配置与效率分析（参数量和计算复杂度）

**文件路径**: `tutorial/stage01_foundation/lesson05_model_config/`

**学习目标**:
- 理解 ModelConfig 所有参数
- 计算 FLOPs（浮点运算量）
- 估算 GPU 显存需求

**核心概念**:
- **配置参数**: vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, rope_theta, use_rms_norm/rope/swiglu
- **参数量估算**:
  - Embedding: `vocab_size × hidden_size`
  - Attention (每层): MHA ~3×hidden_size²，GQA更少
  - FFN (每层 GLU): `2 × hidden_size × intermediate_size`
- **FLOPs 估算**:
  - 注意力: `4×hidden_size²×num_layers` per token
  - FFN: `8×hidden_size×intermediate_size×num_layers` per token
  - 训练 ≈ 3× 前向 FLOPs
- **显存估算**: 模型参数 + KV cache + 激活值

**关键配置**: `configs/tiny_config.py`（约 1000 万参数）用于快速测试

---

## STAGE 2: 预训练（Lessons 06–09）

### Lesson 06 — 数据处理与损失函数（预训练基础）

**文件路径**: `tutorial/stage02_pretrain/lesson06_data_loss/`

**学习目标**:
- 实现数据过滤和预处理流水线
- 掌握交叉熵损失数学原理
- 理解数据质量 vs 数量的权衡

**核心概念**:
- **数据预处理**:
  - 去重（精确/近似）防止过拟合
  - 质量过滤：移除特殊字符、重复内容、语言检测
  - 长度过滤：截断/填充
  - 统计过滤：基于困惑度的异常值检测
- **交叉熵损失**: `L_t = -log softmax(z_t)[y_t]`，使用 log-softmax 数值稳定
- **困惑度关系**: `PPL = exp(average_loss)`，期望分支因子

**实现任务**:
- `data/filtering.py`: LengthFilter, QualityFilter 类
- `loss/cross_entropy.py`: CrossEntropyLoss（含 ignore_index=-100 掩码）

---

### Lesson 07 — 优化器与学习率调度（训练动力）

**文件路径**: `tutorial/stage02_pretrain/lesson07_optimizer_scheduler/`

**学习目标**:
- 理解 AdamW 和 Lion 优化器数学原理
- 掌握 Warmup + 余弦退火学习率调度

**核心概念**:
- **AdamW**: 动量 `v_t`、方差缩放 `s_t`、偏差修正、解耦权重衰减（区别于 L2 正则）
- **Lion**: 只维护一个状态 `v_t`，参数更新 `θ = θ - η·sign(v_t)`，内存更小
- **学习率调度**:
  - 线性 Warmup: `η_t = η_max × (t/warmup_steps)`
  - 余弦退火: 平滑从 η_max 衰减到 η_min
  - 典型配置: 预训练 LR 1e-4，微调 1e-5，warmup 5–10% 总步数

**实现任务**:
- `optimizer/adamw.py`: AdamW 参数更新逻辑
- `optimizer/scheduler.py`: WarmupCosineScheduler

---

### Lesson 08 — 检查点、训练器与技巧（训练工程）

**文件路径**: `tutorial/stage02_pretrain/lesson08_trainer/`

**学习目标**:
- 实现检查点保存/加载以应对故障恢复
- 掌握梯度累积和梯度裁剪
- 构建完整 Trainer 类

**核心概念**:
- **检查点**: 保存模型/优化器/调度器/RNG 状态，限制历史数量管理磁盘
- **梯度累积**: 模拟更大 batch（`loss/accumulation_steps`），多步后才更新参数
- **梯度裁剪**: 防止梯度爆炸，基于范数 `g = g × (max_norm/||g||)`，Transformer 通常 clip=1.0
- **WandB 集成**: 追踪 loss 曲线、LR、梯度统计、硬件使用率

**实现任务**:
- `training/trainer.py`: Trainer 类，`save_checkpoint`, `load_checkpoint`
- `train_step` 含梯度累积与裁剪
- TrainerLogger 指标追踪

---

### Lesson 09 — 模型评估（质量评测）

**文件路径**: `tutorial/stage02_pretrain/lesson09_evaluation/`

**学习目标**:
- 计算困惑度（语言建模评估）
- 实现 MMLU 基准测试（多任务理解）
- 实现 HumanEval 基准测试（代码生成）

**核心概念**:
- **困惑度**: `PPL = exp(-1/T × ∑_t log P(x_t|x_{<t}))`，越低越好
- **MMLU**: 57 类任务（数学/科学/社会/历史/法律），4 选 1，评估准确率，支持 few-shot
- **HumanEval**: 164 个编程题，Pass@K 指标（K 次采样中至少 1 次正确），需代码执行沙箱

**实现任务**:
- `evaluation/perplexity.py`: PPL 计算
- `evaluation/mmlu.py`: Few-shot MMLU 评估
- `evaluation/humaneval.py`: 代码生成与 Pass@K 评估

---

## STAGE 3: 指令跟随（Lesson 10）

### Lesson 10 — 监督微调 SFT（指令跟随）

**文件路径**: `tutorial/stage03_instruction_follow/lesson10_sft/`

**学习目标**:
- 理解 SFT 与预训练的区别
- 掌握损失掩码（区分指令/回复）
- 实现 SFT 数据集与专用训练器

**核心概念**:
- **SFT 目标**: 预训练赋予通用语言理解，SFT 教模型遵循指令
- **数据格式**: `(instruction, response)` 对，Chat 格式（user/assistant 角色）
- **损失掩码**: 仅对回复 token 计算损失，指令 token 设 label=-100，避免浪费梯度
- **训练配置**: LR 1e-5 ~ 5e-6，epoch 2–3，更高 dropout 防止过拟合
- **Chat 模板**: Llama 风格 `[INST]..[/INST]`，模板一致性对多轮对话至关重要

**实现任务**:
- `data/sft_dataset.py`: SFTDataset（含损失掩码生成）
- `training/sft_trainer.py`: SFT 专用训练循环

---

## STAGE 4: 对齐（Lessons 11–13）

### Lesson 11 — DPO 与 IPO（直接偏好优化）

**文件路径**: `tutorial/stage04_alignment/lesson11_dpo/`

**学习目标**:
- 理解 DPO 数学原理（消除显式奖励模型）
- 实现 DPO 损失函数
- 了解 IPO 改进

**核心概念**:
- **RLHF 痛点**: 传统流水线（预训练→SFT→奖励模型→PPO）复杂且计算昂贵
- **DPO 核心**: 直接使用偏好数据，无需显式 RM 或 RL 算法
- **Bradley-Terry 模型**: 偏好概率通过 sigmoid 函数表达
- **DPO 损失**: `L_DPO = -log σ(Δf)`，其中 `Δf = f(x, y_w) - f(x, y_l)`
- **IPO 改进**: 使用平方差目标，对噪声偏好数据更鲁棒

**实现任务**:
- `data/dpo_dataset.py`: DPODataset（prompt/chosen/rejected 格式）
- `loss/dpo_loss.py`: DPOLoss（含 beta 温度参数和标签平滑）
- `training/dpo_trainer.py`: DPOTrainer（含冻结参考模型）

---

### Lesson 12 — PPO 与 GAE（近端策略优化）

**文件路径**: `tutorial/stage04_alignment/lesson12_ppo_gae/`

**学习目标**:
- 理解 PPO 算法原理
- 掌握广义优势估计（GAE）
- 实现 PPO 完整训练流程

**核心概念**:
- **PPO 目标**: 限制策略更新幅度的裁剪目标函数，`r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`
- **价值函数与优势**: `A(s,a) = Q(s,a) - V(s)`，TD 误差 `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- **GAE**: λ-return，`λ` 参数（0–1）控制偏差-方差权衡，典型值 0.95
- **KL 惩罚**: 防止偏离 SFT 模型过远
- **裁剪阈值 ε**: 通常 0.2

**实现任务**:
- `compute_gae()`: GAE 计算（反向迭代累积 TD 误差）
- `compute_ppo_loss()`: 重要性采样比率计算与裁剪
- `training/ppo_trainer.py`: PPOTrainer（含 `generate_responses`, `compute_rewards`, `training_step`）

---

### Lesson 13 — GRPO（群组相对偏好优化）

**文件路径**: `tutorial/stage04_alignment/lesson13_grpo/`

**学习目标**:
- 理解 GRPO 原理（DeepSeek 方案）
- 实现 GRPO 损失函数
- 对比 GRPO 与 DPO

**核心概念**:
- **GRPO 核心**: 每个 prompt 采样多个回复，按奖励排名，基于相对排名的加权损失
- **GRPO 损失**: `L_GRPO = -E[Σᵢ softmax(sᵢ) · log P(yᵢ|x)]`，高奖励回复获更高权重
- **GRPO vs DPO**:
  - GRPO: 无需参考模型，直接使用奖励，更稳定，VRAM更低
  - DPO: 需要参考模型，使用成对偏好数据，实现简单

**实现任务**:
- `data/grpo_dataset.py`: GRPODataset（每 prompt 多个回复）
- `compute_grpo_loss()`: Softmax 加权交叉熵
- `training/grpo_trainer.py`: GRPOTrainer（含 `generate_responses`, `compute_rewards`）

---

## STAGE 5: 高级技术（Lessons 14–15）

### Lesson 14 — LoRA 与 QLoRA（参数高效微调）

**文件路径**: `tutorial/stage05_advanced/lesson14_lora_qlora/`

**学习目标**:
- 理解 LoRA 原理与实现
- 掌握 QLoRA 量化技术
- 实现混合精度训练

**核心概念**:
- **LoRA 核心**: 在冻结权重上叠加低秩更新 `W = W₀ + BA`，其中 `B∈ℝ^{d×r}`, `A∈ℝ^{r×k}`, `r ≪ min(d,k)`
- **前向传播**: `h = W₀x + BAx`，低维计算分离
- **参数压缩示例**: 4096×4096 矩阵 → rank-8 LoRA → 250x 压缩（1600万→6.5万参数）
- **NF4 量化**: 专为 LLM 权重分布优化的 4-bit 归一化浮点格式
- **QLoRA 流水线**: 加载量化模型 → FP16/BF16 训练 LoRA 参数 → 推理时合并
- **精度类型**: FP32, FP16, BF16（更大动态范围）, INT8/INT4

**实现任务**:
- `model/lora.py`: LoRALayer（含冻结原始权重、可训练 A/B 矩阵、`merge()` 方法）
- `get_lora_model()`: 将 LoRA 应用到目标模块（q/k/v/o_proj + FFN 层）
- `quantize_tensor()` / `dequantize_tensor()`: 基于块的 NF4 量化/反量化

---

### Lesson 15 — 高效注意力机制

**文件路径**: `tutorial/stage05_advanced/lesson15_efficient_attention/`

**学习目标**:
- 理解 Flash Attention 原理
- 掌握 MQA 和 GQA 技术
- 了解各类注意力优化方法

**核心概念**:
- **标准注意力瓶颈**: `O(N²d)` 计算，`O(N²)` 内存（注意力矩阵存储）
- **Flash Attention**: IO 感知算法，分块计算避免存储完整 NxN 矩阵，在线 Softmax（2-pass→1-pass），内存 `O(N²)→O(N)`
- **MQA（多查询注意力）**: 所有 Q 头共享单一 K/V，大幅减少 KV cache，但可能影响质量
- **GQA（分组查询注意力）**: Q 头分组，每组共享 K/V，`num_groups = num_heads/num_kv_heads`，在质量与效率之间取得平衡（Llama 3/Qwen 使用）
- **其他优化**: 稀疏注意力（局部窗口/随机/块稀疏）、线性注意力（核函数近似）、滑动窗口注意力

**实现任务**:
- `model/attention.py`: GroupedQueryAttention（含 `repeat_kv()` 扩展 KV 头数）
- GQA vs MHA 内存权衡分析
- 滑动窗口注意力掩码实现

---

## 核心框架组件总结

### 实现依赖关系

```
tokenizer/          → Stage 1 Lesson 01
  base_tokenizer.py
  bpe_tokenizer.py
  byte_level_tokenizer.py

model/              → Stage 1 Lessons 01–05
  embedding.py      (TokenEmbedding, PE, RoPE)
  norm.py           (LayerNorm, RMSNorm)
  attention.py      (MultiHeadAttention, GroupedQueryAttention)
  feedforward.py    (BasicFFN, GeGLU, SwiGLU)
  transformer_block.py
  causal_lm.py
  lora.py / qlora.py

data/               → Stages 2–4
  filtering.py
  sft_dataset.py
  dpo_dataset.py
  grpo_dataset.py

loss/               → Stages 2–4
  cross_entropy.py
  dpo_loss.py

optimizer/          → Stage 2
  adamw.py
  scheduler.py

training/           → Stages 2–4
  trainer.py
  pretrain_trainer.py
  sft_trainer.py
  dpo_trainer.py
  ppo_trainer.py
  grpo_trainer.py

evaluation/         → Stage 2
  perplexity.py
  mmlu.py
  humaneval.py
```

### 测试运行命令

```bash
# 运行全部测试
python -m pytest tests -v
python -m pytest tutorial/**/testcases -v

# 按阶段运行
python -m pytest tutorial/stage01_foundation/ -v
python -m pytest tutorial/stage02_pretrain/ -v
python -m pytest tutorial/stage03_instruction_follow/ -v
python -m pytest tutorial/stage04_alignment/ -v
python -m pytest tutorial/stage05_advanced/ -v

# 单课时测试
python -m pytest tutorial/stage01_foundation/lesson01_tokenizer_embedding/testcases -v
```

---

## 学习路径建议

1. **基础夯实（L01–L05）**: 确保每个组件的形状/数值契约通过测试，理解每个模块的数学原理
2. **训练系统（L06–L09）**: 关注数据质量对最终模型的影响，实验不同超参数配置
3. **指令对齐（L10）**: 理解损失掩码对指令跟随能力的关键作用
4. **偏好对齐（L11–L13）**: 逐步递进：DPO（最简单）→ GRPO（DeepSeek 方案）→ PPO（最复杂）
5. **效率优化（L14–L15）**: 应用 LoRA/GQA 到前面实现的模型，体验参数效率提升