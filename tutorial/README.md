# Intro2LLM Complete Course Preview Report

## Project Overview

**Intro2LLM** is an educational framework for implementing large language models from scratch, systematically covering the complete LLM technology stack from text representation to alignment optimization. The course is divided into 5 stages with a total of 15 lessons. Each lesson includes:
- `README.md`: Theoretical explanation and implementation guidance
- `testcases/basic_test.py` and `testcases/advanced_test.py`: Behavioral contract testing

---

## Course Overview

| Stage | Lessons | Topic | Core Capability |
|------|------|------|----------|
| Stage 1: Foundation | 01–05 | Tokenization/Embedding/Attention/FFN/KV Cache/Config Analysis | Build a complete Transformer inference pipeline |
| Stage 2: Pre-training | 06–09 | Data/Loss/Optimizer/Trainer/Evaluation | Build an end-to-end pre-training system |
| Stage 3: Instruction Following | 10 | SFT Supervised Fine-Tuning | Make the model follow instructions |
| Stage 4: Alignment | 11–13 | DPO / PPO+GAE / GRPO | Align the model with human preferences |
| Stage 5: Advanced Techniques | 14–15 | LoRA/QLoRA / Efficient Attention | Parameter-efficient fine-tuning and inference acceleration |

---

## STAGE 1: Foundation Architecture (Lessons 01–05)

### Lesson 01 — Tokenizer and Embedding (Text Representation Foundation)

**File Path**: `tutorial/stage01_foundation/lesson01_tokenizer_embedding/`

**Learning Objectives**:
- Master BPE and Byte-Level BPE tokenization algorithms
- Understand the principles of Token Embedding (Discrete IDs → Continuous vectors)
- Implement Sinusoidal PE and RoPE positional encodings

**Core Concepts**:
- **BaseTokenizer Interface**: Unified contract for batch processing, save/load, and special token management
- **BPE Algorithm**: Character-level merge rules, greedy merging strategy
- **Byte-Level BPE**: UTF-8 byte sequences + reversible byte-to-Unicode mapping, covering all languages
- **RoPE**: Modeling relative position via 2D vector rotation, the mainstream solution for modern LLMs

**Implementation Tasks**:
- `tokenizer/base_tokenizer.py`: `encode_batch`, `save/load`, `vocab_size`
- `tokenizer/bpe_tokenizer.py`: `train`, `encode`, `decode`, `_pretokenize`
- `tokenizer/byte_level_tokenizer.py`: Bidirectional byte-Unicode mapping
- `model/embedding.py`: TokenEmbedding, PE construction and application, RoPE rotation

**Key Design**: Four-level information transformation → Symbols → Discrete IDs → Dense vectors → Position-aware vectors

---

### Lesson 02 — Normalization Layers and Attention Mechanism (Transformer Core)

**File Path**: `tutorial/stage01_foundation/lesson02_normalization_attention/`

**Learning Objectives**:
- Compare the effects of LayerNorm vs. RMSNorm on training stability
- Master the mathematics of scaled dot-product attention
- Implement multi-head attention with causal masking

**Core Concepts**:
- **LayerNorm**: `(x - E[x]) / sqrt(Var + ε) ⊗ γ + β`, including weight and bias
- **RMSNorm**: `x / RMS(x) ⊗ γ`, no bias, ~30% faster computation, preferred by modern LLMs (LLaMA/Qwen/Mistral)
- **Scaled Dot-Product Attention**: `Attention(Q,K,V) = softmax(QKᵀ/√dk)V`, `√dk` prevents vanishing gradients
- **Pre-LN vs Post-LN**: Pre-LN (Norm→Sublayer→Add) has better gradient flow, no warmup required

**Implementation Tasks**:
- `model/norm.py`: LayerNorm, RMSNorm `__init__` and `forward` methods
- `model/attention.py`: MultiHeadAttention, including Q/K/V projections, attention scores, causal masking

---

### Lesson 03 — Feed-Forward Networks and Transformer Blocks (Non-linear Expressivity)

**File Path**: `tutorial/stage01_foundation/lesson03_ffn_transformer_block/`

**Learning Objectives**:
- Compare architectural differences between BasicFFN / GeGLU / SwiGLU
- Understand information flow control in gating mechanisms
- Implement a complete Transformer Block

**Core Concepts**:
- **BasicFFN**: `ReLU(xW₁)W₂`, simple two-layer
- **GeGLU**: `GELU(xW_gate) ⊙ (xW_up)`, GELU activation gating
- **SwiGLU**: `SiLU(xW_gate) ⊙ (xW_up)`, SiLU = x·sigmoid(x), preferred by modern LLMs
- **Parameter Count Comparison**: BasicFFN ~8d², GLU variants ~12d² (50% more, but better performance)
- **Residual Connections**: Pre-LN structure → `Norm→Attention→Add` and `Norm→FFN→Add`

**Implementation Tasks**:
- `model/feedforward.py`: BasicFFN, FeedForward (GeGLU), SwiGLU
- `model/transformer_block.py`: Complete Pre-LN Transformer Block with residual connections

---

### Lesson 04 — KV Cache and Causal Language Models (Inference Optimization)

**File Path**: `tutorial/stage01_foundation/lesson04_kv_cache_causal_lm/`

**Learning Objectives**:
- Understand the autoregressive generation mechanism
- Master the principles of KV Cache optimization
- Implement greedy/temperature/top-k/top-p sampling strategies

**Core Concepts**:
- **Causal LM**: The prediction at each position depends only on previous tokens; self-supervised training objective: predict the next token
- **KV Cache**: Cache and reuse historical K, V to avoid recomputation at each step, trading memory for computation
- **Sampling Strategies**:
  - Greedy: Choose the token with the highest probability
  - Temperature: Scale logits; higher T leads to a more uniform distribution
  - Top-k: Only consider the top k tokens with the highest probabilities
  - Top-p (Nucleus): Choose from a set of tokens whose cumulative probability reaches p

**Implementation Tasks**:
- `model/causal_lm.py`: `__init__`, `forward` (including causal mask), `generate` methods
- KV cache processing logic and text generation loop

---

### Lesson 05 — Model Configuration and Efficiency Analysis (Parameter Count and Computational Complexity)

**File Path**: `tutorial/stage01_foundation/lesson05_model_config/`

**Learning Objectives**:
- Understand all parameters in ModelConfig
- Calculate FLOPs (Floating Point Operations)
- Estimate GPU VRAM requirements

**Core Concepts**:
- **Configuration Parameters**: vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, rope_theta, use_rms_norm/rope/swiglu
- **Parameter Count Estimation**:
  - Embedding: `vocab_size × hidden_size`
  - Attention (per layer): MHA ~3×hidden_size², GQA is less
  - FFN (per layer GLU): `2 × hidden_size × intermediate_size`
- **FLOPs Estimation**:
  - Attention: `4×hidden_size²×num_layers` per token
  - FFN: `8×hidden_size×intermediate_size×num_layers` per token
  - Training ≈ 3× Forward FLOPs
- **VRAM Estimation**: Model parameters + KV cache + activations

**Key Configuration**: `configs/tiny_config.py` (~10 million parameters) for quick testing

---

## STAGE 2: Pre-training (Lessons 06–09)

### Lesson 06 — Data Processing and Loss Functions (Pre-training Foundation)

**File Path**: `tutorial/stage02_pretrain/lesson06_data_loss/`

**Learning Objectives**:
- Implement data filtering and preprocessing pipelines
- Master the mathematics of cross-entropy loss
- Understand the trade-off between data quality vs. quantity

**Core Concepts**:
- **Data Preprocessing**:
  - Deduplication (exact/approximate) to prevent overfitting
  - Quality filtering: Remove special characters, repetitive content, language detection
  - Length filtering: Truncation/Padding
  - Statistical filtering: Perplexity-based outlier detection
- **Cross-Entropy Loss**: `L_t = -log softmax(z_t)[y_t]`, using log-softmax for numerical stability
- **Perplexity Relationship**: `PPL = exp(average_loss)`, expected branching factor

**Implementation Tasks**:
- `data/filtering.py`: LengthFilter, QualityFilter classes
- `loss/cross_entropy.py`: CrossEntropyLoss (including ignore_index=-100 masking)

---

### Lesson 07 — Optimizers and Learning Rate Scheduling (Training Dynamics)

**File Path**: `tutorial/stage02_pretrain/lesson07_optimizer_scheduler/`

**Learning Objectives**:
- Understand the mathematical principles of AdamW and Lion optimizers
- Master Warmup + Cosine Annealing learning rate scheduling

**Core Concepts**:
- **AdamW**: Momentum `v_t`, variance scaling `s_t`, bias correction, decoupled weight decay (distinct from L2 regularization)
- **Lion**: Only maintains one state `v_t`, parameter update `θ = θ - η·sign(v_t)`, smaller memory footprint
- **Learning Rate Scheduling**:
  - Linear Warmup: `η_t = η_max × (t/warmup_steps)`
  - Cosine Annealing: Smoothly decays from η_max to η_min
  - Typical configurations: Pre-training LR 1e-4, fine-tuning 1e-5, warmup 5–10% of total steps

**Implementation Tasks**:
- `optimizer/adamw.py`: AdamW parameter update logic
- `optimizer/scheduler.py`: WarmupCosineScheduler

---

### Lesson 08 — Checkpoints, Trainers, and Techniques (Training Engineering)

**File Path**: `tutorial/stage02_pretrain/lesson08_trainer/`

**Learning Objectives**:
- Implement checkpoint saving/loading for crash recovery
- Master gradient accumulation and gradient clipping
- Build a complete Trainer class

**Core Concepts**:
- **Checkpoints**: Save model/optimizer/scheduler/RNG states, limit the number of history checkpoints to manage disk space
- **Gradient Accumulation**: Simulate larger batches (`loss/accumulation_steps`), update parameters only after multiple steps
- **Gradient Clipping**: Prevent exploding gradients, norm-based `g = g × (max_norm/||g||)`, Transformers usually use clip=1.0
- **WandB Integration**: Track loss curves, LR, gradient statistics, hardware utilization

**Implementation Tasks**:
- `training/trainer.py`: Trainer class, `save_checkpoint`, `load_checkpoint`
- `train_step` including gradient accumulation and clipping
- TrainerLogger metric tracking

---

### Lesson 09 — Model Evaluation (Quality Benchmarking)

**File Path**: `tutorial/stage02_pretrain/lesson09_evaluation/`

**Learning Objectives**:
- Calculate perplexity (language modeling evaluation)
- Implement the MMLU benchmark (multi-task understanding)
- Implement the HumanEval benchmark (code generation)

**Core Concepts**:
- **Perplexity**: `PPL = exp(-1/T × ∑_t log P(x_t|x_{<t}))`, lower is better
- **MMLU**: 57 task categories (math/science/social/history/law), 4-choice multiple choice, evaluates accuracy, supports few-shot
- **HumanEval**: 164 programming problems, Pass@K metric (at least 1 correct out of K samples), requires a code execution sandbox

**Implementation Tasks**:
- `evaluation/perplexity.py`: PPL calculation
- `evaluation/mmlu.py`: Few-shot MMLU evaluation
- `evaluation/humaneval.py`: Code generation and Pass@K evaluation

---

## STAGE 3: Instruction Following (Lesson 10)

### Lesson 10 — Supervised Fine-Tuning SFT (Instruction Following)

**File Path**: `tutorial/stage03_instruction_follow/lesson10_sft/`

**Learning Objectives**:
- Understand the difference between SFT and pre-training
- Master loss masking (distinguishing instruction/response)
- Implement SFT datasets and dedicated trainers

**Core Concepts**:
- **SFT Objective**: Pre-training grants general language understanding, SFT teaches the model to follow instructions
- **Data Format**: `(instruction, response)` pairs, Chat format (user/assistant roles)
- **Loss Masking**: Calculate loss only on response tokens, set instruction tokens to label=-100 to avoid wasting gradients
- **Training Configuration**: LR 1e-5 ~ 5e-6, 2–3 epochs, higher dropout to prevent overfitting
- **Chat Templates**: Llama-style `[INST]..[/INST]`, template consistency is crucial for multi-turn conversations

**Implementation Tasks**:
- `data/sft_dataset.py`: SFTDataset (including loss mask generation)
- `training/sft_trainer.py`: Dedicated SFT training loop

---

## STAGE 4: Alignment (Lessons 11–13)

### Lesson 11 — DPO and IPO (Direct Preference Optimization)

**File Path**: `tutorial/stage04_alignment/lesson11_dpo/`

**Learning Objectives**:
- Understand DPO mathematical principles (eliminating explicit reward models)
- Implement the DPO loss function
- Understand IPO improvements

**Core Concepts**:
- **RLHF Pain Points**: The traditional pipeline (pre-training→SFT→reward model→PPO) is complex and computationally expensive
- **DPO Core**: Directly uses preference data, requiring no explicit RM or RL algorithms
- **Bradley-Terry Model**: Preference probabilities expressed via sigmoid functions
- **DPO Loss**: `L_DPO = -log σ(Δf)`, where `Δf = f(x, y_w) - f(x, y_l)`
- **IPO Improvement**: Uses a squared-error objective, more robust to noisy preference data

**Implementation Tasks**:
- `data/dpo_dataset.py`: DPODataset (prompt/chosen/rejected format)
- `loss/dpo_loss.py`: DPOLoss (including beta temperature parameter and label smoothing)
- `training/dpo_trainer.py`: DPOTrainer (including frozen reference model)

---

### Lesson 12 — PPO and GAE (Proximal Policy Optimization)

**File Path**: `tutorial/stage04_alignment/lesson12_ppo_gae/`

**Learning Objectives**:
- Understand PPO algorithm principles
- Master Generalized Advantage Estimation (GAE)
- Implement the complete PPO training pipeline

**Core Concepts**:
- **PPO Objective**: Clipped objective function to restrict policy update magnitude, `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`
- **Value Function and Advantage**: `A(s,a) = Q(s,a) - V(s)`, TD error `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- **GAE**: λ-return, parameter `λ` (0–1) controls the bias-variance trade-off, typical value 0.95
- **KL Penalty**: Prevents deviating too far from the SFT model
- **Clipping Threshold ε**: Typically 0.2

**Implementation Tasks**:
- `compute_gae()`: GAE computation (backward iteration accumulating TD errors)
- `compute_ppo_loss()`: Importance sampling ratio calculation and clipping
- `training/ppo_trainer.py`: PPOTrainer (including `generate_responses`, `compute_rewards`, `training_step`)

---

### Lesson 13 — GRPO (Group Relative Policy Optimization)

**File Path**: `tutorial/stage04_alignment/lesson13_grpo/`

**Learning Objectives**:
- Understand GRPO principles (DeepSeek's approach)
- Implement the GRPO loss function
- Compare GRPO vs. DPO

**Core Concepts**:
- **GRPO Core**: Sample multiple responses for each prompt, rank by reward, weight loss based on relative rankings
- **GRPO Loss**: `L_GRPO = -E[Σᵢ softmax(sᵢ) · log P(yᵢ|x)]`, higher reward responses get higher weights
- **GRPO vs DPO**:
  - GRPO: Requires no reference model, directly uses rewards, more stable, lower VRAM
  - DPO: Requires a reference model, uses pairwise preference data, simpler to implement

**Implementation Tasks**:
- `data/grpo_dataset.py`: GRPODataset (multiple responses per prompt)
- `compute_grpo_loss()`: Softmax-weighted cross-entropy
- `training/grpo_trainer.py`: GRPOTrainer (including `generate_responses`, `compute_rewards`)

---

## STAGE 5: Advanced Techniques (Lessons 14–15)

### Lesson 14 — LoRA and QLoRA (Parameter-Efficient Fine-Tuning)

**File Path**: `tutorial/stage05_advanced/lesson14_lora_qlora/`

**Learning Objectives**:
- Understand LoRA principles and implementation
- Master QLoRA quantization techniques
- Implement mixed-precision training

**Core Concepts**:
- **LoRA Core**: Add low-rank updates `W = W₀ + BA` on top of frozen weights, where `B∈ℝ^{d×r}`, `A∈ℝ^{r×k}`, `r ≪ min(d,k)`
- **Forward Pass**: `h = W₀x + BAx`, separate low-dimensional computation
- **Parameter Compression Example**: 4096×4096 matrix → rank-8 LoRA → 250x compression (16M → 65K parameters)
- **NF4 Quantization**: 4-bit normalized floating-point format optimized for LLM weight distributions
- **QLoRA Pipeline**: Load quantized model → FP16/BF16 training for LoRA parameters → Merge during inference
- **Precision Types**: FP32, FP16, BF16 (larger dynamic range), INT8/INT4

**Implementation Tasks**:
- `model/lora.py`: LoRALayer (including frozen original weights, trainable A/B matrices, `merge()` method)
- `get_lora_model()`: Apply LoRA to target modules (q/k/v/o_proj + FFN layers)
- `quantize_tensor()` / `dequantize_tensor()`: Block-based NF4 quantization/dequantization

---

### Lesson 15 — Efficient Attention Mechanisms

**File Path**: `tutorial/stage05_advanced/lesson15_efficient_attention/`

**Learning Objectives**:
- Understand Flash Attention principles
- Master MQA and GQA techniques
- Explore various attention optimization methods

**Core Concepts**:
- **Standard Attention Bottleneck**: `O(N²d)` computation, `O(N²)` memory (attention matrix storage)
- **Flash Attention**: IO-aware algorithm, tiling computation to avoid storing the full NxN matrix, online Softmax (2-pass→1-pass), memory `O(N²)→O(N)`
- **MQA (Multi-Query Attention)**: All Q heads share a single K/V, drastically reducing KV cache, but may affect quality
- **GQA (Grouped-Query Attention)**: Q heads are grouped, each group shares K/V, `num_groups = num_heads/num_kv_heads`, balances quality and efficiency (used by Llama 3/Qwen)
- **Other Optimizations**: Sparse attention (local window/random/block sparse), linear attention (kernel function approximation), sliding window attention

**Implementation Tasks**:
- `model/attention.py`: GroupedQueryAttention (including `repeat_kv()` to expand KV heads)
- GQA vs MHA memory trade-off analysis
- Sliding window attention mask implementation

---

## Core Framework Components Summary

### Implementation Dependencies


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

### Test Execution Commands

```bash
# Run all tests
python -m pytest tests -v
python -m pytest tutorial/**/testcases -v

# Run by stage
python -m pytest tutorial/stage01_foundation/ -v
python -m pytest tutorial/stage02_pretrain/ -v
python -m pytest tutorial/stage03_instruction_follow/ -v
python -m pytest tutorial/stage04_alignment/ -v
python -m pytest tutorial/stage05_advanced/ -v

# Single lesson test
python -m pytest tutorial/stage01_foundation/lesson01_tokenizer_embedding/testcases -v

```

---

## Learning Path Suggestions

1. **Solidify the Foundation (L01–L05)**: Ensure the shape/numerical contracts of each component pass the tests, and understand the mathematical principles of each module.
2. **Training System (L06–L09)**: Focus on the impact of data quality on the final model, and experiment with different hyperparameter configurations.
3. **Instruction Alignment (L10)**: Understand the critical role of loss masking in instruction following capabilities.
4. **Preference Alignment (L11–L13)**: Progress step-by-step: DPO (simplest) → GRPO (DeepSeek's approach) → PPO (most complex).
5. **Efficiency Optimization (L14–L15)**: Apply LoRA/GQA to previously implemented models and experience the improvement in parameter efficiency.