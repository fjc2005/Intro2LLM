# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intro2LLM is an educational "From Scratch" implementation of a Large Language Model in PyTorch. The codebase demonstrates how to build a complete LLM training pipeline including tokenization, model architecture, training (Pretrain → SFT → DPO/GRPO/PPO), and evaluation.

## Unique Code Style (Critical)

This project follows a **strict documentation-first teaching style**:

1. **Function bodies must be `pass` or `return ...` only** - Never write executable PyTorch code like `torch.matmul`, `x + y`, etc.

2. **Comments are the implementation** - Each function contains detailed Chinese comments explaining:
   - Algorithm steps (Step 1, Step 2...)
   - Tensor shape transformations (e.g., `# [batch, seq, hidden] -> [batch, seq, heads, head_dim]`)
   - Mathematical formulas (e.g., `# y = x * rsqrt(mean(x^2) + eps)`)
   - Edge case handling

3. **Purpose**: Students should be able to implement the code by following the natural language description without seeing the actual code.

Example pattern:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [batch, seq_len, hidden_size]
    Returns:
        normalized: [batch, seq_len, hidden_size]
    """
    # Step 1: Convert to float32 for numerical stability
    # Step 2: Calculate mean square along last dimension
    #    variance = x.pow(2).mean(dim=-1, keepdim=True)
    #    Shape: [batch, seq_len, 1]
    # Step 3: Compute RMS and apply normalization
    #    Formula: output = x / sqrt(variance + eps) * self.weight
    # Step 4: Return to original dtype
    pass
```

## Common Commands

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_rmsnorm -v

# Run with coverage
pytest tests/ --cov=.
```

### Training Scripts
```bash
# Pretraining
python scripts/train_pretrain.py \
    --config configs/tiny_config.py \
    --data_path data/pretrain \
    --output_dir outputs/pretrain \
    --batch_size 32 \
    --num_epochs 3

# Supervised Fine-Tuning (SFT)
python scripts/train_sft.py \
    --model_path outputs/pretrain/final_model \
    --data_path data/sft/alpaca.jsonl \
    --output_dir outputs/sft \
    --batch_size 16 \
    --num_epochs 3 \
    --learning_rate 2e-5

# DPO Training
python scripts/train_dpo.py \
    --model_path outputs/sft/final_model \
    --data_path data/dpo/preference.jsonl \
    --output_dir outputs/dpo \
    --beta 0.1

# Evaluation
python scripts/evaluate.py \
    --model_path outputs/dpo/final_model \
    --task perplexity \
    --data_path data/eval/wiki
```

## High-Level Architecture

### Training Pipeline
```
Pretrain (Causal LM) → SFT (Instruction Tuning) → DPO/GRPO/PPO (Alignment)
```

Each stage uses its own Trainer class (PretrainTrainer, SFTTrainer, DPOTrainer, GRPOTrainer, PPOTrainer) that inherits from the base Trainer.

### Model Architecture
The model follows the modern LLaMA/Qwen architecture:

- **Config**: `model/config.py` - Central configuration via dataclass (hidden_size, num_layers, num_heads, etc.)
- **Embedding**: `model/embedding.py` - Token embeddings + RoPE (Rotary Position Embedding)
- **Attention**: `model/attention.py` - MultiHeadAttention (MHA) and GroupedQueryAttention (GQA)
- **FFN**: `model/feedforward.py` - SwiGLU/GeGLU gated feed-forward networks
- **Norm**: `model/norm.py` - RMSNorm and LayerNorm
- **Block**: `model/transformer_block.py` - Pre-LN Transformer decoder block
- **CausalLM**: `model/causal_lm.py` - Full model with generate() method
- **LoRA/QLoRA**: `model/lora.py`, `model/qlora.py` - Parameter-efficient fine-tuning

Key architectural switches in ModelConfig:
- `use_rms_norm` (True=LLaMA-style, False=Original Transformer)
- `use_rope` (True=modern, False=Sinusoidal)
- `num_key_value_heads` (< num_attention_heads enables GQA)

### Data Flow
- **PretrainDataset**: Simple causal LM on raw text (next token prediction)
- **SFTDataset**: Instruction-following with loss masking on prompts (labels=-100 for prompt tokens)
- **DPODataset**: Preference pairs (chosen vs rejected) for direct preference optimization

### Loss Functions
- `loss/cross_entropy.py` - Standard CE for pretraining/SFT
- `loss/dpo_loss.py` - Direct Preference Optimization without explicit reward model
- `loss/grpo_loss.py` - Group Relative Preference Optimization (offline RL)
- `loss/ppo_loss.py` - PPO with actor-critic for RLHF

### Key Dependencies Between Modules
```
causal_lm.py → transformer_block.py → [attention.py, feedforward.py, norm.py]
attention.py → embedding.py (RoPE)
train_*.py → trainer.py → loss/*.py + data/*.py
```

## Project Structure Summary

```
tokenizer/      # BPE, Byte-level BPE implementations
model/          # Core LLM architecture (config, norm, embedding, attention, ffn, block, causal_lm, lora, qlora)
data/           # Dataset classes (pretrain, sft, dpo)
loss/           # Loss functions (cross_entropy, dpo, grpo, ppo)
optimizer/      # AdamW, Lion, learning rate schedulers
training/       # Trainer classes (base, pretrain, sft, dpo, grpo, ppo)
evaluation/     # Perplexity, MMLU, HumanEval
utils/          # Checkpoint, W&B, early stopping, mixed precision, flash attention
configs/        # Model configs (tiny_config, qwen3_0.5b, lora_config)
scripts/        # Training and evaluation entry points
tests/          # Unit tests with pytest
```
