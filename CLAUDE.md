# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational PyTorch-based LLM (Large Language Model) training framework. It implements the complete pipeline for training language models from scratch through alignment techniques.

## Common Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run single test function
pytest tests/test_model.py::test_causal_lm -v

# Run with verbose output
pytest -v
```

## Architecture

### Training Pipeline Stages
The framework follows the standard LLM training progression:
1. **Pretraining** (`scripts/train_pretrain.py`) - Unsupervised language modeling
2. **SFT (Supervised Fine-tuning)** (`scripts/train_sft.py`) - Instruction tuning
3. **Alignment** - DPO (`scripts/train_dpo.py`), GRPO, PPO

### Core Modules

- **`model/`** - Transformer architecture: attention (MHA/GQA), feedforward (SwiGLU), embeddings (RoPE), norm layers (LayerNorm/RMSNorm), causal LM head, LoRA/QLoRA support
- **`training/`** - Trainers for each stage: `PretrainTrainer`, `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `PPOTrainer`
- **`data/`** - Datasets: `PretrainDataset`, `SFTDataset`, `DpoDataset`, plus data filtering utilities
- **`tokenizer/`** - Tokenizer implementations: BPE, byte-level
- **`loss/`** - Loss functions: cross-entropy, DPO, GRPO, PPO
- **`optimizer/`** - Optimizers: AdamW, Lion, with `WarmupCosineScheduler`
- **`evaluation/`** - Benchmarks: perplexity, MMLU, HumanEval
- **`utils/`** - Checkpointing, mixed precision (fp16/bf16), Flash Attention, early stopping, W&B logging
- **`configs/`** - Model configurations (e.g., `tiny_config.py` for ~10M parameter model)

### Configuration
Model configs are defined in `configs/` as `ModelConfig` dataclass:
- `tiny_config.py` - ~10M parameter model for testing
- `qwen3_0.5b.py` - Qwen 3 0.5B configuration
- `lora_config.py` - LoRA configuration

### Tutorials
The `tutorial/` directory contains educational notebooks organized by training stage:
- `stage01_foundation/` - Fundamentals
- `stage02_pretrain/` - Pretraining
- `stage03_instruction_follow/` - SFT
- `stage04_alignment/` - DPO/GRPO alignment
- `stage05_advanced/` - Advanced topics

### Entry Points
- `scripts/train_pretrain.py` - Pretraining script
- `scripts/train_sft.py` - SFT script
- `scripts/train_dpo.py` - DPO alignment script
- `scripts/evaluate.py` - Evaluation script
