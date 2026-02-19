# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Intro2LLM** is an educational framework for learning Large Language Model implementation from first principles. It covers tokenization, transformer architecture, pretraining, instruction tuning, alignment (DPO/PPO/GRPO), and efficiency techniques (LoRA/QLoRA, flash attention) across 15 structured lessons.

## Commands

Requires Python 3.10+ in a virtual environment.

```bash
# Core regression tests
python -m pytest tests -v

# All tutorial lesson tests
python -m pytest tutorial/**/testcases -v

# Single lesson tests
python -m pytest tutorial/stage01_foundation/lesson01_tokenizer_embedding/testcases -v

# CLI entry points
python scripts/train_pretrain.py --help
python scripts/train_sft.py --help
python scripts/train_dpo.py --help
python scripts/evaluate.py --help
```

## Architecture

### Core Packages

**`model/`** — Composable transformer components:
- `config.py`: `ModelConfig` dataclass (all hyperparameters: vocab size, hidden size, layers, heads, RoPE theta, activations)
- `embedding.py`: `TokenEmbedding`, sinusoidal `PositionalEncoding`, `RoPE`
- `attention.py`: `MultiHeadAttention`, `GroupedQueryAttention` (GQA)
- `transformer_block.py`: Composes norm + attention + FFN into a single layer
- `causal_lm.py`: Full `CausalLM` model with KV cache for generation
- `lora.py` / `qlora.py`: LoRA and QLoRA for parameter-efficient fine-tuning

**`tokenizer/`** — Inherits from `BaseTokenizer` (abstract: encode/decode, special tokens, save/load):
- `bpe_tokenizer.py`: Standard BPE
- `byte_level_tokenizer.py`: Byte-level BPE (GPT-2 style)

**`training/`** — `Trainer` base class extended by task-specific trainers:
- `pretrain_trainer.py`, `sft_trainer.py`, `dpo_trainer.py`, `ppo_trainer.py`, `grpo_trainer.py`

**`data/`** — Dataset classes for each training paradigm (pretrain, SFT, DPO); `filtering.py` for data cleaning.

**`configs/`** — Reusable `ModelConfig` instances: `tiny_config.py` (10M params, for fast tests), `qwen3_0.5b.py`, `lora_config.py`.

**`loss/`**, **`optimizer/`**, **`evaluation/`**, **`utils/`** — Domain-focused modules (loss functions, AdamW/Lion optimizers, schedulers, perplexity/MMLU/HumanEval evaluation, checkpointing, flash attention, mixed precision).

### Tutorial Structure

`tutorial/` is organized into 5 stages, each with lessons containing:
- `README.md` — Theory and implementation requirements
- `testcases/basic_test.py` and `testcases/advanced_test.py` — Tests that define expected behavior

| Stage | Lessons | Topics |
|-------|---------|--------|
| `stage01_foundation/` | 01–05 | Tokenizer, embeddings, attention, FFN, KV cache, model config |
| `stage02_pretrain/` | 06–09 | Data loading, loss, optimizer, checkpointing, evaluation |
| `stage03_instruction_follow/` | 10 | SFT |
| `stage04_alignment/` | 11–13 | DPO, PPO/GAE, GRPO |
| `stage05_advanced/` | 14–15 | LoRA/QLoRA, flash attention |

### Design Patterns

- **Configuration-driven**: All model hyperparameters flow through `ModelConfig`; configs live in `configs/` and are passed via script arguments.
- **Base class interfaces**: `BaseTokenizer`, `Trainer`, and evaluators define contracts; task-specific subclasses extend them.
- **Test contracts**: Tests verify shape, dtype, and numerical output contracts — not just that code runs.
- **Tiny config for tests**: Use `configs/tiny_config.py` (10M params) to keep test runs fast.

## Coding Conventions

- PEP 8, 4-space indentation; `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Docstrings on public classes/functions; explicit type hints on new APIs
- New schedulers → `optimizer/`; new evaluators → `evaluation/`; new configs → `configs/`

## Commit Style

Short imperative messages with a scope hint: `update: lesson3 tutorial`, `fix: attention mask bug`. Keep tutorial-only changes separate from core framework changes.
