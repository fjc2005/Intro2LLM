# Repository Guidelines

## Project Structure & Module Organization
Core implementation lives in top-level Python packages:
- `model/`: transformer, attention, embeddings, LoRA/QLoRA, config.
- `training/`: base and task-specific trainers (`pretrain`, `sft`, `dpo`, `ppo`, `grpo`).
- `data/`, `tokenizer/`, `optimizer/`, `loss/`, `evaluation/`, `utils/`: data pipelines, tokenization, optimization, objectives, eval, and shared helpers.
- `scripts/`: entry points for train/eval workflows (for example `scripts/train_pretrain.py`).
- `tests/`: repository-level pytest suite.
- `tutorial/`: stage/lesson teaching materials with per-lesson `testcases/`.

## Build, Test, and Development Commands
Use Python 3.10+ in a virtual environment.
- `python -m pytest tests -v`: run core regression tests.
- `python -m pytest tutorial/**/testcases -v`: run tutorial lesson tests.
- `python scripts/train_pretrain.py --help`: inspect pretraining CLI options.
- `python scripts/train_sft.py --help`: inspect SFT training options.
- `python scripts/evaluate.py --help`: inspect evaluation options.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and readable line lengths.
- Use `snake_case` for functions/variables/files, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep modules focused by domain (for example new schedulers in `optimizer/`, new evaluators in `evaluation/`).
- Add docstrings for public classes/functions; prefer explicit type hints on new APIs.

## Testing Guidelines
- Framework: `pytest`.
- Test files should be named `test_*.py`; test functions should be `test_*`.
- Add unit tests next to changed behavior in `tests/`; if editing lesson content, update the relevant `tutorial/.../testcases/` tests too.
- Cover shape, dtype, and key output contract checks for model/training code.

## Commit & Pull Request Guidelines
- Current history follows short, imperative messages with a scope hint (for example `update: lesson3 tutorial`). Keep this format.
- Keep commits focused; avoid mixing tutorial-only and core-framework refactors in one commit.
- PRs should include:
  - What changed and why.
  - How to validate (exact pytest or script commands).
  - Linked issue/context.
  - Screenshots/log snippets only when outputs or metrics changed.

## Security & Configuration Tips
- Keep dataset paths, checkpoints, and secrets out of git; use local or ignored output directories such as `outputs/`.
- Put reusable hyperparameters in `configs/` and pass them via script arguments instead of hardcoding.
