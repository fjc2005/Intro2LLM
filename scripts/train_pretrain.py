"""
预训练脚本
执行因果语言模型的无监督预训练。

使用示例:
    python scripts/train_pretrain.py \
        --config configs/tiny_config.py \
        --data_path data/pretrain \
        --output_dir outputs/pretrain \
        --batch_size 32 \
        --num_epochs 3

关键步骤:
1. 加载配置和模型
2. 准备数据
3. 设置优化器和学习率调度器
4. 训练循环
5. 保存检查点
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from model.causal_lm import CausalLM
from model.config import ModelConfig
from data.pretrain_dataset import PretrainDataset
from training.pretrain_trainer import PretrainTrainer
from optimizer.adamw import AdamW
from optimizer.scheduler import WarmupCosineScheduler
from utils.checkpoint import CheckpointManager
from utils.wandb_utils import init_wandb, log_metrics


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="预训练语言模型")

    # 模型配置
    parser.add_argument("--config", type=str, required=True,
                        help="模型配置文件路径")

    # 数据配置
    parser.add_argument("--data_path", type=str, required=True,
                        help="预训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器路径 (None 表示使用默认)")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大训练步数 (覆盖 num_epochs)")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup 步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # 系统配置
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 检查点配置
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="检查点保存目录 (默认使用 output_dir/checkpoints)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="每多少步保存检查点")

    # 日志配置
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用 W&B 记录")
    parser.add_argument("--wandb_project", type=str, default="llm-pretrain",
                        help="W&B 项目名称")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="每多少步记录日志")

    # 混合精度
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=[None, "fp16", "bf16"],
                        help="混合精度类型")

    return parser.parse_args()


def main():
    """主训练函数。"""
    args = parse_args()

    # Step 1: 设置随机种子
    # torch.manual_seed(args.seed)

    # Step 2: 加载配置
    # config = load_config(args.config)

    # Step 3: 创建输出目录
    # os.makedirs(args.output_dir, exist_ok=True)

    # Step 4: 初始化模型
    # model = CausalLM(config)

    # Step 5: 准备数据
    # train_dataset = PretrainDataset(args.data_path, tokenizer, max_length=config.max_position_embeddings)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     collate_fn=train_dataset.collate_fn,
    # )

    # Step 6: 设置优化器和学习率调度器
    # optimizer = AdamW(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay,
    # )
    # scheduler = WarmupCosineScheduler(
    #     optimizer,
    #     num_warmup_steps=args.warmup_steps,
    #     num_training_steps=len(train_dataloader) * args.num_epochs,
    # )

    # Step 7: 初始化训练器
    # trainer = PretrainTrainer(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     optimizer=optimizer,
    #     lr_scheduler=scheduler,
    #     config=vars(args),
    #     device=args.device,
    # )

    # Step 8: 从检查点恢复 (如果需要)
    # if args.resume_from_checkpoint:
    #     trainer.load_checkpoint(args.resume_from_checkpoint)

    # Step 9: 初始化 W&B (如果需要)
    # if args.use_wandb:
    #     init_wandb(
    #         project=args.wandb_project,
    #         config=vars(args),
    #     )

    # Step 10: 开始训练
    # trainer.train(num_epochs=args.num_epochs)

    # Step 11: 保存最终模型
    # trainer.save_model(os.path.join(args.output_dir, "final_model"))

    pass


if __name__ == "__main__":
    main()
