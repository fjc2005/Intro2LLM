"""
监督微调脚本
执行指令微调 (SFT)。

使用示例:
    python scripts/train_sft.py \
        --model_path outputs/pretrain/final_model \
        --data_path data/sft/alpaca.jsonl \
        --output_dir outputs/sft \
        --batch_size 16 \
        --num_epochs 3 \
        --learning_rate 2e-5
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="监督微调")

    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练模型路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器路径")

    # 数据配置
    parser.add_argument("--data_path", type=str, required=True,
                        help="SFT 数据路径")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--template", type=str, default="alpaca",
                        choices=["alpaca", "chatml", "vicuna"],
                        help="提示词模板")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="outputs/sft",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率 (通常比预训练小)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup 比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # 系统配置
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载进程数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # LoRA 配置
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA 目标模块")

    # 日志配置
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用 W&B")
    parser.add_argument("--wandb_project", type=str, default="llm-sft",
                        help="W&B 项目名称")

    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    # Step 1: 加载预训练模型和分词器
    # model = load_model(args.model_path)
    # tokenizer = load_tokenizer(args.tokenizer_path or args.model_path)

    # Step 2: 应用 LoRA (如果需要)
    # if args.use_lora:
    #     from model.lora import get_lora_model
    #     model = get_lora_model(
    #         model,
    #         r=args.lora_r,
    #         lora_alpha=args.lora_alpha,
    #         lora_dropout=args.lora_dropout,
    #         target_modules=args.lora_target_modules,
    #     )

    # Step 3: 准备 SFT 数据
    # from data.sft_dataset import SFTDataset
    # train_dataset = SFTDataset(
    #     args.data_path,
    #     tokenizer,
    #     max_length=args.max_length,
    #     template=args.template,
    # )

    # Step 4: 设置优化器和学习率调度器
    # SFT 通常使用较小的学习率和较短的 warmup

    # Step 5: 初始化训练器并训练
    # from training.sft_trainer import SFTTrainer
    # trainer = SFTTrainer(...)
    # trainer.train(num_epochs=args.num_epochs)

    # Step 6: 保存模型
    # 如果使用 LoRA，保存 LoRA 权重或合并后的模型

    pass


if __name__ == "__main__":
    main()
