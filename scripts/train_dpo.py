"""
DPO 训练脚本
执行直接偏好优化。

使用示例:
    python scripts/train_dpo.py \
        --model_path outputs/sft/final_model \
        --ref_model_path outputs/sft/final_model \
        --data_path data/dpo/preference.jsonl \
        --output_dir outputs/dpo \
        --batch_size 8 \
        --beta 0.1
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="DPO 训练")

    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="SFT 模型路径 (策略模型)")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="参考模型路径 (默认与 model_path 相同)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器路径")

    # 数据配置
    parser.add_argument("--data_path", type=str, required=True,
                        help="DPO 偏好数据路径")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="最大 prompt 长度")

    # DPO 配置
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO 温度系数")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="标签平滑")

    # 训练配置
    parser.add_argument("--output_dir", type=str, default="outputs/dpo",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="训练轮数 (DPO 通常 1 轮即可)")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="学习率 (比 SFT 更小)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup 比例")

    # 系统配置
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载进程数")

    # LoRA 配置
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA 秩")

    # 日志配置
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用 W&B")
    parser.add_argument("--wandb_project", type=str, default="llm-dpo",
                        help="W&B 项目名称")

    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    # Step 1: 加载策略模型和参考模型
    # policy_model = load_model(args.model_path)
    # ref_model = load_model(args.ref_model_path or args.model_path)

    # Step 2: 冻结参考模型
    # for param in ref_model.parameters():
    #     param.requires_grad = False

    # Step 3: 应用 LoRA (如果需要)
    # if args.use_lora:
    #     from model.lora import get_lora_model
    #     policy_model = get_lora_model(policy_model, r=args.lora_r)

    # Step 4: 准备 DPO 数据
    # from data.dpo_dataset import DPODataset
    # train_dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_length)

    # Step 5: 初始化训练器
    # from training.dpo_trainer import DPOTrainer
    # trainer = DPOTrainer(
    #     model=policy_model,
    #     ref_model=ref_model,
    #     beta=args.beta,
    #     ...
    # )

    # Step 6: 训练
    # trainer.train(num_epochs=args.num_epochs)

    # Step 7: 保存模型
    # trainer.save_model(args.output_dir)

    pass


if __name__ == "__main__":
    main()
