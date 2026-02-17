"""
训练器基类模块
定义 LLM 训练的通用框架和流程。

训练器职责:
1. 管理训练循环 (epoch/batch 迭代)
2. 处理模型前向/反向传播
3. 优化器和学习率调度
4. 日志记录和指标追踪
5. 检查点保存和恢复
6. 分布式训练支持

设计原则:
- 基类提供通用框架，子类实现具体训练逻辑
- 支持灵活的扩展和自定义
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Optional, Any, List
import os
import time
from pathlib import Path


class Trainer:
    """
    训练器基类

    提供完整的训练框架，包括:
    - 训练循环管理
    - 验证流程
    - 检查点保存/加载
    - 日志记录
    - 分布式训练支持

    子类需要实现:
    - train_step(): 单步训练逻辑
    - compute_loss(): 损失计算
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        初始化训练器。

        Args:
            model: 要训练的模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器 (可选)
            optimizer: 优化器 (可选，子类可创建)
            lr_scheduler: 学习率调度器 (可选)
            config: 训练配置字典，包含:
                - output_dir: 输出目录
                - num_epochs: 训练轮数
                - max_steps: 最大训练步数 (覆盖 num_epochs)
                - gradient_accumulation_steps: 梯度累积步数
                - max_grad_norm: 梯度裁剪阈值
                - logging_steps: 日志记录间隔
                - eval_steps: 验证间隔
                - save_steps: 检查点保存间隔
                - save_total_limit: 保留的检查点数量
            device: 训练设备，默认自动选择

        初始化内容:
            Step 1: 保存模型和数据加载器
            Step 2: 设置设备 (cuda/cpu)
            Step 3: 移动模型到设备
            Step 4: 初始化优化器 (如果没有提供)
            Step 5: 初始化学习率调度器 (如果没有提供)
            Step 6: 设置训练状态 (global_step, current_epoch 等)
            Step 7: 创建输出目录
        """
        pass

    def train(self, num_epochs: Optional[int] = None):
        """
        主训练循环。

        Args:
            num_epochs: 训练轮数，覆盖 config 中的设置

        训练流程:
            Step 1: 训练前准备
                    - 设置模型为训练模式 (model.train())
                    - 初始化统计变量 (total_loss, step_count 等)

            Step 2: Epoch 循环
                    for epoch in range(start_epoch, num_epochs):
                        self.current_epoch = epoch

                        Step 2.1: 批次循环
                        for batch in train_dataloader:
                            Step 2.1.1: 移动批次到设备
                            Step 2.1.2: 调用 train_step() 执行训练
                            Step 2.1.3: 更新全局步数
                            Step 2.1.4: 梯度累积处理
                            Step 2.1.5: 更新优化器和调度器
                            Step 2.1.6: 记录日志
                            Step 2.1.7: 定期验证
                            Step 2.1.8: 定期保存检查点

                        Step 2.2: Epoch 结束处理
                                - 保存 epoch 检查点
                                - 执行验证
                                - 记录 epoch 统计

            Step 3: 训练结束
                    - 保存最终模型
                    - 保存训练状态
        """
        pass

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行单步训练。

        Args:
            batch: 批次数据，来自 DataLoader

        Returns:
            包含训练指标的字典，如 {"loss": loss_value}

        子类必须实现此方法。

        一般流程:
            Step 1: 前向传播
                    outputs = model(**batch)

            Step 2: 计算损失
                    loss = outputs.loss

            Step 3: 梯度累积缩放 (如果需要)
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps

            Step 4: 反向传播
                    loss.backward()

            Step 5: 梯度裁剪 (如果需要)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            Step 6: 优化器步骤 (如果达到累积步数)
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()  # 每步更新
                        optimizer.zero_grad()

            Step 7: 返回指标
        """
        raise NotImplementedError("Subclasses must implement train_step()")

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估模型。

        Returns:
            包含验证指标的字典

        流程:
            Step 1: 设置模型为评估模式
                    model.eval()

            Step 2: 禁用梯度计算
                    with torch.no_grad():

            Step 3: 批次循环
                    total_loss = 0
                    for batch in val_dataloader:
                        Step 3.1: 移动批次到设备
                        Step 3.2: 前向传播
                        Step 3.3: 计算损失
                        Step 3.4: 累加损失

            Step 4: 计算平均指标
                    avg_loss = total_loss / len(val_dataloader)
                    perplexity = exp(avg_loss)

            Step 5: 恢复训练模式
                    model.train()

            Step 6: 返回指标
        """
        pass

    def save_checkpoint(self, output_dir: str, epoch: int, step: int, **kwargs):
        """
        保存训练检查点。

        Args:
            output_dir: 保存目录
            epoch: 当前 epoch
            step: 当前全局步数
            **kwargs: 额外要保存的状态

        保存内容:
            - model_state_dict: 模型权重
            - optimizer_state_dict: 优化器状态
            - lr_scheduler_state_dict: 调度器状态
            - epoch: 当前 epoch
            - global_step: 全局步数
            - config: 训练配置
            - loss: 当前损失 (可选)

        流程:
            Step 1: 创建保存目录
                    os.makedirs(output_dir, exist_ok=True)

            Step 2: 构建检查点字典
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": step,
                        "config": config,
                    }
                    checkpoint.update(kwargs)

            Step 3: 保存到文件
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}.pt")
                    torch.save(checkpoint, checkpoint_path)

            Step 4: 管理检查点数量
                    如果 save_total_limit 设置，删除旧的检查点
        """
        pass

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载训练检查点。

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            加载的检查点字典

        流程:
            Step 1: 加载检查点文件
                    checkpoint = torch.load(checkpoint_path, map_location=device)

            Step 2: 恢复模型状态
                    model.load_state_dict(checkpoint["model_state_dict"])

            Step 3: 恢复优化器状态
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            Step 4: 恢复调度器状态 (如果存在)
                    if "lr_scheduler_state_dict" in checkpoint:
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            Step 5: 恢复训练状态
                    current_epoch = checkpoint["epoch"]
                    global_step = checkpoint["global_step"]

            Step 6: 返回检查点字典
        """
        pass

    def log(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        记录训练指标。

        Args:
            metrics: 指标字典，如 {"loss": 2.5, "lr": 1e-4}
            step: 当前步数
            prefix: 指标前缀，如 "train/"、"eval/"

        功能:
            - 打印到控制台
            - 记录到日志文件
            - 发送到 WandB/TensorBoard
        """
        pass

    def get_train_dataloader(self) -> DataLoader:
        """
        获取训练数据加载器。

        Returns:
            DataLoader 实例
        """
        pass

    def get_eval_dataloader(self) -> DataLoader:
        """
        获取验证数据加载器。

        Returns:
            DataLoader 实例
        """
        pass

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        将批次数据移动到训练设备。

        Args:
            batch: 批次数据字典

        Returns:
            移动到设备后的批次数据
        """
        pass

    def _set_seed(self, seed: int):
        """
        设置随机种子，确保可复现。

        Args:
            seed: 随机种子
        """
        pass
