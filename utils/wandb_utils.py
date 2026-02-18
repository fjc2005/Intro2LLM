"""
Weights & Biases (W&B) 工具模块
集成 W&B 进行实验跟踪和可视化。

W&B 功能:
- 指标记录 (loss、accuracy、learning rate 等)
- 超参数记录
- 模型检查点版本管理
- 实验对比
- 可视化 (曲线、直方图、表格等)
- 协作分享

替代方案:
- TensorBoard: 本地可视化，无需联网
- MLflow: 开源，可自托管
- Neptune: 商业方案
"""

import os
from typing import Dict, Optional, Any, List
import torch
import torch.nn as nn


# 尝试导入 wandb，如果未安装则不报错
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def init_wandb(
    project: str = "llm-scratch",
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    job_type: str = "training",
    resume: bool = False,
    entity: Optional[str] = None,
) -> Any:
    """
    初始化 W&B。

    Args:
        project: W&B 项目名称
        name: 实验名称 (None 则自动生成)
        config: 超参数字典
        tags: 标签列表
        group: 实验分组 (用于对比多次运行)
        job_type: 任务类型 (training/evaluation)
        resume: 是否恢复之前的运行
        entity: W&B 团队/用户名称

    Returns:
        wandb run 对象

    初始化流程:
        Step 1: 检查 wandb 是否可用
                如果 wandb 库未安装，打印提示信息并返回 None

        Step 2: 初始化 wandb
                使用传入的参数初始化 wandb 运行
                包括项目名、实验名、配置、标签、分组等

        Step 3: 返回 run 对象
    """
    pass


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
):
    """
    记录指标。

    Args:
        metrics: 指标字典，如 {"loss": 2.5, "accuracy": 0.85}
        step: 当前步数
        prefix: 指标前缀，如 "train/"、"eval/"

    示例:
        log_metrics({"loss": 2.5, "lr": 1e-4}, step=100, prefix="train/")
        # 在 W&B 中记录为: train/loss=2.5, train/lr=0.0001
    """
    pass


def log_hyperparameters(config: Dict[str, Any]):
    """
    记录超参数。

    Args:
        config: 超参数字典

    示例:
        log_hyperparameters({
            "learning_rate": 1e-4,
            "batch_size": 32,
            "model": "llama-7b",
        })
    """
    pass


def log_model_checkpoint(
    checkpoint_path: str,
    aliases: Optional[List[str]] = None,
):
    """
    记录模型检查点。

    Args:
        checkpoint_path: 检查点文件路径
        aliases: 别名列表，如 ["best", "epoch-10"]

    功能:
    - 上传检查点到 W&B Artifacts
    - 版本化管理
    - 自动保存检查点元数据
    """
    pass


def log_histogram(
    values: torch.Tensor,
    name: str,
    step: Optional[int] = None,
):
    """
    记录直方图。

    用于可视化权重分布、梯度分布等。

    Args:
        values: 要可视化的张量
        name: 直方图名称
        step: 当前步数
    """
    pass


def log_gradients(
    model: nn.Module,
    step: int,
    prefix: str = "gradients/",
):
    """
    记录模型梯度统计。

    Args:
        model: 模型
        step: 当前步数
        prefix: 指标前缀

    记录内容:
    - 每层梯度的 L2 范数
    - 梯度的均值、标准差
    - 梯度消失/爆炸检测
    """
    pass


def log_weights(
    model: nn.Module,
    step: int,
    prefix: str = "weights/",
):
    """
    记录模型权重统计。

    Args:
        model: 模型
        step: 当前步数
        prefix: 指标前缀

    记录内容:
    - 权重的分布直方图
    - 权重的均值、标准差
    - 权重更新幅度
    """
    pass


def watch_model(
    model: nn.Module,
    log: str = "gradients",
    log_freq: int = 100,
):
    """
    监控模型。

    W&B 自动记录模型的梯度、权重直方图。

    Args:
        model: 要监控的模型
        log: 记录内容，"gradients" / "parameters" / "all"
        log_freq: 记录频率

    示例:
        watch_model(model, log="all", log_freq=100)
    """
    pass


def log_table(
    name: str,
    data: List[List[Any]],
    columns: Optional[List[str]] = None,
):
    """
    记录表格数据。

    用于展示预测结果、对比等。

    Args:
        name: 表格名称
        data: 表格数据，二维列表
        columns: 列名列表

    示例:
        log_table("predictions", [
            ["input 1", "predicted 1", "actual 1"],
            ["input 2", "predicted 2", "actual 2"],
        ], columns=["Input", "Prediction", "Actual"])
    """
    pass


def log_text(
    text: str,
    name: str = "generated_text",
    step: Optional[int] = None,
):
    """
    记录文本。

    用于展示生成的文本样本。

    Args:
        text: 文本内容
        name: 文本名称
        step: 当前步数
    """
    pass


def finish_wandb():
    """
    结束 W&B 运行。

    确保所有数据都被正确上传。
    """
    pass


class WandbLogger:
    """
    W&B 日志记录器封装类

    提供更面向对象的接口。
    """

    def __init__(
        self,
        project: str = "llm-scratch",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        """
        初始化日志记录器。

        Args:
            project: W&B 项目名
            name: 实验名
            config: 超参数配置
            enabled: 是否启用 (方便调试时关闭)
        """
        pass

    def log(self, metrics: Dict, step: Optional[int] = None):
        """记录指标。"""
        pass

    def log_hyperparams(self, params: Dict):
        """记录超参数。"""
        pass

    def save_checkpoint(self, path: str):
        """保存检查点。"""
        pass

    def finish(self):
        """结束记录。"""
        pass


def get_wandb_artifact(
    artifact_name: str,
    artifact_type: str = "model",
) -> str:
    """
    获取 W&B Artifact。

    Args:
        artifact_name: artifact 名称，如 "project/model:best"
        artifact_type: artifact 类型

    Returns:
        artifact 本地路径
    """
    pass
