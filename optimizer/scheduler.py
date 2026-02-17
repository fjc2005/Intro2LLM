"""
学习率调度器模块
实现各种学习率调度策略。

学习率调度是训练大语言模型的关键，影响:
- 收敛速度
- 最终性能
- 训练稳定性

常用策略:
- Warmup: 训练初期线性/指数增加学习率
- Cosine Annealing: 余弦退火衰减
- Warmup + Cosine: 现代 LLM 标准组合
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Optional


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    余弦退火 Warm Restarts 学习率调度器

    论文: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2016)
    链接: https://arxiv.org/abs/1608.03983

    核心思想:
    学习率按照余弦函数周期性变化，周期结束后重置。

    学习率公式:
        η_t = η_min + (η_max - η_min) * 0.5 * (1 + cos(π * t_cur / T_i))

    其中:
        - η_min: 最小学习率
        - η_max: 最大学习率 (初始学习率)
        - t_cur: 当前周期内的步数
        - T_i: 当前周期的长度
        - T_{i+1} = T_i * T_mult: 下一个周期的长度

    调度曲线示意图:
    η_max │╲                              ╱╲
          │ ╲                            ╱  ╲
          │  ╲                          ╱    ╲
          │   ╲                        ╱      ╲
    η_min │    ╲______________________╱        ╲______
          └─────────────────────────────────────────────
           T_0    T_0*T_mult    (T_0*T_mult)*T_mult

    特点:
    - 周期性重启帮助跳出局部最优
    - 余弦形状平滑衰减
    - 适用于长周期训练
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        """
        初始化调度器。

        Args:
            optimizer: 优化器实例
            T_0: 第一个周期的长度 (epochs 或 steps)
            T_mult: 周期增长因子，默认 1 (周期长度不变)
                   如果设为 2，周期长度将翻倍: T_0, 2*T_0, 4*T_0, ...
            eta_min: 最小学习率，默认 0
            last_epoch: 最后一个 epoch 的索引，默认 -1

        计算当前周期:
            if epoch < T_0:
                T_cur = epoch
                T_i = T_0
            else:
                # 计算当前属于第几个周期
                n = int(math.log((epoch / T_0) * (T_mult - 1) + 1, T_mult))
                T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
                T_i = T_0 * T_mult ** n
        """
        # 验证参数
        if T_0 <= 0:
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1:
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0

        super().__init__(optimizer, last_epoch)
        pass

    def get_lr(self) -> List[float]:
        """
        计算当前学习率。

        Returns:
            每个参数组的学习率列表

        计算步骤:
            Step 1: 确定当前周期信息
                    如果是第一次调用，直接返回初始学习率 base_lrs

            Step 2: 计算 t_cur (当前周期内的位置)
                    如果当前 epoch 超过第一个周期:
                        如果周期增长因子为 1 (固定周期长度):
                            使用取模运算计算周期内位置
                        否则 (周期长度增长):
                            计算当前属于第几个周期 n
                            计算当前周期内的位置 t_cur
                            计算当前周期长度 T_i
                    否则 (仍在第一个周期内):
                        t_cur 等于当前 epoch
                        T_i 等于 T_0

            Step 3: 计算余弦衰减因子
                    使用余弦函数计算衰减程度:
                        cosine_factor = 0.5 * (1 + cos(π * t_cur / T_i))
                    当 t_cur = 0 时，cosine_factor = 1 (最大学习率)
                    当 t_cur = T_i 时，cosine_factor = 0 (最小学习率)

            Step 4: 计算学习率
                    对每个参数组，根据余弦因子插值计算学习率:
                        lr = η_min + (base_lr - η_min) * cosine_factor
                    这实现了从 base_lr 到 η_min 的余弦衰减

            Step 5: 返回
                    返回每个参数组的学习率列表
        """
        pass


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + 余弦退火调度器

    现代 LLM 训练的标准学习率调度方案。

    三个阶段:
    1. Warmup (预热): 从 0 线性增加到初始学习率
    2. Stable (稳定): 可选的恒定学习率阶段
    3. Decay (衰减): 余弦退火衰减到最小学习率

    学习率曲线:
    lr    │              ╭──────╮
          │             ╱        ╲
          │            ╱          ╲
          │           ╱            ╲
          │          ╱              ╲
          │         ╱                ╲
          │        ╱                  ╲
          │       ╱                    ╲
          │      ╱                      ╲
          │     ╱                        ╲
          │    ╱                          ╲
          │   ╱                            ╲
    0     │──╱                              ╲──────
          └──────────────────────────────────────────
           warmup  stable  cosine decay

    为什么需要 Warmup:
    - 训练初期梯度大，高学习率导致不稳定
    - 逐步增加学习率让优化器状态稳定
    - 防止早期梯度爆炸
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_stable_steps: int = 0,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        初始化调度器。

        Args:
            optimizer: 优化器实例
            num_warmup_steps: warmup 步数
            num_training_steps: 总训练步数
            num_stable_steps: 稳定期步数 (恒定学习率)，默认 0
            num_cycles: 余弦周期数，默认 0.5 (半周期)
            min_lr_ratio: 最小学习率比例，默认 0.0
                         实际最小学习率 = base_lr * min_lr_ratio
            last_epoch: 最后一个 epoch 索引

        阶段划分:
            total_steps = num_warmup_steps + num_stable_steps + decay_steps
            decay_steps = num_training_steps - num_warmup_steps - num_stable_steps
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_stable_steps = num_stable_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio

        super().__init__(optimizer, last_epoch)
        pass

    def get_lr(self) -> List[float]:
        """
        计算当前学习率。

        计算逻辑:

        根据当前训练步数所处的阶段，使用不同的学习率计算策略:

        Warmup 阶段 (current_step < num_warmup_steps):
            学习率从 0 线性增加到初始学习率:
                lr = base_lr * (current_step / num_warmup_steps)

        稳定阶段 (current_step < num_warmup_steps + num_stable_steps):
            保持恒定的初始学习率:
                lr = base_lr

        余弦衰减阶段 (其他):
            首先计算衰减进度 (0 到 1 之间):
                progress = (current_step - warmup_steps - stable_steps) /
                          (total_steps - warmup_steps - stable_steps)

            然后应用余弦衰减公式:
                cosine_factor = 0.5 * (1 + cos(π * num_cycles * 2 * progress))
                lr = base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)

            这实现了从 base_lr 到 base_lr * min_lr_ratio 的余弦衰减

        Returns:
            每个参数组的学习率列表
        """
        pass

    def get_last_lr(self) -> List[float]:
        """
        返回最后计算的学习率。

        Returns:
            学习率列表
        """
        pass


class WarmupLinearScheduler(_LRScheduler):
    """
    Warmup + 线性衰减调度器

    类似于 WarmupCosine，但衰减阶段是线性的而非余弦。

    常用于 Transformer 训练的某些变体。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        初始化调度器。

        Args:
            optimizer: 优化器
            num_warmup_steps: warmup 步数
            num_training_steps: 总训练步数
            min_lr_ratio: 最小学习率比例
            last_epoch: 最后 epoch 索引
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        pass

    def get_lr(self) -> List[float]:
        """
        计算当前学习率。

        计算逻辑:
            根据当前步数所处阶段计算学习率:

            Warmup 阶段 (step < warmup):
                学习率从 0 线性增加到初始学习率:
                    lr = base_lr * (step / warmup)

            线性衰减阶段 (其他):
                首先计算衰减进度 (0 到 1 之间):
                    progress = (step - warmup) / (total - warmup)

                然后应用线性衰减:
                    lr = base_lr * (min_lr_ratio + (1 - min_lr_ratio) * (1 - progress))

                这实现了从 base_lr 到 base_lr * min_lr_ratio 的线性衰减
        """
        pass


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int = 0,
    num_training_steps: int = None,
    **kwargs
) -> _LRScheduler:
    """
    工厂函数，根据名称创建调度器。

    Args:
        name: 调度器名称，"cosine" / "linear" / "warmup_cosine"
        optimizer: 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        **kwargs: 额外参数

    Returns:
        调度器实例

    示例:
        scheduler = get_scheduler(
            "warmup_cosine",
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=100000,
        )
    """
    pass
