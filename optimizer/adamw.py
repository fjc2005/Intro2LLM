"""
AdamW 优化器模块
实现 Adam + 解耦权重衰减的优化器。

Adam (Adaptive Moment Estimation):
- 结合 Momentum 和 RMSprop 的优点
- 使用一阶矩估计 (momentum) 和二阶矩估计 (adaptive learning rate)

AdamW 改进:
- 将权重衰减 (L2 regularization) 从梯度计算中解耦
- 权重衰减直接应用于参数更新，不参与自适应学习率计算
- 泛化性能更好，是现代 LLM 训练的标准优化器
"""

import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional, Tuple, Any


class AdamW(Optimizer):
    """
    AdamW 优化器

    论文: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    链接: https://arxiv.org/abs/1711.05101

    Adam 与 AdamW 的区别:

    L2 Regularization (Adam):
        gradient = gradient + weight_decay * parameter
        # 权重衰减会影响自适应学习率

    Decoupled Weight Decay (AdamW):
        parameter = parameter - lr * weight_decay * parameter
        # 权重衰减与学习率解耦

    为什么 AdamW 更好:
    1. 权重衰减系数独立于学习率调度
    2. 大学习率时不会过度正则化
    3. 实验上泛化性能更好

    参数更新公式:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t          # 一阶矩估计 (momentum)
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2        # 二阶矩估计 (velocity)

        m̂_t = m_t / (1 - β1^t)                       # 偏差修正
        v̂_t = v_t / (1 - β2^t)                       # 偏差修正

        θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)      # Adam 更新
        θ_t = θ_t - lr * λ * θ_{t-1}                  # 解耦权重衰减

    其中:
        - g_t: 第 t 步的梯度
        - m_t: 一阶矩 (动量)
        - v_t: 二阶矩 (二阶动量)
        - β1, β2: 衰减系数 (默认 0.9, 0.999)
        - ε: 数值稳定常数 (默认 1e-8)
        - λ: 权重衰减系数 (默认 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
    ):
        """
        初始化 AdamW 优化器。

        Args:
            params: 可训练参数迭代器
            lr: 学习率，默认 1e-3
            betas: (β1, β2) 衰减系数，默认 (0.9, 0.999)
            eps: 数值稳定常数，默认 1e-8
            weight_decay: 权重衰减系数，默认 0.01
            correct_bias: 是否进行偏差修正，默认 True

        参数状态初始化:
            为每个参数创建:
            - exp_avg: 一阶矩估计 m，初始化为 0
            - exp_avg_sq: 二阶矩估计 v，初始化为 0
            - step: 当前步数 t，初始化为 0
        """
        # 验证参数
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 调用父类初始化
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)
        pass

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        执行单步优化。

        Args:
            closure: 可选的闭包函数，用于重新计算损失

        Returns:
            如果提供了 closure，返回损失值

        更新流程 (对每个参数组):

            for group in param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    Step 1: 获取梯度
                            grad = p.grad
                            如果 grad 是稀疏的，需要特殊处理 (跳过或 densify)

                    Step 2: 获取或初始化状态
                            state = self.state[p]
                            if len(state) == 0:
                                state['step'] = 0
                                state['exp_avg'] = torch.zeros_like(p)      # m_0
                                state['exp_avg_sq'] = torch.zeros_like(p)   # v_0

                    Step 3: 获取状态变量
                            exp_avg = state['exp_avg']       # m_{t-1}
                            exp_avg_sq = state['exp_avg_sq'] # v_{t-1}
                            beta1, beta2 = group['betas']

                    Step 4: 更新一阶矩估计 (momentum)
                            使用指数移动平均更新一阶矩:
                                m_t = β1 * m_{t-1} + (1 - β1) * grad
                            其中 m_t 存储在 exp_avg 中

                    Step 5: 更新二阶矩估计 (adaptive lr)
                            使用指数移动平均更新二阶矩:
                                v_t = β2 * v_{t-1} + (1 - β2) * grad^2
                            其中 v_t 存储在 exp_avg_sq 中

                    Step 6: 更新步数
                            将当前参数的步数计数器加 1

                    Step 7: 偏差修正 (可选)
                            如果启用偏差修正:
                                计算一阶矩和二阶矩的偏差修正系数:
                                    bias_correction1 = 1 - β1^step
                                    bias_correction2 = 1 - β2^step
                                调整步长:
                                    step_size = lr / bias_correction1
                                调整二阶矩:
                                    denom = (√v_t / √bias_correction2) + ε
                            否则:
                                step_size = lr
                                denom = √v_t + ε

                    Step 8: Adam 更新 (自适应学习率部分)
                            使用自适应学习率更新参数:
                                θ_t = θ_{t-1} - step_size * m̂_t / (√v̂_t + ε)
                            其中 m̂_t 和 v̂_t 是偏差修正后的一阶矩和二阶矩

                    Step 9: 解耦权重衰减
                            如果权重衰减系数不为零:
                                直接对参数应用权重衰减:
                                    θ_t = θ_t - lr * λ * θ_{t-1}
                            注意: 使用当前学习率 lr，而非自适应学习率

        关键区别总结:
        - Adam: weight_decay 参与梯度计算，影响自适应学习率
        - AdamW: weight_decay 在更新后应用，与学习率解耦
        """
        pass

    def zero_grad(self, set_to_none: bool = False):
        """
        清零梯度。

        Args:
            set_to_none: 如果 True，将梯度设为 None 而不是 0
                        可以节省内存，但在某些情况下需要梯度为 0
        """
        pass

    def get_lr(self) -> List[float]:
        """
        获取当前学习率。

        Returns:
            每个参数组的学习率列表
        """
        pass

    def set_lr(self, lr: float):
        """
        设置学习率。

        Args:
            lr: 新的学习率
        """
        pass


class AdamW8bit(AdamW):
    """
    8-bit AdamW 优化器

    使用 bitsandbytes 库实现 8-bit 量化优化器状态。
    可以显著减少优化器状态的显存占用 (约 75%)。

    适用场景:
    - 大模型训练时显存不足
    - 需要增加 batch size 或序列长度

    注意:
    - 需要安装 bitsandbytes
    - 可能略微影响收敛速度
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 8-bit AdamW。

        需要:
            from bitsandbytes.optim import AdamW8bit as BNBAdamW8bit
            # 或自己实现量化逻辑
        """
        super().__init__(*args, **kwargs)
        pass
