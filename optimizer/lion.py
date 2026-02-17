"""
Lion 优化器模块
实现 Lion (EvoLved Sign Momentum) 优化器。

Lion 特点:
- 仅使用第一阶动量 (momentum)
- 使用符号函数 (sign) 进行更新
- 相比 AdamW 内存占用更少
- 在某些任务上收敛更快

更新公式:
    c_t = β1 * m_{t-1} + (1 - β1) * g_t
    m_t = β2 * m_{t-1} + (1 - β2) * g_t
    update = sign(c_t) * lr + weight_decay * θ_{t-1}
    θ_t = θ_{t-1} - update

其中:
    - c_t: 中间变量
    - m_t: 动量
    - β1, β2: 动量系数 (默认 0.9, 0.99)
    - sign: 符号函数

论文: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
链接: https://arxiv.org/abs/2302.06675
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Optional


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) 优化器

    相比 AdamW:
    - 只存储一个动量状态 (AdamW 存两个)
    - 使用 sign 函数，更新更稀疏
    - 通常需要更大的学习率 (约 AdamW 的 1/10 ~ 1/3)

    推荐超参数:
    - lr: 3e-4 (AdamW 常用 1e-4)
    - β1: 0.9
    - β2: 0.99
    - weight_decay: 0.1 (AdamW 常用 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        初始化 Lion 优化器。

        Args:
            params: 可训练参数
            lr: 学习率 (通常比 AdamW 大 3-10 倍)
            betas: (β1, β2) 动量系数
            weight_decay: 权重衰减系数
        """
        # 验证参数
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        执行单步优化。

        Args:
            closure: 可选的闭包函数

        Returns:
            损失值 (如果提供了 closure)

        更新流程:
            for group in param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    Step 1: 获取梯度和状态
                            grad = p.grad
                            state = self.state[p]

                    Step 2: 初始化状态
                            if len(state) == 0:
                                state['step'] = 0
                                state['exp_avg'] = torch.zeros_like(p)  # m_t

                    Step 3: 获取超参数
                            beta1, beta2 = group['betas']
                            lr = group['lr']
                            weight_decay = group['weight_decay']

                    Step 4: 更新动量 (m_t)
                            使用指数移动平均更新动量:
                                m_t = β2 * m_{t-1} + (1 - β2) * g_t
                            其中 m_t 存储在 exp_avg 中

                    Step 5: 计算更新方向
                            计算中间变量 c_t，结合当前梯度和历史动量:
                                c_t = β1 * m_{t-1} + (1 - β1) * g_t
                            注意: 这里使用更新前的 exp_avg (即 m_{t-1})

                            对 c_t 应用符号函数得到更新方向:
                                update = sign(c_t)
                            sign 函数将正值映射为 1，负值映射为 -1，零映射为 0

                    Step 6: 应用权重衰减
                            Lion 的权重衰减在更新时应用:
                                θ_{t-1} = θ_{t-1} * (1 - lr * weight_decay)
                            这等价于: θ_{t-1} = θ_{t-1} - lr * weight_decay * θ_{t-1}

                    Step 7: 更新参数
                            沿符号梯度的方向更新参数:
                                θ_t = θ_{t-1} - lr * sign(c_t)

                    Step 8: 更新步数
                            将当前参数的步数计数器加 1

        与 AdamW 的关键区别:
        1. 只有一阶动量 (节省内存)
        2. 使用 sign 函数 (更新更稀疏)
        3. 权重衰减在更新时应用 (与 AdamW 类似)
        """
        pass

    def zero_grad(self, set_to_none: bool = False):
        """清零梯度。"""
        pass


class LionW(Lion):
    """
    Lion with decoupled weight decay (与 Lion 相同)

    Lion 本身已经使用了解耦的权重衰减，这个类只是为了命名清晰。
    """
    pass
