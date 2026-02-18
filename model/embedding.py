"""
嵌入与位置编码模块
包含 Token 嵌入、Sinusoidal 位置编码和 RoPE (旋转位置编码)。

位置编码的作用:
- Transformer 本身对位置不敏感 (Self-Attention 是置换等变的)
- 位置编码为模型提供序列顺序信息

两种主要方案:
1. Sinusoidal: 原始 Transformer 使用，固定的三角函数编码
2. RoPE: 现代 LLM 标准，通过旋转矩阵注入位置信息
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class TokenEmbedding(nn.Module):
    """
    Token 嵌入层

    将离散的 token ID 转换为连续的向量表示。
    这是模型的入口，每个 token 对应一个可学习的向量。

    嵌入本质上是一个可学习的查找表 (Lookup Table)：
        - 形状：[vocab_size, hidden_size]
        - 第 i 行表示第 i 个 token 的向量表示
        - 这个矩阵是模型的参数，通过反向传播学习得到

    torch已经给出了预定义的nn.Embedding接口，但是为了深入理解原理,
    我们在这个项目中要手动实现Embedding Layer：
        - 使用 nn.Parameter 创建可学习参数矩阵
        - 使用索引操作手动实现查表

    属性:
        vocab_size: 词表大小
        hidden_size: 嵌入维度
        embedding_table: 形状 [vocab_size, hidden_size] 的嵌入矩阵
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        """
        初始化 Token 嵌入层。

        这个函数需要实现的内容有：
        1. 调用父类的初始化方法
        2. 创建一个形状为 [vocab_size, hidden_size] 的嵌入矩阵作为参数
        3. 使用 nn.Parameter 包装，使其成为可学习的模型参数
        4. 对嵌入矩阵进行初始化，通常使用正态分布，均值为0，标准差为0.02

        注意：不要使用 nn.Embedding，而是直接创建一个 nn.Parameter。

        Args:
            vocab_size: 词表大小
            hidden_size: 嵌入维度
        """
        super().__init__()
        pass

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        将 token ID 转换为嵌入向量。

        手动实现查表操作：
        1. 接收形状为 [batch_size, seq_len] 的 token ID 张量
        2. 使用索引操作 self.embedding_table[input_ids] 从嵌入矩阵中提取对应向量
        3. 返回形状为 [batch_size, seq_len, hidden_size] 的嵌入向量

        注意：不要调用 nn.Embedding 的 forward，而是手动使用索引查表。

        Args:
            input_ids: token ID，形状 [batch_size, seq_len]

        Returns:
            嵌入向量，形状 [batch_size, seq_len, hidden_size]
        """
        pass


class PositionalEncoding(nn.Module):
    """
    Sinusoidal 位置编码 (正弦/余弦位置编码)

    论文: "Attention Is All You Need" (Vaswani et al., 2017)
    这是原始 Transformer 使用的位置编码方式。

    编码公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中:
        - pos: 位置索引 (0, 1, 2, ...)
        - i: 维度索引
        - d_model: 模型维度

    特点:
    1. 固定的编码，不需要学习
    2. 可以外推到训练时未见过的长度
    3. 相对位置可以通过线性变换得到
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化 Sinusoidal 位置编码。

        Args:
            d_model: 模型维度 (hidden_size)
            max_len: 最大序列长度，预设的缓冲区大小

        预计算:
            创建形状 [max_len, d_model] 的位置编码矩阵 pe
            pe[pos, :] 表示位置 pos 的编码向量
        """
        super().__init__()
        # 创建位置编码缓冲区，形状 [max_len, d_model]
        # 使用 register_buffer 使其不参与梯度更新
        pass

    def _create_pe(self, d_model: int, max_len: int) -> torch.Tensor:
        """
        创建位置编码矩阵。

        计算步骤:
            Step 1: 创建一个从 0 到 max_len-1 的位置索引向量
                    形状 [max_len, 1]

            Step 2: 创建维度除数 div_term
                    公式: div_term = 10000 的 -2i/d_model 次方，其中 i = 0, 2, 4, ...
                    形状 [d_model/2]

            Step 3: 计算 sin 和 cos 编码
                    对偶数位置 (2i) 应用正弦编码: sin(position * div_term)
                    对奇数位置 (2i+1) 应用余弦编码: cos(position * div_term)

        Returns:
            位置编码矩阵，形状 [max_len, d_model]
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入添加位置编码。

        Args:
            x: 输入张量，形状 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量，形状同输入

        操作: x = x + pe[:seq_len, :]
              利用广播，pe 加到 batch 中的每个样本
        """
        pass


class RoPE(nn.Module):
    """
    RoPE (Rotary Position Embedding，旋转位置编码)

    论文: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
          (Su et al., 2021)
    链接: https://arxiv.org/abs/2104.09864

    核心思想:
    通过旋转矩阵将位置信息编码到 Q、K 向量中，使得内积结果只与相对位置有关。

    数学原理:
    对于二维向量 (x1, x2)，位置 m 的旋转编码为:
        [cos(m*theta)  -sin(m*theta)]   [x1]
        [sin(m*theta)   cos(m*theta)] * [x2]

    在高维中，每两个维度组成一对进行旋转，不同维度对使用不同的 theta:
        theta_i = base^(-2i/d)  其中 i 是维度对索引

    优势:
    1. 相对位置编码: <f(q,m), f(k,n)> 只依赖于 (m-n)
    2. 长度外推性好
    3. 现代 LLM 标准 (LLaMA、Qwen、Mistral 等)
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        初始化 RoPE。

        Args:
            dim: 每个注意力头的维度 (head_dim)
            max_position_embeddings: 最大位置，用于预计算
            base: 基础频率，控制旋转角度随位置的变化速度

        预计算:
            inv_freq: 频率倒数，形状 [dim/2]
            公式: inv_freq[i] = base 的 -2i/dim 次方分之一
        """
        super().__init__()
        # 预计算 inv_freq (频率倒数)
        # 形状 [dim // 2]
        pass

    def _compute_cos_sin(self, position_ids: torch.Tensor, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算位置的 cos 和 sin 值。

        Args:
            position_ids: 位置 ID，形状 [batch_size, seq_len] 或 [seq_len]
            seq_len: 序列长度
            device: 计算设备

        计算步骤:
            Step 1: 计算频率 freqs
                    将位置 ID 与频率倒数进行外积运算
                    广播到形状 [batch, seq_len, dim/2]

            Step 2: 将 freqs 扩展为复数形式
                    为了同时计算 sin 和 cos，将频率复制拼接
                    形状变为 [batch, seq_len, dim]

            Step 3: 计算 cos 和 sin
                    对频率分别计算余弦和正弦值
                    形状 [batch, seq_len, dim]

        Returns:
            cos, sin 两个张量
        """
        pass

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        将张量的后一半维度旋转。

        这是 RoPE 的核心操作，实现将向量后半部分取负并与前半部分交换位置。

        Args:
            x: 输入张量，形状 [..., dim]

        Returns:
            旋转后的张量，形状同输入

        操作:
            Step 1: 将最后一维分成前后两半
                    前半部分 x1 和后半部分 x2

            Step 2: 构造旋转后的向量
                    将后半部分取负，然后与前半部分拼接
                    旋转后的向量形式为 [-x2, x1]
        """
        pass

    @staticmethod
    def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                             cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE 到 Q 和 K。

        公式:
            q_rot = q * cos + rotate_half(q) * sin
            k_rot = k * cos + rotate_half(k) * sin

        Args:
            q: Query 张量，形状 [batch, num_heads, seq_len, head_dim]
            k: Key 张量，形状 [batch, num_kv_heads, seq_len, head_dim]
            cos: 余弦值，形状 [batch, seq_len, head_dim]
            sin: 正弦值，形状 [batch, seq_len, head_dim]

        Returns:
            (q_rot, k_rot): 旋转后的 Q 和 K

        操作步骤:
            Step 1: 扩展 cos/sin 以匹配 Q/K 的 heads 维度
                    在 heads 维度添加大小为 1 的维度以便广播
                    cos 形状从 [batch, seq, head_dim] 变为 [batch, 1, seq, head_dim]

            Step 2: 应用旋转公式
                    对 Query: q * cos + rotate_half(q) * sin
                    对 Key: k * cos + rotate_half(k) * sin
                    这里的 rotate_half 是将向量后半部分旋转 180 度

        边界条件:
            - cos/sin 的 batch 维度通过广播匹配
            - cos/sin 的 heads 维度为 1，通过广播扩展到 num_heads
        """
        pass

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 应用旋转位置编码。

        Args:
            q: Query，形状 [batch, num_heads, seq_len, head_dim]
            k: Key，形状 [batch, num_kv_heads, seq_len, head_dim]
            position_ids: 位置 ID，形状 [batch, seq_len] 或 [seq_len]
                          如果为 None，使用默认位置 [0, 1, 2, ...]

        Returns:
            (q_embed, k_embed): 应用 RoPE 后的 Q 和 K

        流程:
            Step 1: 获取序列长度和设备信息
            Step 2: 计算 cos 和 sin
            Step 3: 调用 apply_rotary_pos_emb 应用旋转
        """
        pass
