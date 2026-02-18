"""
模型配置模块
定义 LLM 的所有超参数配置，使用 dataclass 便于管理和序列化。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    大语言模型完整配置类。

    该类包含构建一个 Transformer Decoder 模型所需的全部超参数，
    涵盖模型结构、位置编码、归一化、激活函数等各个方面。

    Attributes:
        vocab_size: 词表大小，决定嵌入层和输出层的维度
        hidden_size: 隐藏层维度 (d_model)，Transformer 主维度的核心参数
        intermediate_size: FFN 中间层维度，通常为 2-4 倍 hidden_size
        num_hidden_layers: Transformer 层数 (L)
        num_attention_heads: 注意力头数，hidden_size 必须能被整除
        num_key_value_heads: GQA 分组数，用于减少 KV 缓存显存

        max_position_embeddings: 最大序列长度，影响位置编码和注意力掩码
        rope_theta: RoPE 基础频率，控制位置编码的旋转速度

        rms_norm_eps: RMSNorm 的 epsilon，防止除零
        attention_dropout: 注意力 dropout 概率
        hidden_act: 激活函数名称，"gelu" / "silu" / "swish" 等

        use_rms_norm: 是否使用 RMSNorm (否则使用 LayerNorm)
        use_rope: 是否使用 RoPE (否则使用 Sinusoidal)
        use_swiglu: 是否使用 SwiGLU (否则使用 GeGLU)

    References:
        - LLaMA: https://arxiv.org/abs/2302.13971
        - Qwen: https://arxiv.org/abs/2309.16609
    """

    # ============================================
    # 基础结构参数 (Core Architecture)
    # ============================================

    # 词表大小，决定 token 嵌入矩阵的行数
    # 例如: 32000 (LLaMA), 152064 (Qwen3)
    vocab_size: int

    # 隐藏层维度，Transformer 的核心维度 d_model
    # 所有层的输入输出都是这个维度
    hidden_size: int

    # FFN 中间层维度，通常是 hidden_size 的 2-4 倍
    # 在 Gated FFN (GeGLU/SwiGLU) 中，实际参数量是这个值的约 3 倍
    intermediate_size: int

    # Transformer 层数 L
    # 层数越多模型容量越大，但训练推理成本也越高
    num_hidden_layers: int

    # 注意力头数，hidden_size 必须能被 num_attention_heads 整除
    # 每个头的维度 head_dim = hidden_size // num_attention_heads
    num_attention_heads: int

    # KV 头数，用于 Grouped Query Attention (GQA)
    # 当 num_key_value_heads < num_attention_heads 时启用 GQA
    # 可以减少 KV 缓存的显存占用
    # 例如: num_attention_heads=32, num_key_value_heads=4 表示每 8 个 Q 头共享 1 个 KV 头
    num_key_value_heads: int

    # ============================================
    # 位置编码参数 (Positional Encoding)
    # ============================================

    # 模型支持的最大序列长度
    # 超过此长度的序列需要截断或使用其他处理
    max_position_embeddings: int

    # RoPE 基础频率 theta，默认 10000.0
    # 较小的值 (如 10000) 适合短序列
    # 较大的值 (如 1000000) 可以支持更长的上下文 (如 CodeLlama)
    rope_theta: float

    # ============================================
    # 归一化与正则化 (Normalization & Regularization)
    # ============================================

    # RMSNorm 的 epsilon，用于数值稳定性，防止除以零
    # 通常设为 1e-6
    rms_norm_eps: float

    # 注意力层的 dropout 概率
    # 训练时使用，推理时应该设为 0
    attention_dropout: float

    # 隐藏层激活函数类型
    # "gelu": GELU 激活
    # "silu" 或 "swish": SiLU/Swish 激活，用于 SwiGLU
    hidden_act: str

    # ============================================
    # 架构开关选项 (Architecture Toggles)
    # ============================================

    # 是否使用 RMSNorm
    # True: 使用 RMSNorm (现代 LLM 如 LLaMA、Qwen 使用)
    # False: 使用传统 LayerNorm
    use_rms_norm: bool

    # 是否使用 RoPE (旋转位置编码)
    # True: 使用 RoPE (现代 LLM 标准)
    # False: 使用 Sinusoidal 位置编码 (原始 Transformer)
    use_rope: bool

    # 是否使用 SwiGLU 作为 FFN 激活
    # True: 使用 SwiGLU (SiLU 作为门控)
    # False: 使用 GeGLU (GELU 作为门控)
    use_swiglu: bool

    # ============================================
    # 便捷属性 (Convenienced Properties)
    # ============================================

    @property
    def head_dim(self) -> int:
        """
        计算每个注意力头的维度。

        head_dim 等于 hidden_size 除以 num_attention_heads

        这是注意力计算中 Q、K、V 向量的实际维度。
        所有注意力头的 head_dim 必须相等。
        """
        pass

    @property
    def num_key_value_groups(self) -> int:
        """
        计算 GQA 中的分组数。

        num_key_value_groups 等于 num_attention_heads 除以 num_key_value_heads

        表示每个 KV 头被多少个 Q 头共享。
        例如: 32 个 Q 头，4 个 KV 头，则每组 8 个 Q 头共享 1 个 KV 头。
        """
        pass
