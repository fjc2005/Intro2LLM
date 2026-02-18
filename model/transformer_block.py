"""
Transformer 解码器块模块
实现完整的 Transformer Decoder Layer，包含注意力子层和前馈子层。

架构设计:
- 采用 Pre-LayerNorm (Pre-RMSNorm) 结构
- 每个子层前有归一化，后有残差连接
- 这是现代 LLM 的标准设计，训练更稳定

结构对比:
- Pre-LN:  Norm -> Sublayer -> Add  (现代标准)
- Post-LN: Sublayer -> Add -> Norm (原始 Transformer)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TransformerBlock(nn.Module):
    """
    Transformer 解码器块 (Decoder Block)

    这是构建大语言模型的基本单元，由两个子层组成:
    1. 自注意力子层: 捕捉序列内 token 之间的关系
    2. 前馈网络子层: 对每个位置独立进行非线性变换

    Pre-LayerNorm 结构:
        ┌─────────────────────────────────────────────┐
        │  Input: hidden_states                        │
        │        ↓                                     │
        │  residual = hidden_states                    │
        │        ↓                                     │
        │  hidden_states = Norm(hidden_states)         │
        │        ↓                                     │
        │  hidden_states = Attention(hidden_states)    │
        │        ↓                                     │
        │  hidden_states = residual + hidden_states    │
        │        ↓                                     │
        │  residual = hidden_states                    │
        │        ↓                                     │
        │  hidden_states = Norm(hidden_states)         │
        │        ↓                                     │
        │  hidden_states = FFN(hidden_states)          │
        │        ↓                                     │
        │  hidden_states = residual + hidden_states    │
        │        ↓                                     │
        │  Output: hidden_states                       │
        └─────────────────────────────────────────────┘

    残差连接的作用:
    - 缓解梯度消失，使深层网络可训练
    - 提供恒等映射捷径，信息可以直接传递

    为什么选择 Pre-LN:
    - Post-LN 深层网络训练不稳定
    - Pre-LN 在极深网络 (如 96 层) 上训练更稳定
    - 学习率可以设置得更大
    """

    def __init__(self, config, layer_idx: int = None):
        """
        初始化 Transformer 块。

        Args:
            config: 模型配置，包含:
                - hidden_size: 隐藏层维度
                - use_rms_norm: 是否使用 RMSNorm
                - rms_norm_eps: 归一化 epsilon
                - attention_dropout: 注意力 dropout
            layer_idx: 当前层的索引 (0, 1, 2, ...)，用于调试和标识

        需要创建的子模块:
            - input_layernorm: 注意力前的归一化
            - self_attn: 自注意力层 (MHA 或 GQA)
            - post_attention_layernorm: FFN 前的归一化
            - mlp: 前馈网络 (GeGLU 或 SwiGLU)
        """
        super().__init__()
        # 保存 layer_idx 用于标识当前层
        # 根据 config.use_rms_norm 选择 LayerNorm 或 RMSNorm
        # 初始化 input_layernorm (Pre-Attention Norm)
        # 初始化 self_attn (根据 config 选择 MHA 或 GQA)
        # 初始化 post_attention_layernorm (Pre-FFN Norm)
        # 初始化 mlp (根据 config 选择 SwiGLU 或 GeGLU)
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        前向传播。

        Args:
            hidden_states: 输入张量，[batch_size, seq_len, hidden_size]
            position_ids: 位置 ID，用于 RoPE，[batch_size, seq_len]
            attention_mask: 注意力掩码，[batch, 1, seq_len, total_len]
                           包含因果掩码和 padding 掩码
            past_key_value: 缓存的 KV，用于生成阶段，Tuple of (past_k, past_v)
            use_cache: 是否返回 KV 缓存

        Returns:
            hidden_states: 输出张量，[batch_size, seq_len, hidden_size]
            present_key_value: 新的 KV 缓存 (如果 use_cache=True)

        计算流程 (详细步骤):

            # ========== 自注意力子层 ==========
            Step 1: 保存残差
                    在归一化前保存输入，作为残差连接的基础
                    形状: [batch, seq_len, hidden_size]

            Step 2: Pre-Attention 归一化
                    在进入注意力子层前进行归一化 (Pre-LN 结构)
                    形状: [batch, seq_len, hidden_size]

            Step 3: 自注意力计算
                    通过自注意力机制处理输入，实现 token 间信息交互
                    hidden_states 形状: [batch, seq_len, hidden_size]
                    present_kv: 返回的 KV 缓存，用于自回归生成

            Step 4: 残差连接
                    将注意力子层的输出与原始输入相加
                    这允许梯度直接流过恒等路径，帮助训练深层网络

            # ========== 前馈网络子层 ==========
            Step 5: 保存残差
                    在归一化前保存输入，作为残差连接的基础
                    形状: [batch, seq_len, hidden_size]

            Step 6: Pre-FFN 归一化
                    在进入前馈网络前进行归一化
                    形状: [batch, seq_len, hidden_size]

            Step 7: 前馈网络计算
                    通过前馈网络对每个位置进行非线性变换
                    形状: [batch, seq_len, hidden_size]

            Step 8: 残差连接
                    将前馈网络的输出与原始输入相加

            Step 9: 返回结果
                    return hidden_states, present_kv

        注意事项:
            - 所有形状中的 hidden_size 是模型的隐藏层维度
            - past_key_value 在训练时为 None，在生成时非空
            - 两个子层都有独立的残差连接
        """
        pass
