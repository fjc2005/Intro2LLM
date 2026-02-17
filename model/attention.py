"""
注意力模块
包含 Multi-Head Attention (MHA) 和 Grouped Query Attention (GQA)。

注意力机制是 Transformer 的核心，负责捕捉序列中 token 之间的关系。

MHA: 每个注意力头有独立的 Q、K、V 投影
GQA: 多个 Q 头共享一组 KV 头，减少 KV 缓存显存
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力 (Multi-Head Attention, MHA)

    论文: "Attention Is All You Need" (Vaswani et al., 2017)

    核心公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    多头版本:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
        where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)

    特点:
    - 每个头有独立的 Q、K、V 投影矩阵
    - num_heads * head_dim = hidden_size
    - 适用于需要精细注意力的场景
    - KV 缓存占用较大 (每个头都要存 KV)
    """

    def __init__(self, config):
        """
        初始化 MHA 层。

        Args:
            config: 模型配置，包含:
                - hidden_size: 隐藏层维度
                - num_attention_heads: 注意力头数
                - attention_dropout: dropout 概率

        需要创建的参数:
            - q_proj: Query 投影，[hidden_size, hidden_size]
            - k_proj: Key 投影，[hidden_size, hidden_size]
            - v_proj: Value 投影，[hidden_size, hidden_size]
            - o_proj: 输出投影，[hidden_size, hidden_size]

        注意:
            由于每个头独立的 KV，num_key_value_heads = num_attention_heads
        """
        super().__init__()
        # 从配置中提取参数
        # 验证 hidden_size 能被 num_heads 整除
        # 初始化四个线性投影层
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        前向传播。

        Args:
            hidden_states: 输入张量，[batch_size, seq_len, hidden_size]
            position_ids: 位置 ID，用于 RoPE，[batch_size, seq_len] 或 [seq_len]
            past_key_value: 缓存的 KV，用于生成阶段加速
                           Tuple of (past_key, past_value)
                           形状: ([batch, num_heads, cache_len, head_dim],
                                  [batch, num_heads, cache_len, head_dim])
            attention_mask: 注意力掩码，[batch, 1, seq_len, total_len]
                           用于处理 padding 和因果掩码
            use_cache: 是否返回 KV 缓存，用于自回归生成

        Returns:
            attn_output: 注意力输出，[batch_size, seq_len, hidden_size]
            present_key_value: 新的 KV 缓存 (如果 use_cache=True)

        计算流程:
            Step 1: 线性投影得到 Q、K、V
                    使用三个独立的线性变换将隐藏状态投影到 Query、Key、Value 空间
                    每个投影将 [batch, seq, hidden] 映射到 [batch, seq, hidden]
                    这些投影矩阵是可学习的参数

            Step 2: reshape 为多头形式
                    将隐藏维度分割成 num_heads 个头，每个头维度为 head_dim
                    形状变换: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
                    其中 head_dim = hidden_size // num_heads

            Step 3: 应用 RoPE (如果使用)
                    如果配置使用旋转位置编码，对 Q 和 K 应用 RoPE
                    这通过旋转矩阵注入位置信息

            Step 4: 处理 KV 缓存 (如果提供)
                    在生成阶段，将之前缓存的 Key 和 Value 与当前的拼接
                    沿序列长度维度拼接: [cache_len] + [seq_len] = [total_len]
                    形状: [batch, num_heads, cache_len + seq_len, head_dim]

            Step 5: 计算注意力分数
                    使用缩放点积计算注意力分数:
                    数学公式: scores = (Q · K^T) / sqrt(d_k)
                    其中 d_k 是每个头的维度
                    形状: [batch, num_heads, seq_len, total_len]

            Step 6: 应用注意力掩码
                    将注意力掩码加到分数上，掩码中无效位置为负无穷
                    这样 softmax 后这些位置的权重会变为零

            Step 7: softmax 得到注意力权重
                    对分数应用 softmax 函数，沿最后一个维度 (key 维度) 归一化
                    将分数转换为概率分布，所有权重之和为 1

            Step 8: 应用 dropout (训练时)
                    在训练阶段，随机将部分注意力权重置零以防止过拟合
                    推理阶段不使用 dropout

            Step 9: 计算注意力输出
                    使用注意力权重对 Value 进行加权求和:
                    数学公式: output = weights · V
                    形状: [batch, num_heads, seq_len, head_dim]

            Step 10: reshape 回原始维度
                    将多头结果合并回隐藏维度:
                    [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]

            Step 11: 输出投影
                    使用输出投影矩阵将结果映射回隐藏维度
                    形状: [batch, seq_len, hidden_size]

        边界条件:
            - 训练阶段 use_cache=False，past_key_value=None
            - 生成阶段 use_cache=True，需要提供 past_key_value
            - causal mask 通过 attention_mask 实现，上三角为负无穷，防止看到未来信息
        """
        pass


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention, GQA)

    论文: "GQA: Training Generalized Multi-Query Transformer Models"
          (Ainslie et al., 2023)
    链接: https://arxiv.org/abs/2305.13245

    动机:
    标准 MHA 的 KV 缓存占用显存大，Multi-Query Attention (MQA) 让所有 Q 头共享
    一组 KV 头可以减少缓存，但会损失质量。GQA 是折中方案。

    核心思想:
    - 将 num_attention_heads 分成 num_key_value_heads 组
    - 每组内的 Q 头共享同一对 KV 头
    - 减少 KV 缓存的同时保持较好的注意力质量

    计算方式:
    1. Q 投影: 保持 num_attention_heads 个独立的 Q
    2. K/V 投影: 只投影到 num_key_value_heads 个 K 和 V
    3. 计算时，将 K 和 V 复制 (repeat) num_groups 次以匹配 Q 头数

    缓存优势:
    KV 缓存从 [batch, num_heads, seq, head_dim] 减少到
              [batch, num_kv_heads, seq, head_dim]
    当 num_kv_heads << num_heads 时，显存显著节省
    """

    def __init__(self, config):
        """
        初始化 GQA 层。

        Args:
            config: 模型配置，包含:
                - hidden_size: 隐藏层维度
                - num_attention_heads: 注意力头数
                - num_key_value_heads: KV 头数 (GQA 分组数)
                - attention_dropout: dropout 概率

        需要创建的参数:
            - q_proj: Query 投影，[hidden_size, hidden_size]
            - k_proj: Key 投影，[hidden_size, num_kv_heads * head_dim]
            - v_proj: Value 投影，[hidden_size, num_kv_heads * head_dim]
            - o_proj: 输出投影，[hidden_size, hidden_size]

        注意:
            K/V 投影的输出维度是 num_kv_heads * head_dim，小于 hidden_size
        """
        super().__init__()
        # 从配置中提取参数
        # 计算 num_key_value_groups = num_attention_heads // num_key_value_heads
        # 验证整除关系
        # 初始化 Q、K、V、O 投影层，注意 K、V 的输出维度
        pass

    def repeat_kv(self, x: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        复制 KV 头以匹配 Q 头数量。

        这是 GQA 的核心操作。当 num_kv_heads < num_heads 时，需要将每个 KV 头
        复制 num_groups 次，使得 KV 头数与 Q 头数相同，才能进行标准注意力计算。

        Args:
            x: KV 张量，形状 [batch, num_kv_heads, seq_len, head_dim]
            num_groups: 复制次数，等于 num_attention_heads // num_key_value_heads

        Returns:
            复制后的张量，形状 [batch, num_attention_heads, seq_len, head_dim]

        操作步骤:
            Step 1: 扩展维度
                    在 KV 头维度之后插入一个新维度，用于后续复制
                    形状变化: [batch, num_kv_heads, seq, head_dim] -
                              [batch, num_kv_heads, 1, seq, head_dim]

            Step 2: 复制
                    沿新插入的维度将 KV 头复制 num_groups 次
                    这样每个原始 KV 头会被复制成 num_groups 个副本
                    形状变化: [batch, num_kv_heads, 1, seq, head_dim] -
                              [batch, num_kv_heads, num_groups, seq, head_dim]

            Step 3: reshape
                    合并 KV 头维度和复制维度，使总头数等于注意力头数
                    形状变化: [batch, num_kv_heads, num_groups, seq, head_dim] -
                              [batch, num_attention_heads, seq, head_dim]
                    其中 num_attention_heads = num_kv_heads * num_groups

        示例:
            输入:  [batch, 4, seq, 64]  (4 个 KV 头)
            num_groups = 2 (表示每个 KV 头被 2 个 Q 头共享)
            输出: [batch, 8, seq, 64]  (8 个 Q 头，每个 Q 头对应一个复制的 KV 头)
        """
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        前向传播。

        Args 和 Returns: 同 MultiHeadAttention

        计算流程 (与 MHA 主要区别在于 Step 2 和 Step 5):

            Step 1: 线性投影
                    Query 投影到完整隐藏维度 [batch, seq, hidden]
                    Key 和 Value 投影到缩减维度 [batch, seq, num_kv_heads * head_dim]
                    注意 K/V 的投影输出维度小于 Q，这是 GQA 的核心

            Step 2: reshape 为多头形式
                    Query: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
                    Key/Value: [batch, seq, num_kv_heads * head_dim] ->
                               [batch, num_kv_heads, seq, head_dim]
                    注意 K/V 只有 num_kv_heads 个头，而 Q 有 num_heads 个

            Step 3: 应用 RoPE
                    如果配置使用旋转位置编码，对 Q 和 K 应用 RoPE

            Step 4: 处理 KV 缓存 (如果提供)
                    在生成阶段，将之前缓存的 Key 和 Value 与当前的拼接
                    沿序列长度维度拼接
                    形状: [batch, num_kv_heads, cache_len + seq_len, head_dim]

            Step 5: 复制 KV 以匹配 Q 头数 (GQA 关键步骤)
                    由于 K/V 头数少于 Q 头数，需要将每个 K/V 头复制 num_groups 次
                    num_groups = num_attention_heads // num_key_value_heads
                    复制后 K/V 形状: [batch, num_heads, total_len, head_dim]

            Step 6-9: 注意力计算 (同 MHA)
                    计算缩放点积注意力分数: scores = (Q · K^T) / sqrt(d_k)
                    应用注意力掩码
                    使用 softmax 归一化得到注意力权重
                    应用 dropout (训练时)
                    计算加权输出: output = weights · V

            Step 10-11: reshape 和输出投影 (同 MHA)
                    合并多头结果并应用输出投影

        显存优化:
            KV 缓存只存储 num_kv_heads 个头，而不是 num_heads 个
            这显著减少了自回归生成时的显存占用
            当 num_kv_heads = 1 时，退化为 MQA (Multi-Query Attention)
            当 num_kv_heads = num_heads 时，退化为标准 MHA
        """
        pass
