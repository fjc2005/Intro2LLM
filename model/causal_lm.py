"""
因果语言模型模块
实现完整的因果语言模型 (Causal Language Model)，用于文本生成。

模型结构:
    Input Tokens
         ↓
    Token Embedding + Position Encoding
         ↓
    [Transformer Block] × L
         ↓
    Final Norm
         ↓
    LM Head
         ↓
    Logits (vocab_size)

关键特性:
- 因果掩码: 确保模型只能看到当前位置及之前的 token
- KV 缓存: 加速自回归生成
- 温度采样: 控制生成的多样性
- Top-p 采样: 提高生成质量
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class CausalLMOutputWithPast:
    """
    因果语言模型的输出容器

    包含损失、logits、KV 缓存等信息，便于训练管道使用。

    Attributes:
        loss: 计算得到的损失值 (训练时)，标量张量或 None
        logits: 模型输出的 logits，[batch_size, seq_len, vocab_size]
        past_key_values: 每一层的 KV 缓存，List of Tuple (k, v)
        hidden_states: 可选，所有层的隐藏状态 (用于分析)
        attentions: 可选，所有层的注意力权重 (用于分析)
    """

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
    ):
        """
        初始化输出容器。

        Args:
            loss: 损失值，训练时计算，推理时为 None
            logits: 预测 logits，[batch, seq, vocab_size]
            past_key_values: 每层的 KV 缓存，用于生成加速
            hidden_states: 所有层的隐藏状态 (可选)
            attentions: 所有层的注意力权重 (可选)
        """
        pass


class CausalLM(nn.Module):
    """
    因果语言模型 (Causal Language Model)

    这是一个标准的 Transformer Decoder-only 架构，用于自回归文本生成。

    完整架构:
        ┌─────────────────────────────────────────────┐
        │  embed_tokens: Token 嵌入层                  │
        │  [vocab_size, hidden_size]                   │
        │                                              │
        │  layers: Transformer 块堆叠                 │
        │  [config.num_hidden_layers 个 TransformerBlock]│
        │                                              │
        │  norm: 最终层归一化                          │
        │  RMSNorm 或 LayerNorm                        │
        │                                              │
        │  lm_head: 语言模型头                         │
        │  [hidden_size, vocab_size]                   │
        └─────────────────────────────────────────────┘

    权重共享:
    - 通常 embed_tokens 和 lm_head 共享权重 (tie_weights)
    - 减少参数量，提高训练稳定性

    生成过程:
    1. 输入 prompt，得到第一个新 token 的 logits
    2. 采样得到新 token
    3. 将新 token 加入输入，重复直到满足停止条件
    4. 使用 KV 缓存避免重复计算
    """

    def __init__(self, config):
        """
        初始化因果语言模型。

        Args:
            config: ModelConfig 配置对象

        需要创建的子模块:
            - embed_tokens: TokenEmbedding 或 nn.Embedding
            - layers: nn.ModuleList，包含 num_hidden_layers 个 TransformerBlock
            - norm: 最终归一化层 (RMSNorm 或 LayerNorm)
            - lm_head: 输出投影，将 hidden_size 映射到 vocab_size

        可选:
            - position_embeddings: 如果不使用 RoPE，需要位置编码
        """
        super().__init__()
        # 保存配置
        # 初始化 embed_tokens
        # 初始化 layers (ModuleList 包含所有 TransformerBlock)
        # 初始化 norm (根据 config.use_rms_norm 选择)
        # 初始化 lm_head (nn.Linear 或共享 embed_tokens 权重)
        pass

    def get_input_embeddings(self) -> nn.Module:
        """
        获取输入嵌入层。

        Returns:
            embed_tokens 模块
        """
        pass

    def set_input_embeddings(self, value: nn.Module):
        """
        设置输入嵌入层。

        Args:
            value: 新的嵌入层
        """
        pass

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        前向传播。

        Args:
            input_ids: 输入 token ID，[batch_size, seq_len]
            attention_mask: 注意力掩码，[batch_size, seq_len]
                           1 表示有效 token，0 表示 padding
            position_ids: 位置 ID，[batch_size, seq_len]
                         如果为 None，自动使用 [0, 1, 2, ...]
            labels: 标签，用于计算损失，[batch_size, seq_len]
                   与 input_ids 相同，但 padding 位置为 -100
            past_key_values: 每层的 KV 缓存，用于生成加速
            use_cache: 是否使用 KV 缓存

        Returns:
            CausalLMOutputWithPast 对象

        计算流程:

            Step 1: 确定批次和序列维度
                    从输入张量中提取批次大小和序列长度
                    这两个维度将用于后续的形状变换

            Step 2: 获取嵌入向量
                    使用词嵌入层将输入的 token ID 转换为向量表示
                    每个 token ID 对应嵌入矩阵中的一行
                    形状变化: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]

            Step 3: 准备位置编码 (如果不使用 RoPE)
                    如果配置使用正弦位置编码而非 RoPE
                    将位置编码加到词嵌入上，注入位置信息

            Step 4: 准备注意力掩码
                    根据 attention_mask 创建因果掩码
                    因果掩码确保每个位置只能看到当前及之前的 token
                    掩码形状: [batch_size, 1, seq_len, total_len]
                    其中 total_len 包含当前序列长度和缓存的序列长度

                    因果掩码结构示意:
                    对角线及以下为 0 (允许关注)，对角线以上为负无穷 (禁止关注)
                    这样经过 softmax 后，被掩码的位置概率为 0

            Step 5: 通过所有 Transformer 层
                    初始化隐藏状态为嵌入向量
                    如果使用缓存，准备一个列表存储每层的 KV 缓存

                    遍历每一层 Transformer:
                    - 获取当前层的缓存 (如果存在)
                    - 将隐藏状态传入当前层，得到新的隐藏状态和当前层的 KV 缓存
                    - 如果使用缓存，将当前层的 KV 缓存保存

                    每一层包含自注意力和前馈网络，以及残差连接和归一化

            Step 6: 最终归一化
                    对所有层的输出应用最终的层归一化
                    这稳定了输出分布，有利于语言模型头的预测

            Step 7: 语言模型头
                    使用语言模型头将隐藏状态映射到词表空间
                    形状变化: [batch_size, seq_len, hidden_size] -
                              [batch_size, seq_len, vocab_size]
                    每个位置现在有一个分数向量，表示每个词的概率

            Step 8: 计算损失 (如果提供了 labels)
                    如果提供了训练标签，计算交叉熵损失:
                    - 将 logits 和 labels 都展平为二维张量
                    - 对齐维度: logits 变为 [batch*seq, vocab]
                    - labels 变为 [batch*seq]
                    - 忽略标签值为 -100 的位置 (padding)
                    - 计算预测分布与目标分布的差异

            Step 9: 返回结果
                    构建 CausalLMOutputWithPast 对象返回:
                    - loss: 计算得到的损失 (训练时)
                    - logits: 模型预测分数
                    - past_key_values: 各层的 KV 缓存 (用于生成加速)
        """
        pass

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        **kwargs
    ) -> dict:
        """
        为生成准备输入。

        在自回归生成中，每次只需要输入最后一个 token，
        之前的通过 KV 缓存获取。

        Args:
            input_ids: 当前输入 token ID
            past_key_values: 缓存的 KV

        Returns:
            准备好的输入字典
        """
        pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        自回归生成文本。

        Args:
            input_ids: 输入 prompt，[batch_size, prompt_len]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数，控制随机性
                        - 接近 0: 贪婪采样，选择概率最高的 token
                        - 1.0: 原始分布采样
                        - > 1: 更随机
            top_p: Nucleus sampling 阈值，只从累积概率 top_p 的 token 中采样
            top_k: Top-k 采样，只考虑概率最高的 k 个 token
            repetition_penalty: 重复惩罚，>1 降低重复 token 的概率
            pad_token_id: padding token ID
            eos_token_id: 结束 token ID，遇到则停止生成

        Returns:
            生成的完整序列，[batch_size, prompt_len + generated_len]

        生成流程:

            Step 1: 初始化生成状态
                    初始化 KV 缓存为空 (首次迭代会使用完整的 prompt)
                    获取批次大小
                    将输入的 prompt 作为已生成序列的初始值

            Step 2: 循环生成指定数量的新 token
                    对于每个生成步骤:

                    子步骤 1: 前向传播
                    - 第一次迭代输入完整的 prompt
                    - 后续迭代只输入最后一个生成的 token (利用 KV 缓存)
                    - 启用 KV 缓存以加速生成
                    - 从输出中提取最后一个位置对应的预测分数 (logits)
                    - 保存当前层的 KV 缓存供下次使用

                    子步骤 2: 温度缩放
                    - 将 logits 除以温度参数
                    - 温度控制分布的平滑程度:
                      * 温度趋近于 0: 分布趋于尖锐，接近贪婪采样
                      * 温度为 1: 保持原始分布
                      * 温度大于 1: 分布更平坦，增加随机性

                    子步骤 3: Top-k 过滤
                    - 如果启用 top-k，只保留概率最高的 k 个 token
                    - 将其他 token 的概率设为零 (通过设为负无穷实现)
                    - 这防止低概率的异常 token 被选中

                    子步骤 4: Top-p (Nucleus) 过滤
                    - 如果启用 top-p，按概率降序排序 token
                    - 计算累积概率，找到使累积概率首次超过 p 的最小 token 集合
                    - 只保留这个核心集合中的 token
                    - 这提供了比 top-k 更动态的截断方式

                    子步骤 5: 采样下一个 token
                    - 将过滤后的 logits 转换为概率分布 (softmax)
                    - 从该分布中随机采样一个 token
                    - 这引入了可控的随机性，使生成更多样化

                    子步骤 6: 更新生成序列
                    - 将新采样的 token 追加到已生成序列
                    - 为下次迭代准备输入 (仅新 token)

                    子步骤 7: 检查停止条件
                    - 如果生成了结束标记 (EOS) 且所有序列都生成，提前退出
                    - 这允许变长生成，不需要总是生成到最大长度

            Step 3: 返回完整序列
                    返回包含原始 prompt 和新生成 token 的完整序列
        """
        pass
