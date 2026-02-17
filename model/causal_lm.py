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

            # Step 1: 确定 batch_size 和 seq_len
                    batch_size, seq_len = input_ids.shape

            # Step 2: 获取嵌入
                    inputs_embeds = embed_tokens(input_ids)
                    形状: [batch_size, seq_len, hidden_size]

            # Step 3: 准备位置编码 (如果不使用 RoPE)
                    如果模型不使用 RoPE，需要添加位置编码

            # Step 4: 准备注意力掩码
                    如果提供了 attention_mask，需要转换为因果掩码格式
                    形状: [batch_size, 1, seq_len, total_len]
                    其中 total_len = seq_len (+ cache_len if past_key_values)

                    因果掩码: 上三角为 -inf，防止看到未来 token
                    ┌─────┬─────┬─────┬─────┐
                    │  0  │-inf │-inf │-inf │  位置 0 只能看到自己
                    ├─────┼─────┼─────┼─────┤
                    │  0  │  0  │-inf │-inf │  位置 1 能看到 0, 1
                    ├─────┼─────┼─────┼─────┤
                    │  0  │  0  │  0  │-inf │  位置 2 能看到 0, 1, 2
                    ├─────┼─────┼─────┼─────┤
                    │  0  │  0  │  0  │  0  │  位置 3 能看到 0, 1, 2, 3
                    └─────┴─────┴─────┴─────┘

            # Step 5: 通过所有 Transformer 层
                    hidden_states = inputs_embeds
                    next_cache = [] if use_cache else None

                    for i, layer in enumerate(layers):
                        past_kv = past_key_values[i] if past_key_values else None
                        hidden_states, present_kv = layer(
                            hidden_states=hidden_states,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_value=past_kv,
                            use_cache=use_cache,
                        )
                        if use_cache:
                            next_cache.append(present_kv)

            # Step 6: 最终归一化
                    hidden_states = norm(hidden_states)
                    形状: [batch_size, seq_len, hidden_size]

            # Step 7: 语言模型头
                    logits = lm_head(hidden_states)
                    形状: [batch_size, seq_len, vocab_size]

            # Step 8: 计算损失 (如果提供了 labels)
                    if labels is not None:
                        loss_fct = CrossEntropyLoss()
                        # 将 logits 展平为 [batch*seq, vocab]
                        # 将 labels 展平为 [batch*seq]
                        # 忽略 labels == -100 的位置
                        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))

            # Step 9: 返回结果
                    return CausalLMOutputWithPast(
                        loss=loss,
                        logits=logits,
                        past_key_values=next_cache,
                    )
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

            Step 1: 初始化
                    past_key_values = None
                    batch_size = input_ids.shape[0]
                    generated = input_ids

            Step 2: 循环生成 (for i in range(max_new_tokens))

                    # 2.1 前向传播
                    outputs = forward(
                        input_ids=input_ids,  # 第一次是整个 prompt，之后只传最后一个 token
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    logits = outputs.logits[:, -1, :]  # 取最后一个位置的 logits
                    past_key_values = outputs.past_key_values

                    # 2.2 应用温度缩放
                    logits = logits / temperature

                    # 2.3 应用 top-k 过滤
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = -inf

                    # 2.4 应用 top-p (nucleus) 过滤
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -inf

                    # 2.5 采样下一个 token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # 2.6 拼接生成的 token
                    generated = torch.cat([generated, next_token], dim=1)
                    input_ids = next_token  # 下次只输入新 token

                    # 2.7 检查是否生成 EOS
                    if eos_token_id is not None and (next_token == eos_token_id).all():
                        break

            Step 3: 返回生成的序列
                    return generated
        """
        pass
