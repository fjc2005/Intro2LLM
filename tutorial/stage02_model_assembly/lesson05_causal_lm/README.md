# 课时5：完整因果语言模型与生成策略

## 学习目标

1. 理解Causal Language Model的完整架构
2. 掌握模型组装：Embedding → Transformer层 → 输出投影
3. 理解并实现KV缓存机制优化推理
4. 掌握文本生成策略：Greedy、Temperature、Top-k、Top-p
5. 理解生成终止条件与重复惩罚

---

## 1. Causal Language Model架构

### 1.1 完整架构概览

```
输入: input_ids [batch, seq_len]

Step 1: Token Embedding
    hidden_states = TokenEmbedding(input_ids)
    Shape: [batch, seq_len, hidden_size]

Step 2: 多层Transformer Block
    for i in range(num_layers):
        hidden_states = TransformerBlock[i](hidden_states)
    Shape: [batch, seq_len, hidden_size]

Step 3: 最终归一化
    hidden_states = LayerNorm(hidden_states)

Step 4: 输出投影 (LM Head)
    logits = hidden_states @ W_lm_head.T
    Shape: [batch, seq_len, vocab_size]

输出: logits (每个位置对每个token的预测分数)
```

### 1.2 权重共享 (Weight Tying)

```
标准设置 (无共享):
    TokenEmbedding: vocab_size × hidden_size
    LM Head:        hidden_size × vocab_size
    总计: 2 × vocab_size × hidden_size

权重共享 (Weight Tying):
    LM Head = TokenEmbedding.weight.T
    总计: vocab_size × hidden_size
    节省约50%参数量

注意: 只有当tie_word_embeddings=True时才共享
```

### 1.3 配置驱动的架构

```python
# 小型模型配置 (tiny)
tiny_config = ModelConfig(
    vocab_size=32000,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=2048,
)

# 7B模型配置 (类似LLaMA-7B)
llama_7b_config = ModelConfig(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,  # MHA，可改为8启用GQA
    intermediate_size=11008,
    max_position_embeddings=4096,
)
```

---

## 2. KV缓存机制

### 2.1 为什么需要KV缓存？

```
问题: 自回归生成时，每次都要重新计算所有位置的K、V

示例: 生成 "Hello world!"
Step 1: 输入 [BOS]，生成 "Hello"
    - 计算位置0的Q、K、V

Step 2: 输入 [BOS, "Hello"]，生成 "world"
    - 重新计算位置0和1的K、V (重复计算!)
    - 位置0的K、V与Step 1完全相同

Step 3: 输入 [BOS, "Hello", "world"]，生成 "!"
    - 重新计算位置0、1、2的K、V

浪费: 每次生成都重复计算之前所有位置的K、V
```

### 2.2 KV缓存原理

```
核心思想: 缓存之前计算的K、V，只计算新token的K、V

Step 1: 输入 [BOS]
    - 计算K[0], V[0]
    - Cache: {K: [K[0]], V: [V[0]]}
    - 生成 "Hello"

Step 2: 输入 "Hello" (只输入新token!)
    - 计算K[1], V[1] (只算新的!)
    - Cache: {K: [K[0], K[1]], V: [V[0], V[1]]}
    - Attention时Query只有Q[1]，但Key/Value用全部缓存
    - 生成 "world"

Step 3: 输入 "world"
    - 计算K[2], V[2]
    - Cache: {K: [K[0], K[1], K[2]], V: [V[0], V[1], V[2]]}
    - 生成 "!"

优势:
    - 时间复杂度从O(n²)降到O(n)每步
    - 内存只增加O(n)存储缓存
```

### 2.3 KV缓存实现细节

```python
# 每层Transformer的缓存结构
past_key_value = (key_cache, value_cache)
# key_cache:   [batch, num_heads, cache_len, head_dim]
# value_cache: [batch, num_heads, cache_len, head_dim]

# 前向传播时
if past_key_value is not None:
    # 拼接历史缓存和当前K、V
    key = torch.cat([past_key_value[0], current_key], dim=2)
    value = torch.cat([past_key_value[1], current_value], dim=2)

# 返回更新后的缓存
present_key_value = (key, value)
```

### 2.4 KV缓存内存计算

```
每层缓存大小:
    key_cache:   batch × num_heads × seq_len × head_dim
    value_cache: batch × num_heads × seq_len × head_dim

总缓存 (所有层):
    2 × num_layers × batch × num_heads × seq_len × head_dim × sizeof(dtype)

示例 (LLaMA-7B, batch=1, seq_len=4096, fp16):
    = 2 × 32 × 1 × 32 × 4096 × 128 × 2 bytes
    = 2 GB

GQA优化 (num_kv_heads=8):
    = 2 × 32 × 1 × 8 × 4096 × 128 × 2 bytes
    = 512 MB (减少75%)
```

---

## 3. 文本生成策略

### 3.1 贪心解码 (Greedy Decoding)

```
策略: 每步选择概率最高的token

算法:
    token_id = argmax(logits)

优点:
    - 简单、快速、确定性强

缺点:
    - 容易生成重复、机械的内容
    - 缺乏多样性
    - 不是全局最优 (局部最优≠全局最优)

适用场景:
    - 需要确定性输出的场景
    - 事实性问答
```

### 3.2 Temperature采样

```
策略: 通过温度参数控制分布的"锐度"

算法:
    probs = softmax(logits / temperature)
    token_id = sample(probs)

temperature效果:
    temperature → 0:  趋近贪心解码 (最确定)
    temperature = 1:  原始分布
    temperature > 1:  分布更平缓 (更多样)

温度选择建议:
    - 0.1-0.5: 事实性任务、代码生成
    - 0.7-1.0: 一般对话、创意写作
    - 1.0-1.5: 头脑风暴、探索性生成
```

**Temperature的数学原理**:
```
原始分布: p_i = exp(z_i) / Σ exp(z_j)

加温度后: p_i(t) = exp(z_i / t) / Σ exp(z_j / t)

t → 0:
    z_max/t → ∞ (最大logit)
    其他z/t → -∞ 或有限值
    → 退化为one-hot分布

t → ∞:
    所有z/t → 0
    → 均匀分布
```

### 3.3 Top-k采样

```
策略: 只从概率最高的k个token中采样

算法:
    top_k_logits = top_k(logits, k)
    # 其他位置设为-∞
    probs = softmax(top_k_logits)
    token_id = sample(probs)

优点:
    - 过滤掉长尾的低概率token
    - 减少生成无意义内容的风险

缺点:
    - k值难以确定 (分布有宽有窄)
    - 固定k可能过宽或过严

典型值: k=50
```

### 3.4 Top-p采样 (Nucleus Sampling)

```
策略: 从累积概率达到p的最小token集合中采样

算法:
    1. 按logits排序 (降序)
    2. 计算累积概率: cumsum(softmax(sorted_logits))
    3. 找到最小索引使 cumsum >= p
    4. 只保留这些token，其余设为-∞
    5. softmax + sample

优点:
    - 动态调整候选集大小
    - 分布集中时自动减少候选 (更确定)
    - 分布分散时自动增加候选 (更多样)

典型值: p=0.9 (90%概率质量)

示例:
    分布A: [0.9, 0.05, 0.03, 0.02] → 取第1个token (累积0.9)
    分布B: [0.4, 0.3, 0.2, 0.1] → 取前3个token (累积0.9)
```

### 3.5 重复惩罚 (Repetition Penalty)

```
策略: 降低已生成token的采样概率

算法:
    for token in generated_tokens:
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty

    penalty > 1: 抑制重复
    penalty = 1: 无惩罚
    penalty < 1: 鼓励重复 (罕见)

典型值: 1.0-1.2
```

### 3.6 生成停止条件

```
1. 最大长度限制: max_new_tokens 或 max_length
2. 遇到EOS (End of Sequence) token
3. 遇到自定义停止词 (如 "\n\n" 表示段落结束)
4. 满足某些模式 (如完成JSON结构)

实现:
    stop_reason = None
    for i in range(max_new_tokens):
        token = generate_next_token(...)
        if token == eos_token_id:
            stop_reason = "eos"
            break
        if i >= max_new_tokens - 1:
            stop_reason = "length"
```

---

## 4. 实现指引

### 4.1 model/causal_lm.py

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import torch.nn.functional as F

class CausalLM(nn.Module):
    """
    Causal Language Model (因果语言模型)

    完整的Transformer Decoder-only架构
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Step 1: Token Embedding
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Step 2: 多层Transformer Block
        # self.layers = nn.ModuleList([
        #     TransformerBlock(config) for _ in range(config.num_hidden_layers)
        # ])

        # Step 3: 最终归一化
        # if config.use_rms_norm:
        #     self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # else:
        #     self.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Step 4: LM Head (输出投影)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Step 5: 权重共享 (如果配置启用)
        # if config.tie_word_embeddings:
        #     self.lm_head.weight = self.embed_tokens.weight

        # Step 6: 初始化
        # self._init_weights()

        pass

    def _init_weights(self):
        """权重初始化"""
        # 标准正态初始化，注意embedding和lm_head的特殊处理
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: 注意力掩码
            past_key_values: 每层的前序KV缓存，List长度为num_layers
            use_cache: 是否返回KV缓存

        Returns:
            logits: [batch, seq_len, vocab_size]
            past_key_values: 更新后的KV缓存 (如果use_cache=True)
        """
        # Step 1: Embedding
        # hidden_states = self.embed_tokens(input_ids)

        # Step 2: 逐层通过Transformer Block
        # next_cache = [] if use_cache else None
        # for i, layer in enumerate(self.layers):
        #     past_kv = past_key_values[i] if past_key_values is not None else None
        #     hidden_states, present_kv = layer(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         past_key_value=past_kv,
        #         use_cache=use_cache,
        #     )
        #     if use_cache:
        #         next_cache.append(present_kv)

        # Step 3: 最终归一化
        # hidden_states = self.norm(hidden_states)

        # Step 4: LM Head投影
        # logits = self.lm_head(hidden_states)

        # Step 5: 返回结果
        pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            input_ids: 输入token IDs [batch, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: Top-k采样 (0表示禁用)
            top_p: Top-p采样 (1.0表示禁用)
            repetition_penalty: 重复惩罚系数
            eos_token_id: 结束token ID
            pad_token_id: 填充token ID

        Returns:
            output_ids: 生成的完整序列 [batch, seq_len + max_new_tokens]
        """
        # Step 1: 初始化
        # batch_size = input_ids.shape[0]
        # device = input_ids.device
        # past_key_values = None

        # Step 2: 自回归生成循环
        # for _ in range(max_new_tokens):
        #     # 2.1 前向传播 (使用KV缓存)
        #     outputs = self.forward(
        #         input_ids if past_key_values is None else input_ids[:, -1:],
        #         past_key_values=past_key_values,
        #         use_cache=True,
        #     )
        #     logits, past_key_values = outputs
        #
        #     # 2.2 取最后一个位置的logits
        #     next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
        #
        #     # 2.3 应用重复惩罚
        #     # if repetition_penalty != 1.0:
        #     #     next_token_logits = self._apply_repetition_penalty(
        #     #         next_token_logits, input_ids, repetition_penalty
        #     #     )
        #
        #     # 2.4 应用温度
        #     # if temperature != 1.0:
        #     #     next_token_logits = next_token_logits / temperature
        #
        #     # 2.5 应用Top-k
        #     # if top_k > 0:
        #     #     indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        #     #     next_token_logits[indices_to_remove] = float('-inf')
        #
        #     # 2.6 应用Top-p
        #     # if top_p < 1.0:
        #     #     sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        #     #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        #     #     sorted_indices_to_remove = cumulative_probs > top_p
        #     #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        #     #     sorted_indices_to_remove[..., 0] = False
        #     #     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        #     #     next_token_logits[indices_to_remove] = float('-inf')
        #
        #     # 2.7 采样
        #     # probs = F.softmax(next_token_logits, dim=-1)
        #     # next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        #
        #     # 2.8 拼接到序列
        #     # input_ids = torch.cat([input_ids, next_token], dim=1)
        #
        #     # 2.9 检查是否生成EOS
        #     # if eos_token_id is not None and (next_token == eos_token_id).all():
        #     #     break

        # Step 3: 返回完整序列
        pass

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """应用重复惩罚"""
        # Step 1: 统计已生成的token
        # Step 2: 对正logits除以penalty，负logits乘以penalty
        pass
```

---

## 5. 关键公式总结

### KV缓存大小
```
Cache Size = 2 × num_layers × batch × num_kv_heads × seq_len × head_dim × dtype_size
```

### Temperature采样
```
P(token_i) = exp(z_i / T) / Σ_j exp(z_j / T)
```

### Top-p累积概率
```
Nucleus = {token_i | Σ_{j=1}^{i} P(token_j) <= p}
```

### 生成时间复杂度
```
无KV缓存: O(n²) 每步，总计 O(n³)
有KV缓存: O(n) 每步，总计 O(n²)
```

---

## 6. 常见陷阱与注意事项

1. **KV缓存更新**: 确保每层独立缓存，不要混用
2. **输入截断**: 使用KV缓存时只输入新token，不要重复输入完整序列
3. **Attention掩码**: 缓存模式下需要调整掩码处理
4. **温度位置**: 在softmax前应用，不要在后
5. **Top-p实现**: 注意保持排序索引的对应关系
6. **重复惩罚范围**: 通常只惩罚已生成的token，不是整个词表
7. **EOS处理**: 生成EOS后是否继续影响最终结果

---

## 7. 课后练习

1. **手动计算生成**: 给定小词表和2层Transformer，手动模拟一步生成
2. **KV缓存实验**: 测试seq_len=1024和4096时的缓存内存占用
3. **采样对比**: 对比不同temperature和top-p设置下的生成多样性
4. **贪心vs采样**: 在同一提示词下，对比贪心解码和采样的输出差异
5. **早停策略**: 实现一个基于重复检测的早停机制
