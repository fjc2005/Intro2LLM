"""
模型模块测试
测试模型各组件的正确性。
"""

import torch
import pytest
from model.config import ModelConfig
from model.norm import LayerNorm, RMSNorm
from model.embedding import TokenEmbedding, RoPE
from model.attention import MultiHeadAttention, GroupedQueryAttention
from model.feedforward import FeedForward, SwiGLU
from model.transformer_block import TransformerBlock
from model.causal_lm import CausalLM


def test_model_config():
    """测试模型配置。"""
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    # 验证计算属性
    assert config.head_dim == 64  # 256 / 4
    assert config.num_key_value_groups == 1  # 4 / 4

    print("✓ ModelConfig 测试通过")


def test_layernorm():
    """测试 LayerNorm。"""
    ln = LayerNorm(normalized_shape=256)
    x = torch.randn(2, 10, 256)

    output = ln(x)

    # 验证输出形状
    assert output.shape == x.shape

    print("✓ LayerNorm 测试通过")


def test_rmsnorm():
    """测试 RMSNorm。"""
    rms = RMSNorm(hidden_size=256)
    x = torch.randn(2, 10, 256)

    output = rms(x)

    # 验证输出形状
    assert output.shape == x.shape

    print("✓ RMSNorm 测试通过")


def test_rope():
    """测试 RoPE。"""
    rope = RoPE(dim=64, max_position_embeddings=512)

    batch_size = 2
    num_heads = 4
    seq_len = 10
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    q_embed, k_embed = rope(q, k, position_ids)

    # 验证输出形状
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape

    print("✓ RoPE 测试通过")


def test_multi_head_attention():
    """测试多头注意力。"""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    mha = MultiHeadAttention(config)

    batch_size = 2
    seq_len = 10
    hidden_size = 256

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    output, _ = mha(hidden_states)

    # 验证输出形状
    assert output.shape == hidden_states.shape

    print("✓ MultiHeadAttention 测试通过")


def test_grouped_query_attention():
    """测试分组查询注意力。"""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    gqa = GroupedQueryAttention(config)

    batch_size = 2
    seq_len = 10
    hidden_size = 256

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    output, _ = gqa(hidden_states)

    # 验证输出形状
    assert output.shape == hidden_states.shape

    print("✓ GroupedQueryAttention 测试通过")


def test_feedforward():
    """测试前馈网络。"""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    ff = FeedForward(config)

    batch_size = 2
    seq_len = 10
    hidden_size = 256

    x = torch.randn(batch_size, seq_len, hidden_size)
    output = ff(x)

    # 验证输出形状
    assert output.shape == x.shape

    print("✓ FeedForward 测试通过")


def test_transformer_block():
    """测试 Transformer Block。"""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    block = TransformerBlock(config, layer_idx=0)

    batch_size = 2
    seq_len = 10
    hidden_size = 256

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    output, _ = block(hidden_states)

    # 验证输出形状
    assert output.shape == hidden_states.shape

    print("✓ TransformerBlock 测试通过")


def test_causal_lm():
    """测试因果语言模型。"""
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        use_rms_norm=True,
        use_rope=True,
        use_swiglu=True,
    )

    model = CausalLM(config)

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids=input_ids, labels=labels)

    # 验证输出
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert outputs.loss is not None

    print("✓ CausalLM 测试通过")


if __name__ == "__main__":
    print("运行模型模块测试...\n")

    test_model_config()
    test_layernorm()
    test_rmsnorm()
    test_rope()
    test_multi_head_attention()
    test_grouped_query_attention()
    test_feedforward()
    test_transformer_block()
    test_causal_lm()

    print("\n✅ 所有测试通过！")
