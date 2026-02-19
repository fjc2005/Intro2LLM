"""
L01: Tokenizer 与 Embedding - 进阶测试

覆盖 lesson1 的进阶契约：
- PositionalEncoding: 正余弦公式、加法注入
- ByteLevelTokenizer: bytes<->unicode 双射与可逆性（包含非 ASCII）
- RoPE: 长序列/低精度稳定性、base 参数影响、范数守恒
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.embedding import TokenEmbedding, RoPE, PositionalEncoding
from tokenizer.byte_level_tokenizer import ByteLevelTokenizer


def _reference_sinusoidal_pe(d_model: int, max_len: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(max_len, device=device).unsqueeze(1)  # [max_len, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
    )  # [d_model/2]
    pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class TestRoPEAdvanced:
    """测试 RoPE 进阶特性"""

    def test_rope_long_sequence(self):
        """测试 RoPE 处理长序列"""
        batch, heads, seq_len, head_dim = 1, 4, 4096, 32
        rope = RoPE(dim=head_dim, max_position_embeddings=seq_len)
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        pos = torch.arange(seq_len).unsqueeze(0)

        q2, k2 = rope(q, k, pos)
        assert q2.shape == q.shape
        assert k2.shape == k.shape
        assert not torch.isnan(q2).any()
        assert not torch.isinf(q2).any()

    def test_rope_different_bases(self):
        """测试不同 base 参数的影响"""
        batch, heads, seq_len, head_dim = 1, 2, 32, 32
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        pos = torch.arange(seq_len).unsqueeze(0)

        rope1 = RoPE(dim=head_dim, max_position_embeddings=1024, base=10000.0)
        rope2 = RoPE(dim=head_dim, max_position_embeddings=1024, base=1000.0)

        q1, k1 = rope1(q, k, pos)
        q2, k2 = rope2(q, k, pos)

        att1 = torch.matmul(q1, k1.transpose(-1, -2))
        att2 = torch.matmul(q2, k2.transpose(-1, -2))
        # base 改变频率分布，通常会导致注意力分数不同
        assert not torch.allclose(att1, att2)

    def test_rope_energy_conservation(self):
        """测试 RoPE 能量守恒 (旋转不改变向量长度)"""
        batch, heads, seq_len, head_dim = 2, 4, 64, 32
        rope = RoPE(dim=head_dim, max_position_embeddings=2048)
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)

        q2, k2 = rope(q, k, pos)
        assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-4, rtol=1e-4)
        assert torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_rope_batch_processing(self):
        """测试批量处理"""
        batch, heads, seq_len, head_dim = 3, 2, 17, 32
        rope = RoPE(dim=head_dim, max_position_embeddings=512)
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)
        q2, k2 = rope(q, k, pos)
        assert q2.shape == q.shape
        assert k2.shape == k.shape


class TestPositionalEncoding:
    """测试 Sinusoidal 位置编码"""

    def test_positional_encoding_initialization(self):
        """测试位置编码初始化"""
        pe = PositionalEncoding(d_model=32, max_len=128)
        # 位置表应是 buffer，不参与梯度
        buffers = dict(pe.named_buffers())
        assert len(buffers) >= 1
        any_buf = next(iter(buffers.values()))
        assert any_buf.requires_grad is False

    def test_positional_encoding_forward(self):
        """测试位置编码前向传播"""
        d_model, max_len = 32, 128
        module = PositionalEncoding(d_model=d_model, max_len=max_len)
        x = torch.randn(2, 16, d_model)
        y = module(x)
        assert y.shape == x.shape
        # 差值应等于对应位置的 PE
        ref = _reference_sinusoidal_pe(d_model=d_model, max_len=max_len, device=x.device)[: x.shape[1]]
        delta = (y - x).float()
        assert torch.allclose(delta[0], ref, atol=1e-4, rtol=1e-4)

    def test_positional_encoding_additive(self):
        """测试位置编码是加法形式"""
        d_model = 16
        module = PositionalEncoding(d_model=d_model, max_len=32)
        x = torch.zeros(1, 10, d_model)
        y = module(x)
        # 输入为 0 时输出应等于位置编码本身
        ref = _reference_sinusoidal_pe(d_model=d_model, max_len=32, device=x.device)[:10]
        assert torch.allclose(y[0].float(), ref, atol=1e-4, rtol=1e-4)


class TestTokenEmbeddingAdvanced:
    """测试 TokenEmbedding 进阶特性"""

    def test_embedding_padding(self):
        """测试 padding token 处理"""
        vocab_size, hidden_size = 10, 8
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)
        input_ids = torch.tensor([[0, 1, 0, 2]], dtype=torch.long)  # 假设 0 是 pad
        out = emb(input_ids)
        assert out.shape == (1, 4, hidden_size)

    def test_embedding_weight_tying(self):
        """测试权重共享 (embedding 与 lm_head)"""
        vocab_size, hidden_size = 20, 12
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        # 常见的 weight tying：lm_head.weight 与 embedding_table 共享同一份参数
        lm_head.weight = emb.embedding_table
        assert lm_head.weight is emb.embedding_table

        x = torch.tensor([[1, 2, 3]], dtype=torch.long)
        h = emb(x).sum(dim=1)  # [B, H]
        logits = lm_head(h)
        loss = logits.sum()
        loss.backward()
        assert emb.embedding_table.grad is not None

    def test_embedding_large_vocab(self):
        """测试大词表"""
        vocab_size, hidden_size = 50000, 32
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)
        ids = torch.randint(0, vocab_size, (2, 4), dtype=torch.long)
        out = emb(ids)
        assert out.shape == (2, 4, hidden_size)

class TestRoPENumericalStability:
    """测试 RoPE 数值稳定性"""

    def test_rope_fp16_stability(self):
        """测试 FP16 下的数值稳定性"""
        batch, heads, seq_len, head_dim = 1, 2, 512, 64
        rope = RoPE(dim=head_dim, max_position_embeddings=seq_len)
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16)
        pos = torch.arange(seq_len).unsqueeze(0)
        q2, k2 = rope(q, k, pos)
        assert not torch.isnan(q2).any()
        assert not torch.isinf(q2).any()

    def test_rope_bf16_stability(self):
        """测试 BF16 下的数值稳定性"""
        if not hasattr(torch, "bfloat16"):
            pytest.skip("bfloat16 not available")
        batch, heads, seq_len, head_dim = 1, 2, 512, 64
        rope = RoPE(dim=head_dim, max_position_embeddings=seq_len)
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)
        pos = torch.arange(seq_len).unsqueeze(0)
        q2, k2 = rope(q, k, pos)
        assert not torch.isnan(q2.float()).any()
        assert not torch.isinf(q2.float()).any()


class TestRoPEMathematicalProperties:
    """测试 RoPE 数学性质"""

    def test_rope_2d_rotation(self):
        """测试二维旋转"""
        head_dim = 2
        rope = RoPE(dim=head_dim, max_position_embeddings=16, base=10000.0)
        q = torch.tensor([[[[1.0, 0.0]]]])  # [B=1,H=1,T=1,D=2]
        k = torch.tensor([[[[0.0, 1.0]]]])
        pos = torch.tensor([[1]])
        q2, k2 = rope(q, k, pos)
        # 旋转后仍应是有限数值
        assert torch.isfinite(q2).all()
        assert torch.isfinite(k2).all()

    def test_rope_invariance_to_absolute_position(self):
        """测试绝对位置的可加性 (用于理解 RoPE)"""
        batch, heads, seq_len, head_dim = 1, 1, 8, 32
        rope = RoPE(dim=head_dim, max_position_embeddings=4096)
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        pos_a = torch.arange(seq_len).unsqueeze(0)
        pos_b = (torch.arange(seq_len) + 1234).unsqueeze(0)

        qa, ka = rope(q, k, pos_a)
        qb, kb = rope(q, k, pos_b)
        att_a = torch.matmul(qa, ka.transpose(-1, -2))
        att_b = torch.matmul(qb, kb.transpose(-1, -2))
        assert torch.allclose(att_a, att_b, atol=1e-4, rtol=1e-4)


class TestByteLevelTokenizer:
    def test_bytes_to_unicode_is_bijection(self):
        mapping = ByteLevelTokenizer._create_bytes_to_unicode()
        assert isinstance(mapping, dict)
        assert len(mapping) == 256
        assert len(set(mapping.values())) == 256
        # 每个映射应是单字符 str（GPT-2 风格）
        assert all(isinstance(v, str) and len(v) == 1 for v in mapping.values())

    def test_bytes_unicode_roundtrip(self):
        tok = ByteLevelTokenizer()
        samples = [
            "hello world",
            "Hello\nworld\t!",
            "中文测试",
            "emoji🙂🚀",
        ]
        for s in samples:
            u = tok._bytes_to_unicode(s)
            back = tok._unicode_to_bytes(u)
            assert back == s

    def test_encode_decode_is_reversible_without_merges(self):
        # 构造只包含 256 字节基础符号的 vocab（无 merges 也应可逆）
        mapping = ByteLevelTokenizer._create_bytes_to_unicode()
        vocab: dict[str, int] = {"<pad>": 0, "<|endoftext|>": 1, "<unk>": 2}
        offset = len(vocab)
        for i in range(256):
            vocab[mapping[i]] = offset + i

        tok = ByteLevelTokenizer(vocab=vocab, merges=[])
        text = "Hello, 中文🙂"
        ids = tok.encode(text, add_special_tokens=False)
        assert all(isinstance(i, int) for i in ids)
        assert tok.unk_token_id not in ids
        assert tok.decode(ids) == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
