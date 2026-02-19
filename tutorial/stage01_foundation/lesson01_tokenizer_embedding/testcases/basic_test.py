"""
L01: Tokenizer 与 Embedding - 基础测试

覆盖 lesson1 最小可用链路的核心契约：
- BaseTokenizer: 特殊 token、ID 映射、encode_batch 的 padding/mask
- BPETokenizer: 最小可训练/可编解码路径（单词级样例，避免空格复杂度）
- TokenEmbedding: 查表、shape、梯度回传
- RoPE: rotate/apply 的 shape、广播与相对位置不变性
"""

import torch
import pytest
import sys
import os
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from model.embedding import TokenEmbedding, RoPE
from tokenizer.base_tokenizer import BaseTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer


class DummySpaceTokenizer(BaseTokenizer):
    """
    用于测试 BaseTokenizer 通用逻辑的最小实现：
    - tokenize: 以空格切分
    - encode/decode: 直接基于 vocab 映射
    """

    def tokenize(self, text: str) -> list[str]:
        return [t for t in text.split(" ") if t != ""]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> list[int]:
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            if getattr(self, "bos_token_id", None) is not None:
                ids = [self.bos_token_id] + ids
            ids = ids + [self.eos_token_id]

        if max_length is not None and len(ids) > max_length:
            if not truncation:
                raise ValueError("Sequence too long and truncation=False")
            ids = ids[:max_length]

        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            special = {self.pad_token, self.eos_token, self.unk_token}
            if getattr(self, "bos_token", None) is not None:
                special.add(self.bos_token)
            tokens = [t for t in tokens if t not in special]
        text = " ".join(tokens)
        return " ".join(text.split()) if clean_up_tokenization_spaces else text


class TestBaseTokenizer:
    def test_special_tokens_added_when_vocab_empty(self):
        tok = BaseTokenizer(vocab=None, special_tokens=None)

        assert isinstance(tok.vocab, dict)
        assert isinstance(tok.inverse_vocab, dict)

        assert tok.pad_token in tok.vocab
        assert tok.eos_token in tok.vocab
        assert tok.unk_token in tok.vocab

        assert isinstance(tok.pad_token_id, int)
        assert isinstance(tok.eos_token_id, int)
        assert isinstance(tok.unk_token_id, int)

        assert tok.inverse_vocab[tok.pad_token_id] == tok.pad_token
        assert tok.inverse_vocab[tok.eos_token_id] == tok.eos_token
        assert tok.inverse_vocab[tok.unk_token_id] == tok.unk_token

    def test_convert_tokens_and_ids_oov_falls_back_to_unk(self):
        vocab = {"<pad>": 0, "<|endoftext|>": 1, "<unk>": 2, "hello": 3}
        tok = BaseTokenizer(vocab=vocab)

        assert tok.convert_tokens_to_ids(["hello", "nope"]) == [3, tok.unk_token_id]

        tokens = tok.convert_ids_to_tokens([3, 999999])
        assert tokens[0] == "hello"
        assert tokens[1] == tok.unk_token

    def test_encode_batch_padding_and_attention_mask(self):
        vocab = {"<pad>": 0, "<|endoftext|>": 1, "<unk>": 2, "<bos>": 3, "a": 4, "b": 5}
        tok = DummySpaceTokenizer(vocab=vocab, special_tokens={"bos_token": "<bos>"})

        out = tok.encode_batch(["a", "a b"], padding=True, return_tensors="pt")
        assert set(out.keys()) == {"input_ids", "attention_mask"}

        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        assert input_ids.shape == attention_mask.shape == (2, input_ids.shape[1])

        # 第一条更短，末尾应被 pad，mask 对应为 0
        assert input_ids[0, -1].item() == tok.pad_token_id
        assert attention_mask[0, -1].item() == 0
        # 第二条最后一个 token 应为 eos（DummySpaceTokenizer 默认 add_special_tokens=True）
        assert input_ids[1, -1].item() == tok.eos_token_id
        assert attention_mask[1, -1].item() == 1

    def test_save_and_load_roundtrip(self):
        vocab = {"<pad>": 0, "<|endoftext|>": 1, "<unk>": 2, "hello": 3}
        tok = BaseTokenizer(vocab=vocab)

        with tempfile.TemporaryDirectory() as d:
            tok.save(d)
            tok2 = BaseTokenizer.load(d)

        assert tok2.vocab == tok.vocab
        assert tok2.inverse_vocab == tok.inverse_vocab
        assert tok2.pad_token == tok.pad_token
        assert tok2.eos_token == tok.eos_token
        assert tok2.unk_token == tok.unk_token


class TestBPETokenizer:
    def test_bpe_train_single_merge_is_deterministic(self):
        # 构造明确的最高频 pair: ("a","b")
        texts = ["abababab", "abab"]
        tok = BPETokenizer()
        tok.train(texts=texts, vocab_size=6, min_frequency=2)

        assert len(tok.merges) >= 1
        assert tok.merges[0] == ("a", "b")
        assert "ab" in tok.vocab

        # 目标 vocab_size=6（特殊 token 3 + 'a','b' + 'ab'）
        assert tok.vocab_size == 6

    def test_bpe_encode_decode_roundtrip_no_spaces(self):
        # 手工设置 merges，避免 train 的实现细节差异影响本用例
        vocab = {"<pad>": 0, "<|endoftext|>": 1, "<unk>": 2, "h": 3, "e": 4, "l": 5, "o": 6, "he": 7, "ll": 8, "hell": 9, "hello": 10}
        merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")]
        tok = BPETokenizer(vocab=vocab, merges=merges)

        ids = tok.encode("hello", add_special_tokens=False)
        assert ids == [tok.vocab["hello"]]
        assert tok.decode(ids) == "hello"


class TestTokenEmbedding:
    """测试 TokenEmbedding 类的基本功能"""

    def test_embedding_initialization(self):
        """测试嵌入层初始化"""
        vocab_size = 100
        hidden_size = 32
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)

        assert hasattr(emb, "embedding_table")
        assert isinstance(emb.embedding_table, torch.nn.Parameter)
        assert emb.embedding_table.shape == (vocab_size, hidden_size)

        with torch.no_grad():
            mean = emb.embedding_table.mean().item()
            std = emb.embedding_table.std().item()
        assert abs(mean) < 0.05
        assert 0.005 < std < 0.05

    def test_embedding_forward(self):
        """测试嵌入层前向传播"""
        vocab_size = 50
        hidden_size = 16
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)

        input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
        out = emb(input_ids)

        assert out.shape == (2, 3, hidden_size)
        assert out.dtype == emb.embedding_table.dtype

        # 查表语义：输出应等于 embedding_table 的索引结果
        expected = emb.embedding_table[input_ids]
        assert torch.allclose(out, expected)

    def test_embedding_gradient(self):
        """测试嵌入层梯度计算"""
        vocab_size = 20
        hidden_size = 8
        emb = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)

        input_ids = torch.tensor([[1, 2, 2, 3]], dtype=torch.long)
        out = emb(input_ids)
        loss = out.sum()
        loss.backward()

        assert emb.embedding_table.grad is not None

        # 用到的 token 行应有梯度，未用到的行应为 0（或全 0）
        used = {1, 2, 3}
        for idx in range(vocab_size):
            row = emb.embedding_table.grad[idx]
            if idx in used:
                assert row.abs().sum().item() > 0
            else:
                assert torch.allclose(row, torch.zeros_like(row))


class TestRoPE:
    """测试 RoPE 类的基本功能"""

    def test_rope_initialization(self):
        """测试 RoPE 初始化"""
        dim = 32
        rope = RoPE(dim=dim, max_position_embeddings=128, base=10000.0)
        assert hasattr(rope, "inv_freq")
        assert rope.inv_freq.shape == (dim // 2,)
        assert rope.inv_freq.dtype.is_floating_point

    def test_rotate_half(self):
        """测试 rotate_half 函数"""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        y = RoPE.rotate_half(x)
        assert torch.allclose(y, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    def test_rope_forward(self):
        """测试 RoPE 前向传播"""
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        rope = RoPE(dim=head_dim, max_position_embeddings=64)

        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, 2, seq_len, head_dim)  # num_kv_heads=2
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)

        q2, k2 = rope(q, k, position_ids=position_ids)

        assert q2.shape == q.shape
        assert k2.shape == k.shape

        # 旋转应近似保持范数（正交变换）
        q_norm = q.norm(dim=-1)
        q2_norm = q2.norm(dim=-1)
        assert torch.allclose(q_norm, q2_norm, atol=1e-4, rtol=1e-4)

    def test_rope_relative_position(self):
        """测试 RoPE 的相对位置特性"""
        batch, heads, seq_len, head_dim = 1, 2, 12, 32
        rope = RoPE(dim=head_dim, max_position_embeddings=4096)

        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        pos0 = torch.arange(seq_len).unsqueeze(0)
        pos_shift = (torch.arange(seq_len) + 1000).unsqueeze(0)

        q0, k0 = rope(q, k, position_ids=pos0)
        q1, k1 = rope(q, k, position_ids=pos_shift)

        # attention logits 只依赖相对位移：整体平移 position_ids 不改变 qk^T
        att0 = torch.matmul(q0, k0.transpose(-1, -2))
        att1 = torch.matmul(q1, k1.transpose(-1, -2))
        assert torch.allclose(att0, att1, atol=1e-4, rtol=1e-4)


class TestRoPEIntegration:
    """测试 RoPE 与注意力模块的集成"""

    def test_rope_with_attention_shapes(self):
        """测试 RoPE 在注意力场景下的形状"""
        batch, num_heads, num_kv_heads, seq_len, head_dim = 2, 8, 2, 7, 64
        rope = RoPE(dim=head_dim, max_position_embeddings=1024)

        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        pos = torch.arange(seq_len)  # 允许 [seq_len] 输入

        q2, k2 = rope(q, k, position_ids=pos)
        assert q2.shape == q.shape
        assert k2.shape == k.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
