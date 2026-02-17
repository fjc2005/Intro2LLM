"""
分词器基类模块
定义所有分词器的通用接口和基础功能。

分词器 (Tokenizer) 的作用:
1. encode: 将文本转换为 token ID 序列
2. decode: 将 token ID 序列还原为文本

这是 NLP 模型的入口和出口，不同的分词策略影响:
- 词表大小和覆盖范围
- 序列长度
- 多语言支持能力
- 处理未知词的方式
"""

from typing import List, Union, Dict, Optional, Tuple
import json
import os


class BaseTokenizer:
    """
    分词器抽象基类

    定义分词器的标准接口，所有具体分词器 (BPE、ByteLevel) 都应继承此类。

    核心接口:
    - encode: 文本 -> token IDs
    - decode: token IDs -> 文本
    - encode_batch: 批量文本 -> token IDs
    - save/load: 持久化词表

    特殊 Token:
    - pad_token: 填充符，用于对齐序列长度
    - eos_token: 结束符，标记序列结束
    - unk_token: 未知词符，处理词表外词汇
    - bos_token: 起始符 (可选)，标记序列开始
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        """
        初始化分词器。

        Args:
            vocab: 词表字典，token_str -> token_id
            special_tokens: 特殊 token 字典，包含:
                - pad_token: 填充符，默认 "<pad>"
                - eos_token: 结束符，默认 "<|endoftext|>"
                - unk_token: 未知词符，默认 "<unk>"
                - bos_token: 起始符，默认 None (可选)

        初始化步骤:
            Step 1: 设置词表
                    self.vocab = vocab or {}
                    self.inverse_vocab = {v: k for k, v in self.vocab.items()}

            Step 2: 设置特殊 token
                    self.pad_token = special_tokens.get("pad_token", "<pad>")
                    self.eos_token = special_tokens.get("eos_token", "<|endoftext|>")
                    self.unk_token = special_tokens.get("unk_token", "<unk>")
                    self.bos_token = special_tokens.get("bos_token", None)

            Step 3: 获取特殊 token ID
                    self.pad_token_id = self.vocab.get(self.pad_token, 0)
                    self.eos_token_id = self.vocab.get(self.eos_token, 1)
                    self.unk_token_id = self.vocab.get(self.unk_token, 2)
                    self.bos_token_id = self.vocab.get(self.bos_token, None) if self.bos_token else None

            Step 4: 确保特殊 token 在词表中
                    如果词表是空的或特殊 token 不在词表中，
                    需要将特殊 token 加入词表
        """
        pass

    @property
    def vocab_size(self) -> int:
        """
        获取词表大小。

        Returns:
            词表中 token 的数量
        """
        pass

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """
        将文本编码为 token ID 序列。

        Args:
            text: 输入文本字符串
            add_special_tokens: 是否添加特殊 token (如 BOS、EOS)
            max_length: 最大序列长度，超过则截断
            truncation: 是否启用截断

        Returns:
            token ID 列表，[id1, id2, ..., idn]

        子类必须实现此方法，实现具体的分词逻辑。

        一般流程:
            Step 1: 文本预处理
                    - 统一编码 (UTF-8)
                    - 标准化 (如 NFC 规范化)

            Step 2: 分词
                    - 根据具体算法 (BPE、ByteLevel) 将文本切分为 tokens
                    - 每个 token 查找词表得到 ID

            Step 3: 添加特殊 token (如果需要)
                    - 开头添加 BOS token ID (如果有)
                    - 结尾添加 EOS token ID (如果需要)

            Step 4: 截断 (如果需要)
                    - 如果 len(ids) > max_length 且 truncation=True
                    - 截断到 max_length，通常保留前面的部分

            Step 5: 返回 token ID 列表
        """
        raise NotImplementedError("Subclasses must implement encode()")

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        将 token ID 序列解码为文本。

        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token
            clean_up_tokenization_spaces: 是否清理分词产生的多余空格

        Returns:
            解码后的文本字符串

        子类必须实现此方法。

        一般流程:
            Step 1: ID 转 token
                    - 对每个 id，从 inverse_vocab 查找对应的 token 字符串
                    - 如果 id 不在词表中，使用 unk_token

            Step 2: 拼接 tokens
                    - 将 token 列表拼接成字符串
                    - 注意 subword 的处理 (如 BPE 需要去掉 "##" 前缀)

            Step 3: 处理特殊 token
                    - 如果 skip_special_tokens=True，移除 pad、eos、bos 等

            Step 4: 后处理
                    - 清理多余空格
                    - 处理字节解码 (ByteLevel 需要)

            Step 5: 返回文本
        """
        raise NotImplementedError("Subclasses must implement decode()")

    def encode_batch(
        self,
        texts: List[str],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List[List[int]], "torch.Tensor"]]:
        """
        批量编码文本。

        Args:
            texts: 文本列表
            padding: 是否填充，"longest" | "max_length" | True | False
            truncation: 是否截断
            max_length: 最大长度，用于 padding="max_length" 和 truncation
            return_tensors: 返回格式，"pt" (PyTorch) | "np" (NumPy) | None

        Returns:
            字典，包含:
                - input_ids: token ID 序列列表
                - attention_mask: 注意力掩码，1 表示有效 token，0 表示 padding

        流程:
            Step 1: 逐个编码文本
                    encoded = [self.encode(text, ...) for text in texts]

            Step 2: 如果需要填充
                    - 找出最长序列长度 (或 max_length)
                    - 对每个序列，在末尾添加 pad_token 直到目标长度
                    - 创建 attention_mask，有效位置为 1，padding 为 0

            Step 3: 如果需要返回张量
                    - 将列表转换为 torch.Tensor 或 numpy.ndarray

            Step 4: 返回结果字典
        """
        pass

    def save(self, save_directory: str):
        """
        保存分词器到目录。

        Args:
            save_directory: 保存目录路径

        保存内容:
            - vocab.json: 词表，token -> id 的映射
            - special_tokens.json: 特殊 token 配置
            - tokenizer_config.json: 其他配置参数
        """
        pass

    @classmethod
    def load(cls, load_directory: str) -> "BaseTokenizer":
        """
        从目录加载分词器。

        Args:
            load_directory: 加载目录路径

        Returns:
            加载后的分词器实例
        """
        pass

    def tokenize(self, text: str) -> List[str]:
        """
        仅分词，不转换为 ID。

        Args:
            text: 输入文本

        Returns:
            token 字符串列表
        """
        pass

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        将 token 字符串列表转换为 ID 列表。

        Args:
            tokens: token 字符串列表

        Returns:
            token ID 列表
        """
        pass

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """
        将 ID 列表转换为 token 字符串列表。

        Args:
            token_ids: token ID 列表

        Returns:
            token 字符串列表
        """
        pass
