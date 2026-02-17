"""
数据集基类模块
定义数据加载的标准接口，支持流式加载和内存加载。

数据集设计原则:
1. 统一接口: 所有数据集继承 BaseDataset，实现相同的方法
2. 灵活加载: 支持从文件、内存、HF datasets 加载
3. 高效处理: 支持多进程数据加载、缓存、预取
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any
import json
import random


class BaseDataset(Dataset):
    """
    数据集抽象基类

    继承 PyTorch 的 Dataset，定义 LLM 训练数据的标准接口。

    核心功能:
    - 从数据源加载原始数据
    - 使用 tokenizer 将文本转换为模型输入格式
    - 返回包含 input_ids、attention_mask、labels 的字典

    子类需要实现:
    - _load_data(): 加载原始数据
    - __getitem__(): 获取单个样本并处理
    """

    def __init__(
        self,
        data_path: str,
        tokenizer = None,
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化数据集。

        Args:
            data_path: 数据文件或目录路径
                       支持格式: .jsonl, .json, .txt, 目录
            tokenizer: 分词器实例，用于文本编码
            max_length: 最大序列长度，超过则截断
            cache_dir: 预处理结果缓存目录，避免重复处理

        属性:
            self.data: 原始数据列表，每个元素是一条样本
            self.tokenizer: 分词器
            self.max_length: 最大长度
        """
        super().__init__()
        # 保存配置参数
        # 初始化 tokenizer
        # 尝试从缓存加载，否则调用 _load_data() 加载原始数据
        pass

    def _load_data(self, data_path: str) -> List[Any]:
        """
        加载原始数据。

        Args:
            data_path: 数据路径

        Returns:
            原始数据列表

        支持的格式:
            1. JSON Lines (.jsonl):
               每行一个 JSON 对象
               {"text": "..."} 或 {"instruction": "...", "input": "...", "output": "..."}

            2. JSON (.json):
               整个文件是一个 JSON 数组
               [{"text": "..."}, {"text": "..."}]

            3. 纯文本 (.txt):
               每行一个样本，或整个文件作为一个样本

            4. 目录:
               递归加载目录下所有支持的文件

        流程:
            Step 1: 检测文件格式
            Step 2: 根据格式选择加载方式
            Step 3: 验证数据格式
            Step 4: 返回数据列表
        """
        pass

    def __len__(self) -> int:
        """
        获取数据集大小。

        Returns:
            样本数量
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            包含以下字段的字典:
                - input_ids: token ID 序列，[seq_len]
                - attention_mask: 注意力掩码，[seq_len]，1 表示有效，0 表示 padding
                - labels: 目标标签，[seq_len]，-100 表示忽略

        注意:
            子类必须实现此方法，根据任务类型 (预训练/SFT/DPO) 决定如何处理 labels。
        """
        raise NotImplementedError("Subclasses must implement __getitem__()")

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批次整理函数 (DataLoader 使用)。

        Args:
            batch: 样本列表，每个样本是 __getitem__ 的返回值

        Returns:
            批处理后的字典，张量形状为 [batch_size, seq_len]

        功能:
            - 对 input_ids、attention_mask、labels 进行 padding
            - 将所有样本对齐到批次内的最大长度
            - 将列表转换为 PyTorch 张量

        流程:
            Step 1: 找出批次内最大长度
            Step 2: 对每个样本进行 padding
                    input_ids 用 pad_token_id 填充
                    attention_mask 用 0 填充
                    labels 用 -100 填充
            Step 3: 堆叠成张量
            Step 4: 返回批次字典
        """
        pass

    def get_sample_text(self, idx: int) -> str:
        """
        获取原始文本样本 (用于调试)。

        Args:
            idx: 样本索引

        Returns:
            原始文本字符串
        """
        pass

    def shard(self, num_shards: int, shard_id: int) -> "BaseDataset":
        """
        将数据集分片，用于分布式训练。

        Args:
            num_shards: 总分片数
            shard_id: 当前分片 ID (0-indexed)

        Returns:
            当前分片的数据集

        用途:
            在多机多卡训练中，每个进程只处理部分数据
        """
        pass


class TextDataset(BaseDataset):
    """
    通用文本数据集

    用于简单的文本语言建模，每个样本是一段文本。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer = None,
        max_length: int = 2048,
        stride: Optional[int] = None,
    ):
        """
        初始化文本数据集。

        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大序列长度
            stride: 滑动窗口步长，用于长文本切分
                   如果为 None，每个文本作为一个样本
                   如果设置，长文本以 stride 为步长滑动切分
        """
        super().__init__(data_path, tokenizer, max_length)
        # 保存 stride 参数
        pass

    def _load_data(self, data_path: str) -> List[str]:
        """
        加载文本数据。

        根据 stride 设置决定如何切分长文本:
        - stride=None: 每行或每个 JSON 对象作为一个样本
        - stride=int: 长文本以滑动窗口切分，stride 控制重叠程度
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取文本样本。

        返回:
            input_ids: 编码后的 token IDs
            attention_mask: 全 1 (假设没有 padding)
            labels: 与 input_ids 相同 (因果语言建模)
        """
        pass
