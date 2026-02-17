"""
预训练数据集模块
用于因果语言模型 (Causal LM) 的预训练。

预训练任务:
给定一段文本，模型学习预测下一个 token。
- 输入: "今天天气很好"
- 目标: "今天天气很好" (向后错一位)

数据格式:
- 通常是大量无标注文本 (网页、书籍、论文等)
- 需要拼接成固定长度的序列以提高训练效率
"""

import torch
from typing import Dict, List, Optional
from .dataset import BaseDataset


class PretrainDataset(BaseDataset):
    """
    预训练数据集

    用于标准的因果语言建模 (Causal Language Modeling)。

    特点:
    1. 文本拼接: 将多个短文本拼接成长序列，减少 padding 浪费
    2. 端到端预测: 每个位置预测下一个 token
    3. 无监督学习: 只需要原始文本，无需标注

    样本处理:
        原始文本: "Hello world. This is an example."

        编码后: [15496, 995, 13, 1212, 318, 281, 1672, 13]

        input_ids:  [15496, 995, 13, 1212, 318, 281, 1672, 13]
        labels:     [995, 13, 1212, 318, 281, 1672, 13, -100]
                    # labels 与 input_ids 相同，但向右移动一位
                    # 最后一个 token 无预测目标，设为 -100 (忽略)

    文本拼接策略:
        为了提高训练效率，将多个文本用 EOS token 分隔后拼接:
        "Text 1 <|endoftext|> Text 2 <|endoftext|> Text 3"

        然后按 max_length 切分成长度为 L 的序列。

        这样处理的好处:
        - 减少 padding，提高 GPU 利用率
        - EOS 帮助模型学习文本边界
        - 不同文本在注意力中被分隔
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        concat_every_n_samples: int = 1000,
    ):
        """
        初始化预训练数据集。

        Args:
            data_path: 数据文件路径 (.jsonl, .txt)
            tokenizer: 分词器
            max_length: 最大序列长度
            concat_every_n_samples: 每多少个样本拼接一次
                                     较大的值减少拼接开销，但增加内存占用

        属性:
            self.examples: 拼接并切分后的样本列表
        """
        super().__init__(data_path, tokenizer, max_length)
        # 执行文本拼接和切分
        # 将多个短文本拼接成适合 max_length 的长序列
        pass

    def _concatenate_and_split(self, texts: List[str]) -> List[List[int]]:
        """
        将多个文本拼接并切分为固定长度的序列。

        Args:
            texts: 原始文本列表

        Returns:
            token ID 序列列表，每个序列长度为 max_length

        流程:
            Step 1: 逐个编码文本
                    遍历所有文本，对每个文本进行编码
                    编码时不添加特殊 token
                    在每个文本编码结果后添加 EOS token
                    将所有 token 扩展到一个缓冲区中

            Step 2: 切分成长度为 max_length 的块
                    当缓冲区长度大于等于 max_length 时循环:
                        取出前 max_length 个 token 作为一个样本
                        将取出的样本添加到结果列表
                        更新缓冲区，移除已取出的部分
                        可以选择使用滑动窗口 stride 保留部分重叠内容

            Step 3: 处理剩余部分
                    可以选择丢弃不完整的最后一块
                    或者保留用于下一个 batch 的拼接

        注意:
            - 添加 EOS token 标记文本边界
            - 最后一个不完整的块可以丢弃或 padding
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取预训练样本。

        Args:
            idx: 样本索引

        Returns:
            包含以下字段的字典:
                - input_ids: [max_length]，token ID 序列
                - attention_mask: [max_length]，全 1 (预训练数据无 padding)
                - labels: [max_length]，预测目标

        标签处理:
            创建 input_ids 的副本作为 labels
            因果掩码机制: 每个位置预测下一个 token
            将 labels 整体向左移动一位
            最后一个位置设为 -100 表示无预测目标

        等价操作:
            将 input_ids 从第二个位置开始的部分与 -100 拼接
            形成与 input_ids 长度相同的 labels 序列
        """
        pass

    def create_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        创建 DataLoader。

        预训练数据的特点:
        - 所有序列长度相同 (max_length)，无需动态 padding
        - 可以使用较大的 batch_size
        - 多进程加载提高数据预处理速度

        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载进程数
            pin_memory: 是否将数据固定在页内存 (加速 GPU 传输)

        Returns:
            DataLoader 实例
        """
        pass


class PackedPretrainDataset(BaseDataset):
    """
    打包预训练数据集 (Packed Dataset)

    进一步优化预训练数据的 packing 策略。

    标准预训练数据的问题:
    - 短文本拼接后可能在序列中间截断句子
    - 每个序列最后几个 token 的上下文不完整

    打包策略改进:
    使用 Attention Mask 让不同来源的文本互不可见:

        序列: [Text1 tokens] [EOS] [Text2 tokens] [EOS] [Text3 tokens] [PAD]

        Attention Mask (示意图):
        Text1 只能看到 Text1
        Text2 只能看到 Text2
        Text3 只能看到 Text3

        通过构造特殊的 attention_mask 实现:
        ┌─────┬─────┬─────┬─────┬─────┐
        │  1  │  0  │  0  │  0  │  0  │  Text1 自注意力
        │  0  │  1  │  0  │  0  │  0  │  EOS 分隔
        │  0  │  0  │  1  │  0  │  0  │  Text2 自注意力
        │  0  │  0  │  0  │  1  │  0  │  EOS 分隔
        │  0  │  0  │  0  │  0  │  1  │  Text3 自注意力
        └─────┴─────┴─────┴─────┴─────┘

    需要配合 Flash Attention 的 varlen 功能使用。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_sequences_per_pack: int = 5,
    ):
        """
        初始化打包数据集。

        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大序列长度
            max_sequences_per_pack: 每个 pack 最多包含的原始文本数
        """
        super().__init__(data_path, tokenizer, max_length)
        pass

    def _pack_sequences(self, tokenized_texts: List[List[int]]) -> List[Dict]:
        """
        将多个文本打包成一个序列。

        使用 bin packing 算法 (如 Best Fit Decreasing):
        - 按长度排序文本
        - 将最长的放入当前 pack
        - 尝试放入更多文本直到接近 max_length

        Returns:
            包含 input_ids、attention_mask、position_ids 的字典列表
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取打包后的样本。

        Returns:
            - input_ids: [max_length]
            - attention_mask: [max_length, max_length] 或 [max_length] (cumsum)
            - position_ids: [max_length]，每个文本的位置从 0 重新开始
            - labels: [max_length]
        """
        pass
