"""
DPO (Direct Preference Optimization) 数据集模块
处理偏好数据 (chosen vs rejected)。

DPO 数据格式:
- prompt: 输入提示
- chosen: 偏好回复
- rejected: 非偏好回复

或者 Conversational 格式:
- messages: 对话历史
- chosen: 偏好的 assistant 回复
- rejected: 非偏好的 assistant 回复
"""

import torch
from typing import Dict, List, Optional
from .dataset import BaseDataset


class DPODataset(BaseDataset):
    """
    DPO 偏好数据集

    用于直接偏好优化训练。

    数据格式示例:
        {
            "prompt": "Human: 什么是机器学习?\n\nAssistant:",
            "chosen": "机器学习是人工智能的一个分支...",
            "rejected": "机器学习是写代码。"
        }

    或 Conversational 格式:
        {
            "messages": [
                {"role": "user", "content": "什么是机器学习?"}
            ],
            "chosen": {"role": "assistant", "content": "机器学习是..."},
            "rejected": {"role": "assistant", "content": "机器学习是写代码。"}
        }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
    ):
        """
        初始化 DPO 数据集。

        Args:
            data_path: 数据路径 (.jsonl)
            tokenizer: 分词器
            max_length: 最大序列长度
            max_prompt_length: 最大 prompt 长度
        """
        super().__init__(data_path, tokenizer, max_length)
        self.max_prompt_length = max_prompt_length

    def _format_prompt(self, sample: Dict) -> str:
        """
        格式化 prompt。

        Args:
            sample: 数据样本

        Returns:
            格式化后的 prompt 字符串
        """
        pass

    def _get_chosen_text(self, sample: Dict) -> str:
        """
        获取偏好回复文本。

        Args:
            sample: 数据样本

        Returns:
            偏好回复文本
        """
        pass

    def _get_rejected_text(self, sample: Dict) -> str:
        """
        获取非偏好回复文本。

        Args:
            sample: 数据样本

        Returns:
            非偏好回复文本
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取 DPO 样本。

        Args:
            idx: 样本索引

        Returns:
            包含以下字段的字典:
                - prompt_input_ids: prompt 的 token IDs
                - prompt_attention_mask: prompt 的 attention mask
                - chosen_input_ids: 偏好回复的 token IDs
                - chosen_attention_mask: 偏好回复的 attention mask
                - chosen_labels: 偏好回复的 labels
                - rejected_input_ids: 非偏好回复的 token IDs
                - rejected_attention_mask: 非偏好回复的 attention mask
                - rejected_labels: 非偏好回复的 labels

        处理流程:
            Step 1: 获取样本
                    根据索引从数据集中获取单个样本

            Step 2: 格式化 prompt
                    调用 _format_prompt 方法将样本格式化为 prompt 字符串

            Step 3: 获取回复文本
                    调用 _get_chosen_text 获取偏好回复文本
                    调用 _get_rejected_text 获取非偏好回复文本

            Step 4: 编码 prompt
                    使用 tokenizer 编码 prompt 字符串
                    不添加特殊 token
                    如果长度超过 max_prompt_length 则进行截断

            Step 5: 编码回复
                    分别编码 chosen_text 和 rejected_text
                    不添加特殊 token
                    在每个回复 token 列表末尾添加 EOS token

            Step 6: 拼接序列
                    将 prompt tokens 与 chosen tokens 拼接形成完整 chosen 序列
                    将 prompt tokens 与 rejected tokens 拼接形成完整 rejected 序列

            Step 7: 检查长度并截断
                    如果 chosen 序列长度超过 max_length，截断到 max_length
                    如果 rejected 序列长度超过 max_length，截断到 max_length

            Step 8: 创建 labels
                    只对回复部分计算损失，prompt 部分被 mask
                    创建 chosen_labels: prompt 部分设为 -100，回复部分为对应 token ID
                    创建 rejected_labels: prompt 部分设为 -100，回复部分为对应 token ID

            Step 9: 转换为张量并返回
                    将所有序列和 mask 转换为张量
                    封装为字典返回
        """
        pass

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批次整理函数。

        Args:
            batch: 样本列表

        Returns:
            批处理后的字典

        注意:
        - chosen 和 rejected 长度可能不同
        - 需要分别 padding
        """
        pass


class ConversationalDPODataset(DPODataset):
    """
    对话格式的 DPO 数据集

    处理多轮对话场景下的偏好数据。

    数据格式:
        {
            "conversation": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮你的？"},
                {"role": "user", "content": "讲个笑话"},
            ],
            "chosen": {"role": "assistant", "content": "好的，这是一个笑话..."},
            "rejected": {"role": "assistant", "content": "我不会讲笑话。"}
        }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        chat_template: Optional[str] = None,
    ):
        """
        初始化。

        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大长度
            chat_template: 对话模板 (如 ChatML)
        """
        super().__init__(data_path, tokenizer, max_length)
        self.chat_template = chat_template

    def _format_conversation(self, conversation: List[Dict]) -> str:
        """
        格式化对话历史。

        Args:
            conversation: 对话列表

        Returns:
            格式化后的对话字符串
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本。"""
        pass
