"""
监督微调数据集模块
用于指令微调 (Instruction Tuning / SFT)。

SFT 任务:
学习遵循指令并生成期望的回复。
- 输入: 指令 (Instruction) + 输入 (可选)
- 目标: 期望的输出 (Output)

数据格式:
通常是 (instruction, input, output) 或 (prompt, completion) 格式。

关键区别:
预训练时所有 token 都参与损失计算。
SFT 时只计算 output 部分的损失，instruction 部分被 mask 掉。
"""

import torch
from typing import Dict, List, Optional, Union
from .dataset import BaseDataset


class SFTDataset(BaseDataset):
    """
    监督微调数据集

    用于指令跟随训练。

    数据格式示例:
        {
            "instruction": "将以下英文翻译成中文",
            "input": "Hello, how are you?",
            "output": "你好，你好吗？"
        }

    或 Conversational 格式:
        {
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你？"},
                {"role": "user", "content": "讲个笑话"},
                {"role": "assistant", "content": "好的，这是一个笑话..."}
            ]
        }

    样本处理:
        1. 构造完整 prompt:
           prompt = format_prompt(instruction, input)

           例如使用 ChatML 格式:
           "<|im_start|>user\n将英文翻译成中文: Hello\n<|im_end|>\n"
           "<|im_start|>assistant\n你好\n<|im_end|>"

        2. 编码完整序列:
           full_text = prompt + output + eos_token

        3. 创建 labels:
           - prompt 部分设为 -100 (不参与损失计算)
           - output 部分设为对应 token id
           - eos_token 参与计算

        示例:
           tokens:     [BOS] 你 好 吗 ？[EOS]
           labels:     [-100 -100 -100 -100 id(？) id([EOS])]
                       ↑ prompt ↑        ↑ response ↑
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        template: Optional[str] = None,
    ):
        """
        初始化 SFT 数据集。

        Args:
            data_path: 数据路径 (.jsonl)
            tokenizer: 分词器
            max_length: 最大序列长度
            template: 提示词模板，用于格式化 instruction 和 input
                     例如: "{instruction}\n\nInput: {input}\n\nOutput: "

        模板示例 (Alpaca):
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"

        模板示例 (ChatML):
            "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        """
        super().__init__(data_path, tokenizer, max_length)
        # 保存模板
        # 解析数据格式 (alpaca / sharegpt / conversational)
        pass

    def _format_sample(self, sample: Dict) -> Dict[str, str]:
        """
        格式化单个样本。

        Args:
            sample: 原始数据样本

        Returns:
            {"prompt": "...", "completion": "..."}

        支持的格式:
            1. Alpaca 格式:
               {"instruction": "...", "input": "...", "output": "..."}

            2. Prompt-Completion 格式:
               {"prompt": "...", "completion": "..."}

            3. Conversational 格式 (ShareGPT):
               {"messages": [{"role": "...", "content": "..."}, ...]}

        处理逻辑:
            Step 1: 检测数据格式
            Step 2: 根据格式提取 prompt 和 completion
            Step 3: 应用模板格式化 prompt
            Step 4: 返回统一格式的字典
        """
        pass

    def _create_labels(
        self,
        input_ids: List[int],
        prompt_len: int,
    ) -> List[int]:
        """
        创建 labels，mask 掉 prompt 部分。

        Args:
            input_ids: 完整序列的 token IDs
            prompt_len: prompt 部分的长度 (token 数)

        Returns:
            labels 列表，prompt 部分为 -100，其余为 input_ids 对应值

        示例:
            input_ids:  [BOS, T1, T2, T3, T4, T5, EOS]
            prompt_len: 4 (BOS + 3个 prompt tokens)
            labels:     [-100, -100, -100, -100, T5, T6, T7]
                                      ↑ prompt结束 ↑

        处理流程:
            创建一个与 input_ids 长度相同的列表
            前 prompt_len 个位置设为 -100 (忽略标记)
            剩余位置设为对应 input_ids 的值
            这样只有 response 部分参与损失计算

        注意:
            - prompt_len 需要准确计算，不能简单按字符数
            - 需要使用 tokenizer 预先编码 prompt 来确定长度
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取 SFT 样本。

        Args:
            idx: 样本索引

        Returns:
            包含以下字段的字典:
                - input_ids: [seq_len]，完整序列
                - attention_mask: [seq_len]，1 表示有效，0 表示 padding
                - labels: [seq_len]，-100 表示 mask，其余为预测目标

        流程:
            Step 1: 获取原始样本并格式化
                    根据索引从数据集中获取样本
                    调用 _format_sample 方法将样本格式化为 prompt 和 completion

            Step 2: 编码 prompt 和 completion
                    使用 tokenizer 分别编码格式化后的 prompt 和 completion
                    获取对应的 token ID 列表

            Step 3: 拼接完整序列
                    将 prompt token IDs、completion token IDs 和 EOS token ID 拼接
                    形成完整的 input_ids 序列

            Step 4: 检查长度并截断
                    如果序列长度超过 max_length:
                        优先截断 prompt 部分以保留 completion
                        或者整体截断到 max_length
                    确保截断后序列不为空

            Step 5: 创建 labels
                    调用 _create_labels 方法创建 labels
                    传入 input_ids 和 prompt 的长度
                    prompt 部分会被 mask 为 -100

            Step 6: 转换为张量并返回
                    将 input_ids、attention_mask 和 labels 转换为张量
                    封装为字典返回
        """
        pass

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批次整理函数。

        SFT 数据的 collate 需要:
        1. 对不同长度的序列进行 padding
        2. padding 位置在 labels 中也设为 -100
        3. 创建 attention_mask

        Args:
            batch: 样本列表

        Returns:
            批处理后的字典

        流程:
            Step 1: 找出批次内最大长度
            Step 2: 对每个样本进行 padding
                    input_ids: pad with pad_token_id
                    attention_mask: pad with 0
                    labels: pad with -100
            Step 3: 堆叠成张量
            Step 4: 返回
        """
        pass


class ConversationDataset(BaseDataset):
    """
    对话格式数据集

    处理多轮对话数据，支持更复杂的对话场景。

    数据格式:
        {
            "messages": [
                {"role": "system", "content": "你是一个 helpful 助手"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
                {"role": "user", "content": "今天天气如何？"},
                {"role": "assistant", "content": "我无法获取实时天气信息..."}
            ]
        }

    训练目标:
    只学习生成 assistant 的回复，mask 掉 system 和 user 的内容。

    Loss Mask 策略:
        完整序列: [SYSTEM] [USER1] [ASSISTANT1] [USER2] [ASSISTANT2]
        labels:   [-100]  [-100]  [tokens...]   [-100]  [tokens...]
                               ↑ 只训练助手回复
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        chat_template: Optional[str] = None,
    ):
        """
        初始化对话数据集。

        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大长度
            chat_template: 对话模板，如 ChatML、Llama-2-chat 等
        """
        super().__init__(data_path, tokenizer, max_length)
        pass

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        应用对话模板将消息列表转换为字符串。

        ChatML 模板示例:
            "<|im_start|>system\n{system_message}<|im_end|>\n"
            "<|im_start|>user\n{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"

        Returns:
            格式化后的字符串
        """
        pass

    def _compute_loss_mask(
        self,
        input_ids: List[int],
        message_boundaries: List[int],
    ) -> List[int]:
        """
        计算 loss mask。

        Args:
            input_ids: 完整序列
            message_boundaries: 每条消息的边界位置

        Returns:
            labels，只保留 assistant 消息对应的 token
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取对话样本。

        处理多轮对话，创建正确的 loss mask。
        """
        pass
