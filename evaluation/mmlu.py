"""
MMLU (Massive Multitask Language Understanding) 评估模块
测试模型在多学科知识上的能力。

MMLU 数据集:
- 涵盖 57 个学科，包括 STEM、人文、社科等
- 每个问题有 4 个选项 (A/B/C/D)
- 测试模型的知识广度和推理能力

评估方式:
1. 将问题和选项格式化为 prompt
2. 模型预测下一个 token 是 A/B/C/D 的概率
3. 选择概率最高的选项作为答案
4. 计算准确率

论文: "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2020)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict
import json


class MMLUEvaluator:
    """
    MMLU 评估器

    评估模型在 MMLU 基准上的表现。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
    ):
        """
        初始化 MMLU 评估器。

        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(
        self,
        question: str,
        choices: List[str],
        include_answer: bool = False,
        answer: Optional[str] = None,
    ) -> str:
        """
        格式化 MMLU 问题为 prompt。

        Args:
            question: 问题文本
            choices: 选项列表 [A, B, C, D]
            include_answer: 是否包含答案 (用于 few-shot)
            answer: 答案 (A/B/C/D)

        Returns:
            格式化后的 prompt

        格式示例:
            Question: What is the capital of France?
            A. London
            B. Berlin
            C. Paris
            D. Madrid
            Answer:
        """
        pass

    def evaluate_sample(
        self,
        question: str,
        choices: List[str],
    ) -> str:
        """
        评估单个样本。

        Args:
            question: 问题文本
            choices: 选项列表

        Returns:
            模型选择的答案 (A/B/C/D)

        评估流程:
            Step 1: 格式化 prompt (不包含答案)
                    prompt = format_prompt(question, choices)

            Step 2: 编码 prompt
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")

            Step 3: 获取模型预测
                    with torch.no_grad():
                        logits = model(input_ids).logits

            Step 4: 获取选项 token 的概率
                    # 获取最后一个位置对 A/B/C/D 的 logits
                    last_token_logits = logits[0, -1, :]
                    option_logits = [
                        last_token_logits[tokenizer.encode("A")[0]],
                        last_token_logits[tokenizer.encode("B")[0]],
                        last_token_logits[tokenizer.encode("C")[0]],
                        last_token_logits[tokenizer.encode("D")[0]],
                    ]

            Step 5: 选择概率最高的选项
                    predicted_idx = argmax(option_logits)
                    answer = ["A", "B", "C", "D"][predicted_idx]

            Step 6: 返回
        """
        pass

    def evaluate(
        self,
        test_data: List[Dict],
        num_few_shot: int = 0,
        few_shot_data: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        评估模型在 MMLU 数据集上的表现。

        Args:
            test_data: 测试数据列表
            num_few_shot: few-shot 示例数量
            few_shot_data: few-shot 示例数据

        Returns:
            包含以下字段的字典:
                - overall_accuracy: 总体准确率
                - accuracy_by_subject: 各学科的准确率
                - num_questions: 问题总数

        评估流程:
            Step 1: 准备 few-shot prompt (如果需要)
                    few_shot_prompt = ""
                    if num_few_shot > 0:
                        for sample in few_shot_data[:num_few_shot]:
                            few_shot_prompt += format_prompt(
                                sample["question"],
                                sample["choices"],
                                include_answer=True,
                                answer=sample["answer"]
                            )

            Step 2: 按学科分组统计
                    results_by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

            Step 3: 遍历测试数据
                    for sample in test_data:
                        # 构建完整 prompt
                        prompt = few_shot_prompt + format_prompt(
                            sample["question"],
                            sample["choices"]
                        )

                        # 获取预测
                        predicted = evaluate_sample(sample["question"], sample["choices"])

                        # 统计
                        subject = sample["subject"]
                        results_by_subject[subject]["total"] += 1
                        if predicted == sample["answer"]:
                            results_by_subject[subject]["correct"] += 1

            Step 4: 计算准确率
                    for subject in results_by_subject:
                        acc = correct / total
                        accuracy_by_subject[subject] = acc

                    overall_accuracy = total_correct / total_questions

            Step 5: 返回结果
        """
        pass

    def load_mmlu_data(self, data_path: str, split: str = "test") -> List[Dict]:
        """
        加载 MMLU 数据。

        Args:
            data_path: 数据路径
            split: 数据分割 (test/val/dev)

        Returns:
            数据列表
        """
        pass
