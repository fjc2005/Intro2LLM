"""
HumanEval 评估模块
测试模型的代码生成能力。

HumanEval 数据集:
- 包含 164 个手写编程问题
- 每个问题包含函数签名、docstring 和若干测试用例
- 测试模型生成通过所有测试的代码能力

评估方式:
1. 模型根据函数签名和 docstring 生成函数体
2. 执行生成的代码
3. 运行测试用例
4. 计算 pass@k 指标

论文: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import re
from collections import defaultdict


class HumanEvalEvaluator:
    """
    HumanEval 评估器

    评估模型的代码生成能力。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
    ):
        """
        初始化 HumanEval 评估器。

        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, prompt: str) -> str:
        """
        格式化代码生成 prompt。

        Args:
            prompt: 包含函数签名和 docstring 的代码前缀

        Returns:
            格式化后的 prompt

        格式示例:
            def has_close_elements(numbers: List[float], threshold: float) -> bool:
                \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
                given threshold.
                >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
                False
                >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
                True
                \"\"\"

        注意: 模型需要生成函数体实现
        """
        pass

    def generate_code(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_samples: int = 1,
    ) -> List[str]:
        """
        生成代码补全。

        Args:
            prompt: 代码前缀
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            num_samples: 生成样本数 (用于 pass@k)

        Returns:
            生成的代码列表

        生成流程:
            Step 1: 编码 prompt
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")

            Step 2: 生成代码
                    with torch.no_grad():
                        for _ in range(num_samples):
                            output = model.generate(
                                input_ids,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                stop_sequences=["\ndef", "\nclass", "\nif"],  # 停止标志
                            )

            Step 3: 解码并提取代码
                    generated_code = tokenizer.decode(output[0][len(input_ids[0]):])
                    # 截断到第一个停止标志

            Step 4: 返回
        """
        pass

    def execute_code(
        self,
        code: str,
        test_code: str,
        timeout: int = 5,
    ) -> bool:
        """
        执行代码并运行测试。

        Args:
            code: 生成的函数代码
            test_code: 测试代码
            timeout: 执行超时时间 (秒)

        Returns:
            是否通过所有测试

        执行流程:
            Step 1: 组合完整代码
                    full_code = code + "\n" + test_code

            Step 2: 在安全环境中执行
                    try:
                        exec(full_code, {})
                        return True
                    except Exception as e:
                        return False

        注意:
            - 需要安全沙箱执行用户生成的代码
            - 设置超时防止无限循环
        """
        pass

    def compute_pass_at_k(
        self,
        n: int,
        c: int,
        k: int,
    ) -> float:
        """
        计算 pass@k 指标。

        pass@k: 从 n 个生成样本中随机选 k 个，至少有一个通过的概率

        公式:
            pass@k = 1 - C(n-c, k) / C(n, k)

        其中:
            - n: 总生成样本数
            - c: 通过的样本数
            - k: k 值

        当 n 较大时，使用近似:
            pass@k = 1 - (1 - c/n)^k

        Args:
            n: 总样本数
            c: 通过样本数
            k: k 值

        Returns:
            pass@k 值
        """
        pass

    def evaluate(
        self,
        test_data: List[Dict],
        num_samples_per_task: int = 200,
        k_values: List[int] = [1, 10, 100],
    ) -> Dict[str, float]:
        """
        评估模型在 HumanEval 上的表现。

        Args:
            test_data: HumanEval 测试数据
            num_samples_per_task: 每个任务生成的样本数
            k_values: 计算的 k 值列表

        Returns:
            包含以下字段的字典:
                - pass@1: pass@1 指标
                - pass@10: pass@10 指标
                - pass@100: pass@100 指标
                - results_by_task: 每个任务的结果

        评估流程:
            Step 1: 初始化结果统计
                    results = []

            Step 2: 遍历每个任务
                    for task in test_data:
                        task_id = task["task_id"]
                        prompt = task["prompt"]
                        test_code = task["test"]
                        entry_point = task["entry_point"]

            Step 3: 生成多个代码样本
                        generated_codes = generate_code(
                            prompt,
                            num_samples=num_samples_per_task
                        )

            Step 4: 测试每个样本
                        passed = 0
                        for code in generated_codes:
                            # 组合完整函数
                            full_code = prompt + code

                            # 执行测试
                            if execute_code(full_code, test_code):
                                passed += 1

            Step 5: 计算 pass@k
                        for k in k_values:
                            pass_at_k = compute_pass_at_k(
                                num_samples_per_task,
                                passed,
                                k
                            )

            Step 6: 汇总结果
        """
        pass

    def load_humaneval_data(self, data_path: str) -> List[Dict]:
        """
        加载 HumanEval 数据。

        Args:
            data_path: 数据文件路径

        Returns:
            HumanEval 数据列表
        """
        pass
