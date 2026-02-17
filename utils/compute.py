"""
计算与显存估算模块

本模块提供 LLM 训练和推理过程中的计算复杂度分析和显存估算功能。

核心功能:
1. FLOPs 计算: 估算训练和推理所需的浮点运算次数
2. 显存估算: 估算模型训练和推理所需的 GPU 显存

重要性:
- FLOPs 是衡量模型计算量的核心指标，用于估算训练时间和成本
- 显存估算是实际部署的基础，决定能否在给定硬件上训练/推理
- 这些估算帮助我们在训练前确定可行的 batch size 和模型规模

参考来源:
- "Training Compute-Optimal Large Language Models" (Chinchilla Paper)
- "FLOPs as a More Relevant Metric than Parameters"
- Microsoft DeepSpeed, NVIDIA Megatron 文档
"""

import math
from typing import Literal


def estimate_train_flops(
    model_params: int,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    use_gqa: bool = False,
    num_kv_heads: int = None,
) -> int:
    """
    估算训练阶段的 FLOPs (Floating Point Operations)

    训练一个 token 所需的 FLOPs 估算公式:
        - Attention 部分: O(batch * seq^2 * d_model * num_heads)
        - FFN 部分: O(batch * seq * d_model * d_ffn)
        - 总计约: 6 * N * batch * seq (经验公式)

    其中系数 6 的来源:
        - 前向传播: ~2 FLOPs per parameter (矩阵乘法)
        - 反向传播: ~4 FLOPs per parameter (需要计算输入、权重、输出的梯度)

    Args:
        model_params: 模型参数量
        batch_size: 批次大小
        seq_len: 序列长度
        num_layers: Transformer 层数
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        intermediate_size: FFN 中间层维度
        use_gqa: 是否使用 GQA (Grouped Query Attention)
        num_kv_heads: KV 头的数量 (GQA 时小于 num_heads)

    Returns:
        训练一个 token 所需的 FLOPs

    计算步骤:
        Step 1: 理解 FLOPs 的基本概念
            FLOPs (Floating Point Operations) 衡量浮点运算的次数
            核心运算：矩阵乘法
            矩阵乘法说明：对于 A[m×k] 和 B[k×n] 的矩阵相乘
            计算量：每个输出元素需要 k 次乘法和 k 次加法
            总计：m × n × k 次运算，约等于 2 × m × n × k FLOPs

        Step 2: 估算 Attention 部分的 FLOPs
            Q、K、V 投影：
            将隐藏状态通过三个独立线性变换映射到 Query、Key、Value 空间
            每个变换的计算量：batch × seq × d_model × d_model
            三个变换总计：3 × batch × seq × d_model²

            Attention Score 计算：
            Q 与 K^T 的矩阵乘法
            计算量：batch × heads × seq × seq × head_dim

            Attention 加权计算：
            softmax 归一化后的权重与 V 相乘
            计算量：与 Score 计算相同

            输出投影：
            将多头注意力结果映射回原始维度
            计算量：batch × seq × d_model × d_model

            注意: GQA 时 K、V 头数少于 Q 头数，计算量相应减少

        Step 3: 估算 FFN 部分的 FLOPs
            FFN 包含两个线性层（SwiGLU 激活）
            第一个线性层：上投影，维度从 d_model 扩展到 d_ffn
            第二个线性层：下投影，维度从 d_ffn 收缩回 d_model
            每层的计算量：batch × seq × 输入维度 × 输出维度

        Step 4: 考虑其他操作的 FLOPs
            激活函数（如 SwiGLU）：包含乘法和 sigmoid 运算
            归一化（如 RMSNorm）：计算均方根并应用缩放
            残差连接：主要是加法运算

        Step 5: 应用经验公式
            对于参数量为 N 的大模型，训练每个 token 约需 6×N 次 FLOPs
            原因分析：
            - 前向传播：每个参数约 2 次 FLOPs（矩阵乘法）
            - 反向传播：每个参数约 4 次 FLOPs（需要同时计算参数、输入、输出的梯度）
            验证示例：7B 模型 → 约 42G FLOPs/token

        Step 6: 考虑 GQA 的优化
            GQA 通过共享 K、V 投影矩阵来减少计算量
            减少比例与 KV 头数成反比

        Step 7: 计算总 FLOPs
            单个 token 的 FLOPs × 批次大小 = 总 FLOPs
    """
    pass


def estimate_inference_flops(
    model_params: int,
    batch_size: int,
    seq_len: int,
    use_kv_cache: bool = True,
) -> int:
    """
    估算推理阶段的 FLOPs

    推理与训练不同:
    1. 自回归生成：每个新 token 需要计算整个序列的 Attention
    2. 使用 KV Cache：已计算的 K、V 不需要重新计算
    3. 第一个 token (prefill) 计算量大，后续 tokens (decode) 计算量小

    Args:
        model_params: 模型参数量
        batch_size: 批次大小
        seq_len: 序列长度
        use_kv_cache: 是否使用 KV Cache 优化

    Returns:
        推理时的总 FLOPs

    计算步骤:
        Step 1: 区分两种推理阶段
            - Prefill 阶段: 处理完整输入序列，计算第一个 token
              计算量 ≈ 训练一个 token 的 FLOPs
            - Decode 阶段: 逐个生成后续 tokens
              使用 KV Cache 后，计算量大幅减少

        Step 2: Prefill 阶段 FLOPs
            与训练类似，每个 token 约 2 * model_params FLOPs
            total = batch_size * seq_len * 2 * model_params

        Step 3: Decode 阶段 FLOPs (使用 KV Cache)
            由于 K、V 已缓存，只需计算:
            - Q 投影: batch * d_model^2
            - Attention: batch * seq * d_k (不是 seq^2!)
            - 输出投影: batch * d_model^2
            总计约: 2 * batch * d_model^2 + batch * seq * d_k

        Step 4: 不使用 KV Cache 的情况
            每次都需要完整计算 Attention: batch * seq^2 * d_k
            当 seq 很长时，这会成为瓶颈

        Step 5: 返回总 FLOPs
            典型的推理过程: 1 次 prefill + N 次 decode
    """
    pass


def estimate_train_memory(
    model_params: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    precision: Literal["fp32", "fp16", "bf16", "fp8"] = "fp16",
    optimizer: Literal["adam", "adamw", "lion"] = "adamw",
    use_gradient_checkpointing: bool = False,
    use_flash_attention: bool = True,
) -> int:
    """
    估算训练阶段的 GPU 显存需求

    显存组成:
        1. 模型参数 (Parameters)
        2. 梯度 (Gradients)
        3. 优化器状态 (Optimizer States) - 最重要！
        4. 激活值 (Activations) - 最大的部分
        5. 临时缓冲区 (Temporary Buffers)

    公式:
        Total = Parameters + Gradients + Optimizer States + Activations

    Args:
        model_params: 模型参数量
        batch_size: 批次大小
        seq_len: 序列长度
        hidden_size: 隐藏层维度
        num_layers: 层数
        precision: 训练精度 (fp32, fp16, bf16, fp8)
        optimizer: 优化器类型
        use_gradient_checkpointing: 是否使用梯度检查点
        use_flash_attention: 是否使用 Flash Attention

    Returns:
        估算的显存需求 (字节数)

    计算步骤:
        Step 1: 确定各精度的字节数
            不同浮点数格式占用不同字节数：
            - fp32（单精度）：4 字节
            - fp16/bf16（半精度）：2 字节
            - fp8（八位浮点）：1 字节

            bf16 与 fp16 的区别：
            - 两者都是 2 字节，但分配给指数和尾数的位数不同
            - fp16：1 位符号 + 5 位指数 + 10 位尾数（精度高但动态范围小）
            - bf16：1 位符号 + 8 位指数 + 7 位尾数（动态范围大，训练更稳定）

        Step 2: 计算模型参数的显存
            模型参数显存 = 模型参数量 × 每参数字节数
            注意：某些特殊层（如 LayerNorm 的缩放参数）可能使用不同精度

        Step 3: 计算梯度的显存
            梯度与模型参数形状完全相同
            显存需求与模型参数相当

        Step 4: 计算优化器状态的显存（显存占比最大！）
            Adam/AdamW 优化器需要保存两个辅助状态：
            - 一阶动量 (m)：用于动量优化，形状与参数相同
            - 二阶动量 (v)：用于自适应学习率，形状与参数相同
            由于优化器状态通常用 fp32 存储：
            显存 ≈ 2 × 参数量 × 4 字节 = 8 × 参数量 字节

            Lion 优化器的优势：
            只需保存一个状态（一阶动量）
            显存约为 Adam 的一半

            高级优化器：
            8-bit Adam 等通过量化压缩优化器状态，可显著减少显存

        Step 5: 计算激活值的显存（变化范围最大）
            激活值是训练时显存消耗最大的部分
            原因：需要保存前向传播中的所有中间结果用于反向传播计算

            估算方法：
            与批次大小、序列长度、隐藏维度、层数成正比

            Attention 部分的特殊情况：
            完整注意力矩阵大小：batch × heads × seq × seq
            当序列长度增加时，显存呈平方级增长！

            优化技术：
            - Flash Attention：通过分块计算避免保存完整注意力矩阵
              显存复杂度从 O(seq²) 降低到 O(seq)
            - Gradient Checkpointing（梯度检查点）：
              不保存中间激活，在反向传播时重新计算
              显存减少约 2-3 倍，计算时间增加约 20-30%

        Step 6: 考虑其他显存消耗
            - CUDA 运行时开销（临时缓冲区等）：约 1GB
            - KV Cache 预分配：与批次、最大序列长度、层数成正比
            - 分布式训练：梯度分片、ZeRO 等技术会改变显存分布

        Step 7: 计算总显存
            总显存 = 模型参数 + 梯度 + 优化器状态 + 激活值 + 其他开销
    """
    pass


def estimate_inference_memory(
    model_params: int,
    batch_size: int,
    max_seq_len: int,
    hidden_size: int,
    num_layers: int,
    precision: Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"] = "fp16",
    use_kv_cache: bool = True,
) -> int:
    """
    估算推理阶段的 GPU 显存需求

    推理显存组成:
        1. 模型参数 (必须加载)
        2. KV Cache (如果启用)
        3. 激活值 (较小)

    Args:
        model_params: 模型参数量
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        hidden_size: 隐藏层维度
        num_layers: 层数
        precision: 推理精度
        use_kv_cache: 是否使用 KV Cache

    Returns:
        估算的推理显存 (字节数)

    计算步骤:
        Step 1: 模型参数显存
            与训练类似，但推理时可以使用更激进的量化
            因为推理不需要梯度、优化器状态等

        Step 2: KV Cache 显存
            推理时最关键的显存消耗
            原因：自回归生成需要保存所有历史位置的 Key 和 Value

            计算方式：
            每层需要保存 K 和 V 两个矩阵
            每层的 K/V 大小：batch × max_seq_len × hidden_size
            总计：2 × batch × max_seq_len × 层数 × hidden_size × 字节数

            优化技术：
            - KV Cache 量化：减少每个元素的字节数
            - PageAttention（vLLM）：分页管理，非连续存储

        Step 3: 激活值显存
            推理时激活值比训练时小很多
            因为每次只处理一个 token（decode 阶段）

        Step 4: 考虑量化精度
            量化可以显著减少显存：
            - int8：每参数 1 字节
            - int4：每参数 0.5 字节
            注意：int4 推理时需要先反量化，会略微增加计算时间

        Step 5: 计算并返回总显存
    """
    pass


def estimate_training_time(
    flops_per_token: int,
    batch_size: int,
    num_tokens: int,
    num_gpus: int = 1,
    gpu_tflops: float = 312,  # H100 SXM 理论性能 (TFLOPS)
    utilization: float = 0.5,  # 实际利用率 (通常 30-60%)
) -> float:
    """
    估算训练时间

    Args:
        flops_per_token: 每个 token 的 FLOPs
        batch_size: 批次大小
        num_tokens: 总 token 数
        num_gpus: GPU 数量
        gpu_tflops: 单 GPU 理论 TFLOPS
        utilization: GPU 利用率 (通常 30-60%)

    Returns:
        估算的训练时间 (小时)

    计算步骤:
        Step 1: 计算每秒有效计算性能
            有效 TFLOPS = GPU 理论性能 × 利用率 × GPU 数量
            说明：
            - GPU 理论性能：GPU 规格给出的峰值浮点运算能力
            - 利用率：实际运行效率（通常 30-60%）
              受内存带宽、通信开销、代码效率等因素影响
            - 多 GPU：通过数据并行或模型并行提升

        Step 2: 计算训练所需的总计算量
            总 FLOPs = 每 token FLOPs × 总 token 数

        Step 3: 转换为时间
            时间 = 总 FLOPs / (有效 FLOPs × 3600 秒)
            单位换算：1 TFLOPS = 10¹² FLOPs/秒

        Step 4: 考虑数据并行通信开销
            多 GPU 训练时需要同步梯度，产生额外通信开销
            实际时间通常增加 5-15%

        Step 5: 返回估算时间（小时）

    计算示例:
        假设条件：
        - 模型：7B 参数
        - 训练数据：1T tokens
        - GPU：8 张 H100（每张 312 TFLOPS）
        - 利用率：50%

        计算过程：
        - 每 token FLOPs ≈ 6 × 7×10⁹ = 4.2×10¹⁰
        - 总 FLOPs = 4.2×10¹⁰ × 10¹² = 4.2×10²²
        - 有效性能 = 312 × 0.5 × 8 = 1248 TFLOPS
        - 估算时间 ≈ 4.2×10²² / (1.248×10¹⁵ × 3600) ≈ 0.94 小时
    """
    pass


def print_memory_breakdown(
    model_params: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    precision: str = "fp16",
    optimizer: str = "adamw",
):
    """
    打印详细的显存分解报告

    这是一个辅助函数，帮助理解各部分显存的占比

    Args:
        model_params: 模型参数量
        batch_size: 批次大小
        seq_len: 序列长度
        hidden_size: 隐藏层维度
        num_layers: 层数
        precision: 训练精度
        optimizer: 优化器类型

    输出示例:
        ==================== 显存估算报告 ====================
        模型: 7B 参数
        精度: fp16
        批次: 4, 序列长度: 4096
        ------------------------------------------------
        模型参数:       14.00 GB (  8.4%)
        梯度:           14.00 GB (  8.4%)
        优化器状态:     56.00 GB ( 33.5%)
        激活值:         82.35 GB ( 49.3%)
        其他:            0.97 GB (  0.6%)
        ------------------------------------------------
        总计:          167.32 GB (100.0%)
        需要:           2 x 80GB A100
    """
    pass
