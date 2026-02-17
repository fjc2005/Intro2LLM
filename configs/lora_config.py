"""
LoRA 和 QLoRA 配置模块
定义 LoRA/QLoRA 微调的配置参数。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class LoRAConfig:
    """
    LoRA 配置类

    Attributes:
        r: LoRA 秩，控制低秩矩阵的维度
        lora_alpha: LoRA 缩放参数
        lora_dropout: LoRA 层的 dropout 概率
        target_modules: 要添加 LoRA 的目标模块名列表
        bias: 是否训练 bias 参数
        modules_to_save: 除 LoRA 外还要训练的其他模块
        init_lora_weights: LoRA 权重初始化方式
    """

    # 核心 LoRA 参数
    r: int = 8
    """LoRA 秩，通常 4-64，越大表达能力越强但参数量越多"""

    lora_alpha: int = 16
    """LoRA 缩放参数，缩放因子 = lora_alpha / r"""

    lora_dropout: float = 0.0
    """LoRA 层的 dropout 概率，用于正则化"""

    # 目标模块配置
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj"
    ])
    """
    要添加 LoRA 的目标模块名列表

    常见模块名:
        - q_proj: Query 投影
        - k_proj: Key 投影
        - v_proj: Value 投影
        - o_proj: Output 投影
        - gate_proj: FFN 门控投影
        - up_proj: FFN 上采样投影
        - down_proj: FFN 下采样投影
        - embed_tokens: Token 嵌入
        - lm_head: 语言模型头

    推荐配置:
        - 最省参数: ["q_proj", "v_proj"]
        - 平衡: ["q_proj", "k_proj", "v_proj", "o_proj"]
        - 最强表达: 所有线性层
    """

    # 其他配置
    bias: str = "none"
    """是否训练 bias，可选: 'none', 'all', 'lora_only'"""

    modules_to_save: Optional[List[str]] = None
    """除 LoRA 外还要训练的其他模块，如 ['embed_tokens', 'lm_head']"""

    init_lora_weights: str = "gaussian"
    """LoRA 权重初始化方式，可选: 'gaussian', 'loftq'"""

    # 推理配置
    merge_weights: bool = False
    """是否合并 LoRA 权重到原模型，合并后无推理开销"""

    def __post_init__(self):
        """配置验证和计算派生参数。"""
        # 验证参数
        assert self.r > 0, "r 必须为正整数"
        assert self.lora_alpha >= 0, "lora_alpha 必须为非负数"
        assert 0 <= self.lora_dropout < 1, "lora_dropout 必须在 [0, 1) 范围内"
        assert self.bias in ["none", "all", "lora_only"], "bias 必须是 'none', 'all', 或 'lora_only'"

    @property
    def scaling(self) -> float:
        """计算 LoRA 缩放因子。"""
        pass

    def to_dict(self) -> Dict:
        """转换为字典。"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "LoRAConfig":
        """从字典创建配置。"""
        pass


@dataclass
class QLoRAConfig(LoRAConfig):
    """
    QLoRA 配置类

    继承 LoRAConfig，添加量化相关参数。
    """

    # QLoRA 特定参数
    r: int = 64
    """QLoRA 通常使用更大的秩，如 64"""

    lora_alpha: int = 16

    lora_dropout: float = 0.1
    """QLoRA 通常使用稍大的 dropout"""

    # 量化配置
    load_in_4bit: bool = True
    """是否使用 4-bit 量化"""

    bnb_4bit_quant_type: str = "nf4"
    """
    4-bit 量化类型

    可选:
        - "nf4": Normal Float 4，推荐用于权重
        - "fp4": 标准 4-bit 浮点
    """

    bnb_4bit_compute_dtype: str = "bfloat16"
    """
    计算数据类型

    可选: "bfloat16", "float16", "float32"
    推荐使用 bfloat16 (如果有 Tensor Cores)
    """

    bnb_4bit_use_double_quant: bool = True
    """是否使用双重量化，进一步节省显存"""

    # 分页优化器
    use_page_optimizer: bool = True
    """是否使用分页优化器，处理大梯度"""

    # 梯度检查点
    gradient_checkpointing: bool = True
    """是否启用梯度检查点，节省显存"""

    # 最大内存配置
    max_memory: Optional[Dict[int, str]] = None
    """
    每 GPU 的最大内存配置

    示例:
        {0: "24GB", 1: "24GB"}
    """

    def __post_init__(self):
        """QLoRA 配置验证。"""
        super().__post_init__()
        assert self.bnb_4bit_quant_type in ["nf4", "fp4"], \
            "bnb_4bit_quant_type 必须是 'nf4' 或 'fp4'"
        assert self.bnb_4bit_compute_dtype in ["bfloat16", "float16", "float32"], \
            "bnb_4bit_compute_dtype 必须是 'bfloat16', 'float16', 或 'float32'"

    def get_bnb_config(self) -> object:
        """
        获取 bitsandbytes 配置对象。

        Returns:
            BitsAndBytesConfig 对象 (如果可用)
        """
        pass


# 预定义配置
class LoRAPresets:
    """
    LoRA 预定义配置

    提供针对不同场景的推荐配置。
    """

    @staticmethod
    def minimal() -> LoRAConfig:
        """
        最小参数配置。

        适用于:
        - 资源受限环境
        - 快速实验
        - 简单任务

        可训练参数量: ~0.1%
        """
        pass

    @staticmethod
    def balanced() -> LoRAConfig:
        """
        平衡配置。

        适用于大多数任务的默认配置。

        可训练参数量: ~0.5%
        """
        pass

    @staticmethod
    def aggressive() -> LoRAConfig:
        """
        激进配置。

        适用于:
        - 复杂任务
        - 需要更强表达能力
        - 显存充足

        可训练参数量: ~1-2%
        """
        pass

    @staticmethod
    def qlora_standard() -> QLoRAConfig:
        """
        标准 QLoRA 配置。

        适用于在消费级 GPU 上微调大模型。
        """
        pass

    @staticmethod
    def qlora_max_memory() -> QLoRAConfig:
        """
        最大显存节省配置。

        适用于显存极度受限的情况。
        """
        pass
