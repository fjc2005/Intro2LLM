"""
数据工程模块

本模块提供 LLM 训练数据的处理功能，包括：
1. 数据质量过滤 - 过滤低质量、重复、有害的文本
2. 数据去重 - 使用多种方法去除重复内容
3. 数据清洗 - 规范化文本格式
4. 数据采样 - 按比例采样和权重采样

为什么数据工程重要:
- "Data is the new oil" - 训练数据质量直接决定模型能力
- Chinchilla 定律表明：数据质量和数量同样重要
- 现代 LLM 训练需要数万亿 token，高质量数据筛选至关重要

典型数据处理流程:
    原始数据 -> 质量过滤 -> 去重 -> 清洗 -> 采样 -> 训练数据
"""

import re
import hashlib
from typing import Callable, Optional
from collections import Counter


class DataFilter:
    """
    数据过滤器基类

    所有过滤器都继承这个基类，实现 filter 方法
    """

    def __init__(self, min_length: int = 0, max_length: int = float('inf')):
        """
        初始化过滤器

        Args:
            min_length: 最小文本长度 (字符数)
            max_length: 最大文本长度 (字符数)
        """
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, text: str) -> bool:
        """
        判断文本是否通过过滤

        Args:
            text: 输入文本

        Returns:
            True 表示通过过滤，False 表示被过滤掉
        """
        raise NotImplementedError


class LengthFilter(DataFilter):
    """
    长度过滤器

    过滤掉过短或过长的文本
    """

    def filter(self, text: str) -> bool:
        """
        根据长度过滤文本

        步骤:
            Step 1: 计算文本长度 (按字符数计算)
            Step 2: 检查是否在 [min_length, max_length] 范围内
            Step 3: 返回判断结果

        经验值:
            - min_length: 10-100 字符 (过滤无意义短文本)
            - max_length: 10000-100000 字符 (过长文本可能是垃圾数据)
        """
        pass


class RepetitionFilter(DataFilter):
    """
    重复率过滤器

    过滤掉重复度过高的文本，例如:
    - "aaaaaaaaaa..."
    - "abc abc abc abc..."
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = float('inf'),
        max_repetition_ratio: float = 0.3,
        n_gram: int = 5,
    ):
        """
        Args:
            min_length: 最小文本长度
            max_length: 最大文本长度
            max_repetition_ratio: 最大重复率阈值
            n_gram: 用于检测的 n-gram 大小

        n-gram 说明:
            n=3 时 "aaa bbb aaa" 中的 "aaa" 出现 2 次
            重复率 = 2 * 3 / 总词数
        """
        super().__init__(min_length, max_length)
        self.max_repetition_ratio = max_repetition_ratio
        self.n_gram = n_gram

    def filter(self, text: str) -> bool:
        """
        过滤重复文本

        步骤:
            Step 1: 将文本分词 (按空格或字符)
            Step 2: 生成 n-gram
            Step 3: 统计每个 n-gram 的出现次数
            Step 4: 计算重复率 = (重复 n-gram 的总长度) / (总长度)
            Step 5: 如果重复率超过阈值，返回 False

        示例:
            text = "the the the the dog"
            n-grams = [("the", "the", "the"), ("the", "the", "dog")]
            counts = {("the","the","the"): 2, ("the","the","dog"): 1}
        """
        pass


class LanguageFilter(DataFilter):
    """
    语言过滤器

    过滤掉非目标语言的文本
    """

    def __init__(
        self,
        target_language: str = "en",
        min_language_ratio: float = 0.9,
    ):
        """
        Args:
            target_language: 目标语言代码 ("en", "zh", "multi")
            min_language_ratio: 文本中目标语言的最小比例

        实现方式:
            - 简单: 使用字符分布统计
            - 中等: 使用 fastText 语言分类器
            - 复杂: 使用 langdetect 库
        """
        self.target_language = target_language
        self.min_language_ratio = min_language_ratio

    def filter(self, text: str) -> bool:
        """
        判断文本是否属于目标语言

        步骤:
            Step 1: 选择语言检测方法
                简单方法: 统计字符分布
                    - 英文字母 a-z 在英文文本中占 70%+
                    - 中文字符在中文文本中占 80%+

            Step 2: 计算文本的语言特征
                方式 A: 统计字母/字符分布
                方式 B: 使用语言检测库

            Step 3: 判断是否满足目标语言比例
                language_ratio = target_lang_char_count / total_char_count

            Step 4: 返回判断结果
        """
        pass


class QualityFilter(DataFilter):
    """
    质量过滤器

    过滤低质量文本，包括:
    - 特殊字符过多
    - 句子结构异常
    - Perplexity 异常
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = float('inf'),
        max_special_char_ratio: float = 0.5,
        min_word_count: int = 5,
    ):
        """
        Args:
            min_length: 最小文本长度
            max_length: 最大文本长度
            max_special_char_ratio: 最大特殊字符比例
            min_word_count: 最小单词数
        """
        super().__init__(min_length, max_length)
        self.max_special_char_ratio = max_special_char_ratio
        self.min_word_count = min_word_count

    def filter(self, text: str) -> bool:
        """
        过滤低质量文本

        步骤:
            Step 1: 检查特殊字符比例
                特殊字符: 非字母、非数字、非空格、非常见标点
                ratio = special_char_count / total_char_count

            Step 2: 检查单词数量
                使用空格分词后检查词数

            Step 3: 检查句子结构 (可选)
                - 检查是否有过多的 URL、Email
                - 检查是否有异常的 HTML 标签

            Step 4: 综合判断
                所有条件都满足才返回 True
        """
        pass


class ToxicityFilter(DataFilter):
    """
    有害内容过滤器

    过滤包含有害内容的文本，例如:
    - 暴力、色情内容
    - 仇恨言论
    - 个人信息 (PII)

    实现方式:
        - 规则匹配: 使用关键词/正则表达式
        - 机器学习: 使用分类器
        - API 服务: 调用外部 API
    """

    def __init__(
        self,
        use_ml_classifier: bool = False,
        threshold: float = 0.5,
    ):
        """
        Args:
            use_ml_classifier: 是否使用机器学习分类器
            threshold: 分类阈值
        """
        self.use_ml_classifier = use_ml_classifier
        self.threshold = threshold

    def filter(self, text: str) -> bool:
        """
        过滤有害内容

        步骤:
            Step 1: 规则过滤 (快速)
                - 关键词黑名单匹配
                - 正则表达式匹配 URL、Email、电话号码

            Step 2: ML 分类 (可选，更准确)
                - 使用预训练的有害内容分类器
                - 计算有害概率

            Step 3: 返回判断结果
                - 规则匹配到 -> False
                - ML 概率 > 阈值 -> False
                - 否则 -> True

        实际应用:
            - 可以使用 pretrained model: toxicity, hate_speech detection
            - 可以使用 preset: word-list based filtering
        """
        pass


class Deduplicator:
    """
    数据去重器

    常用去重方法:
    1. Exact Dedup: 精确匹配，删除完全相同的文档
    2. Fuzzy Dedup: 模糊匹配，删除相似度高的文档
    3. MinHash: 大规模数据的近似去重
    """

    def __init__(self, method: str = "exact"):
        """
        Args:
            method: 去重方法 ("exact", "fuzzy", "minhash")

        各方法特点:
            - exact: 精确匹配，速度快，适合初步去重
            - fuzzy: 相似度匹配，速度慢，但能去除近似重复
            - minhash: 近似匹配，适合大规模数据
        """
        self.method = method
        self.seen_hashes = set()

    def add(self, text: str) -> bool:
        """
        添加一个文档，判断是否重复

        Args:
            text: 输入文本

        Returns:
            True 表示是新文档，False 表示是重复文档
        """
        pass

    def is_duplicate(self, text: str) -> bool:
        """
        检查文本是否重复

        步骤 (Exact Dedup):
            Step 1: 对文本进行标准化处理
                - 将文本转换为统一格式（如转小写）
                - 去除首尾空白字符
                - 可选：去除标点符号

            Step 2: 计算哈希值
                使用哈希函数将标准化后的文本转换为固定长度的字符串
                哈希函数特点：
                - MD5/SHA256：计算快速，产生固定长度输出
                - 相同输入产生相同输出，不同输入极大概率产生不同输出
                具体实现：先对文本进行编码（UTF-8），然后计算其哈希值

            Step 3: 查表判断
                维护一个已见过文档的哈希集合
                如果当前哈希值已存在于集合中，说明文档重复
                否则将哈希值加入集合，并标记为新文档
        """
        pass


class FuzzyDeduplicator(Deduplicator):
    """
    模糊去重器

    使用相似度计算来检测近似重复的文档
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        method: str = "simhash",
    ):
        """
        Args:
            similarity_threshold: 相似度阈值
            method: 相似度计算方法 ("simhash", "minhash", "embedding")

        方法比较:
            - simhash: 适合短文本，速度快
            - minhash: 适合大规模数据，需要多哈希函数
            - embedding: 精度最高，但需要预训练模型
        """
        super().__init__(method="fuzzy")
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        SimHash 方法步骤:
            Step 1: 分词，得到特征和权重
            Step 2: 对每个特征计算哈希
            Step 3: 加权 (特征权重 * 哈希值)
            Step 4: 合并为指纹
            Step 5: 计算海明距离

            相似度 = 1 - hamming_distance / fingerprint_bits

        MinHash 方法步骤:
            Step 1: 将文本转为集合 (shingle)
            Step 2: 使用多个哈希函数计算 MinHash
            Step 3: 相似度 = Jaccard 相似度的估计

        Embedding 方法步骤:
            Step 1: 使用预训练模型编码文本
            Step 2: 计算向量余弦相似度
        """
        pass

    def is_duplicate(self, text: str) -> bool:
        """
        判断是否近似重复

        步骤:
            Step 1: 计算当前文本的指纹/向量
            Step 2: 与已存在的文本比较
            Step 3: 如果任意相似度 >= 阈值，返回 True
            Step 4: 否则添加到集合中
        """
        pass


class DataCleaner:
    """
    数据清洗器

    规范化文本格式
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_html_tags: bool = True,
    ):
        """
        Args:
            lowercase: 是否转小写
            remove_extra_whitespace: 移除多余空格
            normalize_unicode: 统一 Unicode 格式
            remove_html_tags: 移除 HTML 标签
        """
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_html_tags = remove_html_tags

    def clean(self, text: str) -> str:
        """
        清洗文本

        步骤:
            Step 1: 统一 Unicode 编码
                将文本转换为标准的 Unicode 形式
                作用：消除同一字符的不同表示形式（如全角/半角）
                NFKC 规范化：将兼容字符分解后重新组合为标准形式

            Step 2: 移除 HTML 标签
                识别并移除 HTML 标记（如 <p>, <div>, <br> 等）
                原理：匹配尖括号包围的内容并替换为空字符串

            Step 3: 处理空白字符
                将各类换行符（\n, \r, \t）替换为标准空格
                使用连续空格检测算法将多个连续空格合并为单个空格

            Step 4: 转小写 (可选)
                将英文字母转换为小写形式
                作用：统一大小写格式，便于后续处理

            Step 5: 去除首尾空白
                移除字符串开头和结尾的空白字符

            Step 6: 返回清洗后的文本
        """
        pass


class DataSampler:
    """
    数据采样器

    从大数据集中按比例或权重采样
    """

    def __init__(
        self,
        sampling_strategy: str = "uniform",
        weights: Optional[list[float]] = None,
    ):
        """
        Args:
            sampling_strategy: 采样策略 ("uniform", "weighted", "priority")
            weights: 采样权重 (与 sampling_strategy="weighted" 配合使用)

        采样策略:
            - uniform: 均匀采样
            - weighted: 按权重采样，高质量数据权重更高
            - priority: 优先采样某些数据
        """
        self.sampling_strategy = sampling_strategy
        self.weights = weights

    def sample(
        self,
        data: list,
        sample_size: int,
        quality_scores: Optional[list[float]] = None,
    ) -> list:
        """
        从数据中采样

        步骤:
            Step 1: 根据采样策略确定采样方法
                首先判断使用哪种采样策略：均匀、加权还是优先级

            Step 2: 均匀采样
                从数据集中随机选择指定数量的样本
                每个样本被选中的概率相等
                实现方式：随机索引生成或拒绝采样

            Step 3: 加权采样
                根据预先设置的权重或质量分数进行采样
                高权重样本被选中的概率更高
                权重来源：
                - 质量分数（perplexity 越低，质量越高）
                - 来源优先级（官方文档 > 维基百科 > 社交媒体）
                权重归一化：所有权重相加为 1

            Step 4: 返回采样结果
        """
        pass


class Pipeline:
    """
    数据处理流水线

    将多个过滤器、去重器、清洗器组合成流水线
    """

    def __init__(self):
        self.filters = []
        self.deduplicator = None
        self.cleaner = None

    def add_filter(self, filter_obj):
        """添加过滤器"""
        pass

    def set_deduplicator(self, deduplicator):
        """设置去重器"""
        pass

    def set_cleaner(self, cleaner):
        """设置清洗器"""
        pass

    def process(self, texts: list[str]) -> list[str]:
        """
        处理文本列表

        典型流程:
            1. 并行清洗所有文本
            2. 去重
            3. 串行通过各过滤器
            4. 返回处理后的文本

        步骤:
            Step 1: 文本清洗
                对输入列表中的每个文本应用清洗操作
                清洗操作：移除 HTML 标签、统一 Unicode、规范化空白字符等

            Step 2: 去重处理
                遍历清洗后的文本
                对每个文本检查是否已存在（通过哈希或相似度判断）
                保留不重复的文本，形成去重后的列表

            Step 3: 过滤器处理
                依次应用每个过滤器
                每个过滤器会根据自身规则排除不合格的文本
                保留通过所有过滤器的文本

            Step 4: 返回处理结果
        """
        pass


# ===================== 便捷函数 =====================

def load_and_process_data(
    data_path: str,
    filters: list = None,
    enable_dedup: bool = True,
    enable_cleaning: bool = True,
) -> list[str]:
    """
    加载并处理数据的便捷函数

    Args:
        data_path: 数据路径
        filters: 过滤器列表
        enable_dedup: 是否启用去重
        enable_cleaning: 是否启用清洗

    Returns:
        处理后的文本列表
    """
    pass
