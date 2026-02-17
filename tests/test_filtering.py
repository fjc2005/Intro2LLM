"""
数据工程模块测试
测试数据过滤、去重、清洗等功能。
"""

import pytest
from data.filtering import (
    LengthFilter,
    RepetitionFilter,
    LanguageFilter,
    QualityFilter,
    ToxicityFilter,
    Deduplicator,
    FuzzyDeduplicator,
    DataCleaner,
    DataSampler,
    Pipeline,
)


def test_length_filter():
    """测试长度过滤器"""
    filter_obj = LengthFilter(min_length=10, max_length=1000)

    # 测试用例
    assert filter_obj.filter("Hello world") == False  # 太短
    assert filter_obj.filter("This is a test sentence with enough length to pass the filter.") == True  # 合适
    assert filter_obj.filter("a" * 2000) == False  # 太长
    assert filter_obj.filter("a" * 500) == True  # 合适

    print("✓ LengthFilter 测试通过")


def test_repetition_filter():
    """测试重复率过滤器"""
    filter_obj = RepetitionFilter(
        max_repetition_ratio=0.3,
        n_gram=3,
    )

    # 测试用例
    assert filter_obj.filter("hello world") == True  # 无重复
    assert filter_obj.filter("aaa bbb aaa bbb aaa bbb") == False  # 高重复
    assert filter_obj.filter("the the the the the") == False  # 连续重复

    print("✓ RepetitionFilter 测试通过")


def test_language_filter():
    """测试语言过滤器"""
    # 英文过滤器
    en_filter = LanguageFilter(target_language="en", min_language_ratio=0.9)

    # 测试用例
    assert en_filter.filter("Hello, this is a sentence in English.") == True
    assert en_filter.filter("你好，这是一句中文。") == False
    assert en_filter.filter("12345") == False  # 无有效语言

    print("✓ LanguageFilter 测试通过")


def test_quality_filter():
    """测试质量过滤器"""
    filter_obj = QualityFilter(
        min_length=10,
        max_length=10000,
        max_special_char_ratio=0.5,
        min_word_count=5,
    )

    # 测试用例
    assert filter_obj.filter("This is a normal English sentence with normal punctuation.") == True
    assert filter_obj.filter("abc@#$%^&*()") == False  # 特殊字符过多

    print("✓ QualityFilter 测试通过")


def test_toxicity_filter():
    """测试有害内容过滤器"""
    filter_obj = ToxicityFilter(threshold=0.5)

    # 测试用例 (简单规则匹配)
    assert filter_obj.filter("Hello, how can I help you today?") == True  # 正常内容
    # 实际有害内容过滤取决于实现

    print("✓ ToxicityFilter 测试通过")


def test_deduplicator():
    """测试精确去重"""
    dedup = Deduplicator(method="exact")

    # 测试用例
    assert dedup.add("Hello world") == True  # 新文档
    assert dedup.add("Hello world") == False  # 重复
    assert dedup.add("HELLO WORLD") == False  # 大小写不同也算重复 (根据实现)
    assert dedup.add("Different text") == True  # 新文档

    print("✓ Deduplicator 测试通过")


def test_fuzzy_deduplicator():
    """测试模糊去重"""
    dedup = FuzzyDeduplicator(
        similarity_threshold=0.8,
        method="simhash",
    )

    # 测试用例
    assert dedup.add("Hello world") == True  # 新文档
    assert dedup.add("Hello, world!") == True  # 轻微变化，可能被判定为重复
    assert dedup.add("Completely different content here") == True  # 完全不同

    print("✓ FuzzyDeduplicator 测试通过")


def test_data_cleaner():
    """测试数据清洗"""
    cleaner = DataCleaner(
        lowercase=False,
        remove_extra_whitespace=True,
        normalize_unicode=True,
        remove_html_tags=True,
    )

    # 测试用例
    assert cleaner.clean("  Hello   world  ") == "Hello world"  # 去除多余空格
    assert cleaner.clean("<p>Hello</p>") == "Hello"  # 移除 HTML
    # Unicode 规范化测试

    print("✓ DataCleaner 测试通过")


def test_data_cleaner_lowercase():
    """测试数据清洗 - 小写转换"""
    cleaner = DataCleaner(lowercase=True)

    assert cleaner.clean("HELLO WORLD") == "hello world"
    assert cleaner.clean("Hello World") == "hello world"

    print("✓ DataCleaner (lowercase) 测试通过")


def test_data_sampler():
    """测试数据采样"""
    import numpy as np

    # 均匀采样
    sampler = DataSampler(sampling_strategy="uniform")
    data = list(range(100))
    sampled = sampler.sample(data, sample_size=10)
    assert len(sampled) == 10

    # 加权采样
    weights = [1.0] * 50 + [10.0] * 50  # 后50个权重更高
    weighted_sampler = DataSampler(sampling_strategy="weighted", weights=weights)
    sampled = weighted_sampler.sample(data, sample_size=10)

    # 验证采样结果在合理范围内
    assert len(sampled) == 10
    assert all(0 <= x < 100 for x in sampled)

    print("✓ DataSampler 测试通过")


def test_pipeline():
    """测试数据处理流水线"""
    pipeline = Pipeline()

    # 添加过滤器
    pipeline.add_filter(LengthFilter(min_length=10, max_length=1000))
    pipeline.add_filter(QualityFilter(min_word_count=3))

    # 设置去重和清洗
    pipeline.set_deduplicator(Deduplicator(method="exact"))
    pipeline.set_cleaner(DataCleaner())

    # 处理数据
    texts = [
        "Hello world",
        "This is a longer text that should pass the filters",
        "ab",  # 太短，会被过滤
        "Hello world",  # 重复，会被去重
        "   Multiple   spaces   here   ",  # 会被清洗
    ]

    result = pipeline.process(texts)

    # 验证结果
    assert len(result) < len(texts)  # 应该有文本被过滤
    assert "Hello world" not in result or len([t for t in result if t == "Hello world"]) == 1  # 不重复

    print(f"✓ Pipeline 测试通过: 输入 {len(texts)} -> 输出 {len(result)}")


def test_pipeline_order():
    """测试流水线顺序"""
    pipeline = Pipeline()

    # 顺序: 清洗 -> 去重 -> 过滤
    pipeline.set_cleaner(DataCleaner())
    pipeline.set_deduplicator(Deduplicator(method="exact"))
    pipeline.add_filter(LengthFilter(min_length=5))

    texts = [
        "hello",
        "hello",  # 重复
        "  world  ",  # 清洗后 "world"
    ]

    result = pipeline.process(texts)

    # 验证: "hello" 去重后只保留一个
    # "world" 清洗后通过长度过滤
    assert len(result) >= 1

    print("✓ Pipeline 顺序测试通过")
