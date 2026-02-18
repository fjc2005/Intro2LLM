"""
课时6基础测试：数据集基础功能验证
"""

import pytest
import torch
from torch.utils.data import DataLoader


class TestPretrainDataset:
    """测试预训练数据集"""

    def test_dataset_initialization(self):
        """测试数据集正确初始化"""
        # TODO: 创建PretrainDataset
        # 验证能正确加载文本数据
        pass

    def test_dataset_length(self):
        """测试数据集长度"""
        # TODO: 验证__len__返回正确
        pass

    def test_getitem_output_format(self):
        """测试__getitem__输出格式"""
        # TODO: 验证返回包含input_ids, attention_mask, labels
        pass

    def test_labels_equal_input_ids(self):
        """测试labels与input_ids相同"""
        # TODO: 因果LM中labels == input_ids
        pass

    def test_packing_behavior(self):
        """测试packing行为"""
        # TODO: 验证packing=True时多个文档合并
        pass


class TestSFTDataset:
    """测试SFT数据集"""

    def test_sft_initialization(self):
        """测试SFT数据集初始化"""
        # TODO: 加载instruction格式数据
        pass

    def test_prompt_masking(self):
        """测试prompt masking"""
        # TODO: 验证prompt部分labels为-100
        pass

    def test_response_labels(self):
        """测试response labels"""
        # TODO: 验证response部分labels正确
        pass

    def test_chat_template_application(self):
        """测试对话模板应用"""
        # TODO: 验证chat_template正确格式化
        pass


class TestDPODataset:
    """测试DPO数据集"""

    def test_dpo_initialization(self):
        """测试DPO数据集初始化"""
        # TODO: 加载preference数据
        pass

    def test_chosen_rejected_format(self):
        """测试chosen/rejected格式"""
        # TODO: 验证同时返回chosen和rejected
        pass

    def test_prompt_length_tracking(self):
        """测试prompt长度记录"""
        # TODO: 验证返回prompt_length用于masking
        pass


class TestDataFiltering:
    """测试数据过滤"""

    def test_length_filter(self):
        """测试长度过滤"""
        # TODO: 验证过短/过长文档被过滤
        pass

    def test_exact_deduplication(self):
        """测试精确去重"""
        # TODO: 验证重复文档被移除
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
