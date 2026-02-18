# L04: KV Cache 与 CausalLM

## 学习目标

1. **理解** 因果语言模型 (Causal LM) 的工作原理
2. **掌握** KV Cache 优化技术
3. **掌握** 完整的 CausalLM 模型实现
4. **能够** 实现文本生成功能

---

## 理论背景

### 1. 因果语言模型

**定义**: 每个位置的预测只能依赖于之前的 token (自回归)。

**注意力掩码**:
```
位置 0: 只看自己
位置 1: 看到位置 0 和 1
位置 2: 看到位置 0, 1, 2
...
```

**训练目标**: 预测下一个 token
```
Input: "The cat sat"
Target: "he cat sat on"
```

### 2. KV Cache

**问题**: 自回归生成时，每次需要重新计算所有历史 token 的注意力。

**解决方案**: 缓存并复用之前计算出的 K 和 V。

```
第一次 (输入 "The"):
  K, V 存入缓存

第二次 (输入 "cat"):
  使用缓存的 K, V
  只计算新的 K, V
```

**显存优化**: GQA + KV Cache 可以大幅减少显存占用。

### 3. 采样策略

- **Greedy**: 总是选择概率最高的 token
- **Temperature**: 控制分布的平滑程度
- **Top-k**: 只考虑概率最高的 k 个 token
- **Top-p (Nucleus)**: 只考虑累积概率达到 p 的 token

---

## 实践练习

### 练习 1: 实现 CausalLM 类

打开 `model/causal_lm.py`，完成 `CausalLM` 类：

```python
class CausalLM(nn.Module):
    def __init__(self, config):
        """
        初始化因果语言模型。

        Args:
            config: ModelConfig 实例
        """
        pass

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        前向传播。

        需要实现:
        1. 词嵌入
        2. 因果掩码应用
        3. Transformer 层堆叠
        4. 最终归一化
        5. LM Head 预测
        """
        pass
```

### 练习 2: 实现文本生成方法

完成 `CausalLM.generate()` 方法：

```python
@torch.no_grad()
def generate(
    self,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    """
    自回归生成文本。

    实现步骤:
    1. 初始化 KV 缓存
    2. 循环生成新 token:
       a. 前向传播获取 logits (使用 KV 缓存)
       b. 温度缩放
       c. Top-k 过滤
       d. Top-p (Nucleus) 过滤
       e. 采样下一个 token
       f. 检查是否达到 EOS
    3. 返回完整序列
    """
    pass
```

### 练习 3: 理解采样策略

分析 `generate` 方法中的采样逻辑：
- **温度**: 控制概率分布的平滑程度
- **Top-k**: 限制候选 token 数量
- **Top-p**: 动态选择累积概率达到阈值的核心 token 集合

---

## 测试验证

```bash
pytest tutorial/stage01_foundation/lesson04_kv_cache_causallm/testcases/basic_test.py -v
pytest tutorial/stage01_foundation/lesson04_kv_cache_causallm/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **KV Cache 优化**: 了解更多推理优化技术
- **采样方法**: 了解不同的解码策略
