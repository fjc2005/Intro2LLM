# L02: Normalization Layers and Attention Mechanism

> **Course Positioning**: This is the **second lab** of the Intro2LLM course, which covers crucial components of the core architecture of LLM models. In lesson01, we learned how to convert text into vector representations; in this lab, we will learn how to stabilize training through **normalization layers** and how to capture the relationships between tokens through the **attention mechanism**.

## Lab Objectives

This lab primarily explains the principles and implementations of normalization layers (LayerNorm and RMSNorm) and the attention mechanism (Multi-Head Attention). Normalization layers are key components for stable training in deep learning models, while the attention mechanism is the core of the Transformer architecture, responsible for modeling the dependencies between different positions in a sequence.

### What You Will Learn in This Chapter

- **Normalization Layers**: Understand the differences between LayerNorm and RMSNorm, and grasp why modern LLMs prefer RMSNorm.
- **Attention Mechanism**: Master the mathematical principles of scaled dot-product attention, and understand the implementation of multi-head attention.
- **Engineering Implementation**: Learn to implement complete normalization and attention modules using Python.

---

## Part 1: Normalization Layers

### 1.1 Why is Normalization Needed?

In deep neural networks, **Internal Covariate Shift** is a classic problem: as the network grows deeper, the input distribution of each layer constantly shifts, leading to:
- Vanishing or exploding gradients
- Slower training convergence
- The need for very small learning rates

Normalization layers solve this problem by adjusting the activation values to a stable distribution.

### 1.2 LayerNorm (Layer Normalization)

#### 1.2.1 Algorithm Principles

LayerNorm was proposed by Ba et al. in 2016 and is the standard normalization method in the Transformer architecture.

**Calculation Formula**:
$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \odot \gamma + \beta
$$

Where:
- $\mathbb{E}[x]$: The mean over the last dimension
- $\text{Var}[x]$: The variance over the last dimension
- $\gamma$ (weight): Learnable scaling parameter
- $\beta$ (bias): Learnable shift parameter
- $\epsilon$: A small constant to prevent division by zero (usually 1e-6)

#### 1.2.2 LayerNorm vs BatchNorm



| Feature | BatchNorm | LayerNorm |
|------|-----------|-----------|
| Normalization Dimension | Batch dimension | Feature dimension |
| Batch Dependency | Yes (requires batch during training) | No |
| Applicable Scenarios | CV (larger image batches) | NLP (variable sequence lengths) |
| RNN Applicability | Not applicable | Applicable |

**Key Differences**:
- BatchNorm: Normalizes across the batch dimension; the same feature is normalized across different samples.
- LayerNorm: Normalizes across the feature dimension; different features are normalized within the same sample.

LayerNorm is more suitable for NLP tasks because:
1. Sequence lengths in NLP are variable, and batch sizes may also change.
2. It does not rely on batch statistics, making inference more stable.

#### 1.2.3 Pre-LN vs Post-LN Transformer



The original Transformer (Post-LN) uses:

```

x → SubLayer(x) → Add & Norm → ... → Output

```

The Pre-LN Transformer uses:

```

x → Norm → SubLayer → Add → ... → Output

```

**Advantages of Pre-LN**:
- Gradients are more stable, making it less prone to gradient explosion.
- Training is more robust, and learning rate tuning is simpler.
- Most modern LLMs now adopt the Pre-LN structure.

#### 1.2.4 LayerNorm Implementation Details

**File Location**: [model/norm.py](../../../model/norm.py)

**Code locations to complete**:
- `LayerNorm.__init__` method (lines 38-53)
- `LayerNorm.forward` method (lines 55-82)

**Implementation Steps**:

1. **In the initialization method**:
   - Call the parent class's initialization method.
   - Create a weight parameter with shape `[normalized_shape]`, initialized to all 1s.
   - Create a bias parameter with shape `[normalized_shape]`, initialized to all 0s.
   - Wrap them with `nn.Parameter` to make them learnable parameters.

2. **In the forward pass method**:
   - Step 1: Save the original data type, and cast the input to float32 to improve numerical stability.
   - Step 2: Calculate the mean along the last dimension, shape becomes `[..., 1]`.
   - Step 3: Calculate the variance along the last dimension (use unbiased=False).
   - Step 4: Standardize `(x - mean) / sqrt(variance + eps)`.
   - Step 5: Apply learnable parameters `output = normalized * weight + bias`.
   - Step 6: Restore the original data type.

---

### 1.3 RMSNorm (Root Mean Square Layer Normalization)

#### 1.3.1 Algorithm Principles

RMSNorm was proposed by Zhang and Sennrich in 2019 as a simplification of LayerNorm.

**Core Idea**: **Remove the centering operation (remove the mean)**, retaining only the Root Mean Square (RMS) scaling.

**Calculation Formula**:
$$
\text{RMS}(x) = \sqrt{\mathbb{E}[x^2] + \epsilon}
$$
$$
y = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

#### 1.3.2 Why is RMSNorm Faster?

1. **Calculates one less mean**:
   - LayerNorm: Needs to calculate mean + variance = 2 statistical operations.
   - RMSNorm: Only needs to calculate the mean square value = 1 statistical operation.

2. **One less bias parameter**:
   - LayerNorm: Has two parameters, weight + bias.
   - RMSNorm: Has only one parameter, weight.

3. **Simpler mathematical operations**:
   - Omits the operation of subtracting the mean.

#### 1.3.3 Why Do Modern LLMs Prefer RMSNorm?

1. **High computational efficiency**: Reduces normalization computation time by about 30%.

2. **Pre-LN structure does not need bias**:
   - In Post-LN, the bias after the residual connection is used to re-center the activation distribution.
   - In Pre-LN, each layer is normalized before the residual calculation, so bias is not needed to "re-center".
   - Removing the bias in RMSNorm is mathematically sound here.

3. **Comparable or better empirical performance**:
   - LLaMA, Qwen, Mistral, Gemma, etc., all adopt RMSNorm.

#### 1.3.4 RMSNorm Implementation Details

**File Location**: [model/norm.py](../../../model/norm.py)

**Code locations to complete**:
- `RMSNorm.__init__` method (lines 107-121)
- `RMSNorm.forward` method (lines 123-162)

**Implementation Steps**:

1. **In the initialization method**:
   - Call the parent class's initialization method.
   - Create a weight parameter with shape `[hidden_size]`, initialized to all 1s (multiplicative identity).
   - Wrap it with `nn.Parameter`.

2. **In the forward pass method**:
   - Step 1: Save the original data type, cast to float32.
   - Step 2: Calculate the mean square value `MS = mean(x^2)`, shape `[..., 1]`.
   - Step 3: Use `rsqrt` to calculate the reciprocal of the square root `inv_rms = 1 / sqrt(MS + eps)`.
   - Step 4: Normalize `x * inv_rms`, utilizing broadcasting.
   - Step 5: Apply learnable scaling `normalized * weight`.
   - Step 6: Restore the original data type.

---

## Part 2: Attention Mechanism

### 2.1 Intuitive Understanding of the Attention Mechanism

The core idea of the attention mechanism is: **When processing the current position, decide which preceding positions' information should be "paid attention to"**.

**Analogy**: When reading a passage of text, we do not memorize every single detail, but selectively focus on keywords. The attention mechanism simulates this process.



**Mathematical Expression**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$ (Query): What the current position "wants"
- $K$ (Key): What each position "offers"
- $V$ (Value): The "actual content" of each position
- $d_k$: The dimension of the Key

### 2.2 Scaled Dot-Product Attention

#### 2.2.1 Mathematical Derivation

**Advantages of dot-product attention**:
- High computational efficiency (matrix multiplication, GPU friendly)
- Can be calculated in parallel

**Why scale (divide by $\sqrt{d_k}$)?**

Assuming the elements of $Q$ and $K$ are independent random variables with a mean of 0 and a variance of 1, then:
- The mean of each element in $QK^T$ is 0
- The variance of each element in $QK^T$ is $d_k$

When $d_k$ is large, the variance will also be very large, causing the input to the softmax function to approach infinity, and the gradients to become very small (vanishing gradients).

**The solution is to divide by $\sqrt{d_k}$**:
- After scaling, the variance of $QK^T$ is restored to 1.
- The input distribution for the softmax function stays within a reasonable range.
- Gradients are more stable.

> **Proof**:
> Let $q_i, k_j \sim \mathcal{N}(0, 1)$, then $E[q_i k_j] = 0$, $Var(q_i k_j) = E[q_i^2]E[k_j^2] - (E[q_i]E[k_j])^2 = 1 \cdot 1 - 0 = 1$ (assuming independence).
> Therefore, $Var(\sum_i q_i k_i) = d_k$.

#### 2.2.2 Masking Mechanism

**Causal Mask**:
- In autoregressive generation, the current position can only see previous positions.
- Implementation method: Set the upper triangle of $QK^T$ to $-\infty$.

### 2.3 Multi-Head Attention (MHA)



#### 2.3.1 Algorithm Principles

A single attention head can only capture one type of dependency. Multi-head attention uses **multiple independent attention heads** to allow the model to jointly attend to information from different representation subspaces.

**Core Formula**:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$
$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Parameter Relationships**:
- `num_heads * head_dim = hidden_size`
- Dimension of each head `head_dim = hidden_size // num_heads`

#### 2.3.2 Characteristics

**Advantages**:
- Each head can learn different types of dependencies (e.g., syntax, semantics, position).
- Parallel computation, high efficiency.

### 2.4 MultiHeadAttention Implementation Details

**File Location**: [model/attention.py](../../../model/attention.py)

**Code locations to complete**:
- `MultiHeadAttention.__init__` method (lines 37-60)
- `MultiHeadAttention.forward` method (lines 62-144)

**Implementation Steps**:

1. **In the initialization method**:
   - Extract hidden_size, num_attention_heads, attention_dropout from the config.
   - Verify that hidden_size is divisible by num_heads.
   - Create four linear projection layers: q_proj, k_proj, v_proj, o_proj.
   - Projection dimensions are all `[hidden_size, hidden_size]`.

2. **In the forward pass method**:
   - **Step 1**: Linear projection to obtain Q, K, V.
   - **Step 2**: Reshape into multi-head format `[batch, num_heads, seq, head_dim]`.
   - **Step 3**: Apply RoPE (Rotary Position Embedding).
   - **Step 4**: Calculate scaled dot-product attention scores `scores = (Q · K^T) / sqrt(d_k)`.
   - **Step 5**: Apply attention mask.
   - **Step 6**: Apply softmax to obtain attention weights.
   - **Step 7**: Calculate weighted output `output = weights · V`.
   - **Step 8**: Reshape back to the original dimensions.
   - **Step 9**: Output projection.

---

## Code Completion Locations Summary

### File 1: [model/norm.py](../../../model/norm.py)

| Class | Method | Lines | Function |
|------|------|------|------|
| `LayerNorm` | `__init__` | 38-53 | Create weight and bias parameters |
| `LayerNorm` | `forward` | 55-82 | Implement layer normalization |
| `RMSNorm` | `__init__` | 107-121 | Create weight parameter |
| `RMSNorm` | `forward` | 123-162 | Implement RMS normalization |

### File 2: [model/attention.py](../../../model/attention.py)

| Class | Method | Lines | Function |
|------|------|------|------|
| `MultiHeadAttention` | `__init__` | 37-60 | Initialize Q/K/V/O projection layers |
| `MultiHeadAttention` | `forward` | 62-144 | Implement multi-head attention |

---

## Exercises

### Lab Report Requirements

- Complete it in Markdown format, primarily text-based.
- Fill in the required report content for each basic exercise.
- List the key knowledge points you consider important in this lab, correspond them to the relevant LLM principles, and briefly explain your understanding of their meanings, relationships, and differences.

### Exercise 1: Understand LayerNorm vs RMSNorm

Read `model/norm.py`, think about and answer:

1. What is the core difference between LayerNorm and RMSNorm? Why is RMSNorm faster?
2. Why doesn't the Pre-LN Transformer need the bias parameter from LayerNorm?
3. If the weight of RMSNorm is initialized to all 0s, what would happen? Why?

### Exercise 2: Understand the Attention Mechanism

Read `model/attention.py`, think about and answer:

1. What is the role of the scaling factor $\sqrt{d_k}$? What problem would occur if we didn't divide by this value?
2. What is the purpose of the causal mask? How is it implemented?
3. Why does MultiHeadAttention need to apply linear projections to Q, K, and V separately, instead of directly using the inputs?

### Exercise 3: Verify Your Implementation

Run the following test code to verify if your implementation is correct:

```python
# Test LayerNorm
import torch
from model.norm import LayerNorm

norm = LayerNorm(normalized_shape=128)
x = torch.randn(4, 16, 128)
output = norm(x)

# Verify shape
assert output.shape == x.shape, f"Shape should be {x.shape}, but got {output.shape}"

# Verify normalization effect
mean = output.mean(dim=-1)
std = output.std(dim=-1)
assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), "Mean should be close to 0"
print("LayerNorm test passed!")

# Test RMSNorm
from model.norm import RMSNorm

rms_norm = RMSNorm(hidden_size=128)
x = torch.randn(4, 16, 128)
output = rms_norm(x)

assert output.shape == x.shape
print("RMSNorm test passed!")

# Test MultiHeadAttention
from model.attention import MultiHeadAttention
from model.config import ModelConfig

config = ModelConfig(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=512,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    hidden_act="silu",
    use_rms_norm=True,
    use_rope=True,
    use_swiglu=True,
)

mha = MultiHeadAttention(config)
x = torch.randn(2, 16, config.hidden_size)
output, _ = mha(x)

assert output.shape == x.shape
print("MultiHeadAttention test passed!")

print("\nAll tests passed!")

```

---

## Frequently Asked Questions (FAQ)

**Q1: How big is the difference between LayerNorm and RMSNorm in practical applications?**
A: In most cases, their performance is very close. The main advantages of RMSNorm are faster computation speed (about 30%) and fewer parameters. Due to the popularity of the Pre-LN Transformer, RMSNorm has become the default choice for modern LLMs.

**Q2: Why divide by $\sqrt{d_k}$ instead of $d_k$?**
A: If divided by $d_k$, the attention score values would become too small, causing the softmax to approach a uniform distribution, and the model wouldn't be able to learn significant attention patterns. Dividing by $\sqrt{d_k}$ restores the variance to 1, making the input distribution for softmax more reasonable.

**Q3: Why does each head need independent Q, K, V projections?**
A: Independent projections allow each head to learn different attention patterns and capture different types of information (such as syntax, semantics, and positional relationships). If the projections were shared, all heads would learn the exact same attention patterns.

**Q4: Where is the causal mask applied?**
A: The causal mask is applied to the $QK^T$ matrix, setting the scores of positions after the current position to $-\infty$, so that after softmax, the attention weights for these positions approach 0.

**Q5: Must the head_dim in multi-head attention be an integer?**
A: Yes, hidden_size must be divisible by num_attention_heads; otherwise, it cannot be evenly divided into multiple heads.

---

## Further Reading

### Original Papers

1. **LayerNorm**: "Layer Normalization" - Ba et al., 2016
* https://arxiv.org/abs/1607.06450


2. **RMSNorm**: "Root Mean Square Layer Normalization" - Zhang & Sennrich, 2019
* https://arxiv.org/abs/1910.07467


3. **Attention**: "Attention Is All You Need" - Vaswani et al., 2017
* https://arxiv.org/abs/1706.03762



### Practical References

* **Flash Attention**: Understand GPU-optimized attention implementations (https://github.com/Dao-AILab/flash-attention)

### Extended Reading

* **Pre-LN Transformer**: "On Layer Normalization in the Transformer Architecture" - Xiong et al., 2020
* Understand the differences between Pre-LN vs Post-LN and their impact on training stability