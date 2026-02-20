# L03: FeedForward and Transformer Block

> **Course Positioning**: This is the **third lab** of the Intro2LLM course, covering a crucial component of the core LLM architecture. In lesson01, we learned how to convert text into vector representations (Tokenizer and Embedding); in lesson02, we learned about normalization layers and the attention mechanism; in this lab, we will learn the principles of the **FeedForward Network (FFN)** and how to combine these components into a complete **Transformer Block**.

## Lab Objectives

This lab mainly explains the principles and implementation of FeedForward Networks (Basic FFN, GeGLU, and SwiGLU), as well as the structural design of the Transformer Block (Pre-LN vs Post-LN). The FeedForward Network is the key component in the Transformer architecture responsible for performing independent non-linear transformations at each position, while the Transformer Block is the fundamental unit that makes up the entire LLM.

### What You Will Learn in This Chapter

- **Basic FFN**: Understand the structure of the FFN in the original Transformer.
- **Gated Linear Unit (GLU)**: Understand the differences and advantages of GeGLU and SwiGLU, and master the principles of gating mechanisms.
- **Pre-LN vs Post-LN**: Understand the differences between the two Transformer Block structures and grasp why modern LLMs prefer Pre-LN.
- **Residual Connection**: Understand the role of residual connections in training deep networks.
- **Engineering Implementation**: Learn to implement complete FeedForward Networks and Transformer Blocks using Python.

---

## Part 1: FeedForward Network (FFN)



### 1.1 Why Do We Need FeedForward Networks?

In the Transformer architecture, the Self-Attention mechanism is responsible for capturing dependencies between different positions in a sequence. However, relying solely on the attention mechanism only allows the model to perform weighted combinations of existing information, without the ability to perform more complex non-linear transformations.

The roles of the FeedForward Network (FFN) include:
1. **Increasing non-linear expression capabilities**: By using non-linear activation functions, the model can learn more complex patterns.
2. **Feature transformation**: Applying an independent non-linear transformation to the representation at each position.
3. **Increasing model capacity**: The FFN typically accounts for a large portion of the model's parameters.

---

### 1.2 Basic FFN (Original Transformer)

#### 1.2.1 Algorithm Principles

The FeedForward Network structure used in the original Transformer paper is very simple:

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

**Structure**:
- First linear transformation layer: hidden_size → intermediate_size
- ReLU activation function
- Second linear transformation layer: intermediate_size → hidden_size

**ReLU Activation Function**:
$$
\text{ReLU}(x) = \max(0, x)
$$

**Characteristics**:
- Simple computation and high efficiency.
- Was once the standard configuration for Transformers.
- Relatively limited expressiveness.

#### 1.2.2 Basic FFN Implementation Essentials

**File Location**: [model/feedforward.py](../../../model/feedforward.py)

**Code locations to complete**:
- `BasicFFN.__init__` method
- `BasicFFN.forward` method

**Implementation Steps**:

1. **In the initialization method**:
   - Call the parent class's initialization method.
   - Extract hidden_size and intermediate_size from the configuration.
   - Create two linear projection layers:
     - `w1`: hidden_size → intermediate_size (with bias)
     - `w2`: intermediate_size → hidden_size (with bias)

2. **In the forward pass method**:
   - Step 1: Map hidden_size to intermediate_size via w1.
   - Step 2: Apply the ReLU activation function.
   - Step 3: Map intermediate_size back to hidden_size via w2.

---

### 1.3 Gated Linear Unit (GLU)

#### 1.3.1 Why Do We Need GLU?

The original Basic FFN uses ReLU activation, which is simple and effective but has limited expressive power. In 2020, Shazeer proposed the concept of the **Gated Linear Unit (GLU)** in the paper "GLU Variants Improve Transformer," introducing a gating mechanism to control information flow, thereby enhancing the model's expressive capabilities.

**Core Idea**:
- Use a "gating" signal to control which information can pass through.
- Similar to the gating mechanism in LSTMs, but simpler and more efficient.

#### 1.3.2 Mathematical Formula for GLU

**Standard GLU** computation formula:

$$
\text{GLU}(x) = (xW + b) \odot \sigma(xV + c)
$$

Where:
- $W, V$: Projection matrices
- $b, c$: Bias vectors
- $\sigma$: Sigmoid activation function
- $\odot$: Element-wise multiplication (Hadamard product)

**Key Point**: The value of the gating signal $\sigma(xV + c)$ falls between 0 and 1, acting as a "switch" for the input information.

---

### 1.4 GeGLU (Gated Linear Unit with GELU)

#### 1.4.1 Algorithm Principles

GeGLU is a variant of GLU that uses **GELU** as the gating activation function.

**Computation Formula**:

$$
\text{GeGLU}(x) = \text{GELU}(xW_{\text{gate}}) \odot (xW_{\text{up}})
$$

Where:
- $W_{\text{gate}}$: Gating projection matrix
- $W_{\text{up}}$: Upsampling projection matrix
- GELU activation function: $\text{GELU}(x) = x \cdot \Phi(x)$, where $\Phi$ is the CDF of the standard normal distribution

**GELU Activation Function**:

$$
\text{GELU}(x) = x \cdot P(X \le x) = x \cdot \Phi(x)
$$

Approximate computation:
$$
\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))
$$

**Characteristics**:
- GELU is a smooth activation function, more non-linear than ReLU.
- GELU can be understood as a "soft" gate.
- Used by PaLM.

#### 1.4.2 GeGLU Implementation Essentials

**File Location**: [model/feedforward.py](../../../model/feedforward.py)

**Code locations to complete**:
- `FeedForward.__init__` method (lines 48-71)
- `FeedForward.forward` method (lines 73-113)

**Implementation Steps**:

1. **In the initialization method**:
   - Call the parent class's initialization method.
   - Extract hidden_size and intermediate_size from the configuration.
   - Create three linear projection layers:
     - `gate_proj`: hidden_size → intermediate_size
     - `up_proj`: hidden_size → intermediate_size
     - `down_proj`: intermediate_size → hidden_size

2. **In the forward pass method**:
   - Step 1: Gating projection - Map hidden_size to intermediate_size via gate_proj.
   - Step 2: Apply the GELU activation function to the gating projection result.
   - Step 3: Upsampling projection - Get another set of intermediate representations via up_proj.
   - Step 4: Element-wise gated multiplication - Activated gating values × upsampled values.
   - Step 5: Downsampling projection - Map intermediate_size back to hidden_size via down_proj.

---

### 1.5 SwiGLU (Swish-Gated Linear Unit)



#### 1.5.1 Algorithm Principles

SwiGLU is another variant of GLU that uses **SiLU (Sigmoid Linear Unit)** as the gating activation function, also known as Swish.

**Computation Formula**:

$$
\text{SwiGLU}(x) = \text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})
$$

Where the SiLU activation function is:

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**Characteristics of SiLU**:
- **Self-gating**: The input is multiplied by its own sigmoid value.
- **Smooth and non-monotonic**: Unlike ReLU, SiLU has non-zero outputs in the negative region.
- **Negative region**: Allows the model to learn to "suppress" certain features.

**Why modern LLMs prefer SwiGLU**:

1. **Smoother gradients**: SiLU has smoother gradients near zero.
2. **Better training stability**: Performs better in deep networks.
3. **Mainstream adoption**: Used by LLaMA, Qwen, Mistral, etc.
4. **Comparable performance**: Slightly more popular compared to GeGLU.

#### 1.5.2 SwiGLU Implementation Essentials

**File Location**: [model/feedforward.py](../../../model/feedforward.py)

**Code locations to complete**:
- `SwiGLU.__init__` method (lines 146-168)
- `SwiGLU.forward` method (lines 170-208)

**Implementation Steps**:

1. **In the initialization method**:
   - Call the parent class's initialization method.
   - Extract hidden_size and intermediate_size from the configuration.
   - Create three linear projection layers (same structure as GeGLU).

2. **In the forward pass method**:
   - Step 1: Gating projection.
   - Step 2: Apply SiLU activation function (use `nn.SiLU()` or `torch.nn.functional.silu`).
   - Step 3: Upsampling projection.
   - Step 4: Element-wise gated multiplication.
   - Step 5: Downsampling projection.

---

### 1.6 FFN Parameter Quantity Analysis

**Comparison of parameter quantities across FFN types**:

Assuming hidden_size = $d$, intermediate_size = $4d$ (usually set to 2-4 times the hidden_size)

| FFN Type | Parameter Calculation | Total Parameters |
|----------|-----------------------|------------------|
| Basic FFN | $w1: d \times 4d$ + bias $4d$ + $w2: 4d \times d$ + bias $d$ | $8d^2 + 5d$ ≈ $8d^2$ |
| GeGLU/SwiGLU | gate: $d \times 4d$ + up: $d \times 4d$ + down: $4d \times d$ | $12d^2$ |

**Analysis**:
- GLU variants do not have bias terms (learnable biases are not needed in Pre-LN structures).
- GLU variants have 12d² parameters, which is 50% more than the original FFN's 8d².
- More parameters bring better expressive capabilities, which is a trade-off modern LLMs are willing to make.

---

## Part 2: Transformer Block Structure



### 2.1 Pre-LN vs Post-LN

There are two main ways to place normalization in a Transformer Block: **Pre-LN** and **Post-LN**. This choice has a significant impact on training stability.

#### 2.1.1 The Two Structures

**Post-LN (Original Transformer)**:


```

Input → Attention → Add → Norm → FFN → Add → Norm → Output

```

Also known as:

```

x → SubLayer(x) → Add → Norm → Output

```

**Pre-LN (Modern Standard)**:


```

Input → Norm → Attention → Add → Norm → FFN → Add → Output

```

Also known as:

```

x → Norm → SubLayer → Add → Output

```

#### 2.1.2 Core Differences Comparison

| Feature | Post-LN | Pre-LN |
|---------|---------|--------|
| Normalization Location | After sub-layer output (after Add) | Before sub-layer input |
| Residual Connection | Before normalization | After normalization |
| Gradient Flow | Unstable in deep layers, large gradients at output layer | More stable gradients, uniform across layers |
| Learning Rate | Requires warm-up | Can use large learning rates |
| Training Stability | Deep networks prone to divergence | More robust training |
| Representative Models | BERT, GPT-2, Original Transformer | LLaMA, Qwen, Mistral |

#### 2.1.3 Problems with Post-LN

**Why is Post-LN training unstable in deep networks?**

1. **Vanishing/Exploding Gradients**:
   - In Post-LN, gradients from the final layer need to pass through multiple normalization layers and residual connections.
   - The gradient path in deep networks is longer, making them more prone to vanishing or exploding gradients.

2. **Reliance on Learning Rate Warm-up**:
   - Post-LN Transformers require learning rate warm-up to stabilize training.
   - Without warm-up, the model can easily diverge.

3. **Large Gradients at the Output Layer**:
   - Layers near the output tend to have large gradients, easily leading to training instability.

#### 2.1.4 Advantages of Pre-LN

**Why do modern LLMs all choose Pre-LN?**

1. **More Stable Gradients**:
   - The input to each layer is normalized, making the scale of activation values more stable.
   - Gradients are distributed more evenly across layers.

2. **No Need for Warm-up**:
   - Pre-LN can use a constant learning rate.
   - Warm-up can still be used, but training remains stable even without it.

3. **Can Use Larger Learning Rates**:
   - Faster training convergence.
   - Hyperparameters are easier to set.

4. **Theoretical Support**:
   - The paper "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) proved that Pre-LN gradients are more bounded.

> **Important Conclusion**: Pre-LN is the standard choice for modern LLMs; all mainstream models (LLaMA, Qwen, Mistral, etc.) adopt the Pre-LN structure.

---

### 2.2 Residual Connection



#### 2.2.1 Principles

The core idea of a residual connection is to let the network learn the **identity mapping**:

$$
y = F(x) + x
$$

Where $F(x)$ is the learned residual.

**Roles**:
1. **Mitigate Vanishing Gradients**: Gradients can flow directly back to the input.
2. **Stabilize Training**: Makes it easier for the network to learn the identity mapping.
3. **Information Transfer**: Low-level information can be passed directly to higher layers.

#### 2.2.2 Application in Transformers

There are two residual connections in a Transformer Block:

1. **Attention Residual**:
   $$
   x_{\text{after\_attn}} = \text{Attention}(x) + x
   $$

2. **FFN Residual**:
   $$
   x_{\text{output}} = \text{FFN}(x_{\text{after\_attn}}) + x_{\text{after\_attn}}
   $$

---

### 2.3 TransformerBlock Implementation Essentials

**File Location**: [model/transformer_block.py](../../../model/transformer_block.py)

**Code locations to complete**:
- `TransformerBlock.__init__` method (lines 61-86)
- `TransformerBlock.forward` method (lines 88-155)

**Implementation Steps**:

1. **In the initialization method**:
   - Call the parent class's initialization method.
   - Save layer_idx to identify the current layer.
   - Choose LayerNorm or RMSNorm based on config.use_rms_norm.
   - Create Pre-Attention normalization layer: `input_layernorm`.
   - Create Self-Attention module: `self_attn` (choose MHA or GQA based on config).
   - Create Pre-FFN normalization layer: `post_attention_layernorm`.
   - Create FeedForward Network module: `mlp` (choose SwiGLU or GeGLU based on config.use_swiglu).

2. **In the forward pass method**:

   **Self-Attention Sub-layer in Pre-LN Structure**:
   - Step 1: Save residual (original input).
   - Step 2: Pre-Attention normalization (before attention).
   - Step 3: Call the self-attention module to get the output and KV cache.
   - Step 4: Residual connection (attention output + original input).

   **FeedForward Network Sub-layer in Pre-LN Structure**:
   - Step 5: Save residual (attention output).
   - Step 6: Pre-FFN normalization (before FFN).
   - Step 7: Call the FeedForward Network module.
   - Step 8: Residual connection (FFN output + attention output).

   - Step 9: Return results (hidden_states, present_key_value).

---

## Code Completion Location Summary

### File 1: [model/feedforward.py](../../../model/feedforward.py)

| Class | Method | Line Numbers | Function |
|-------|--------|--------------|----------|
| `BasicFFN` | `__init__` | - | Initialize Basic FFN |
| `BasicFFN` | `forward` | - | Implement Basic FFN forward pass |
| `FeedForward` (GeGLU) | `__init__` | 48-71 | Initialize gating projection layers |
| `FeedForward` (GeGLU) | `forward` | 73-113 | Implement GeGLU forward pass |
| `SwiGLU` | `__init__` | 146-168 | Initialize gating projection layers |
| `SwiGLU` | `forward` | 170-208 | Implement SwiGLU forward pass |
| `get_feed_forward` | - | 212-226 | Choose FFN type based on configuration |

### File 2: [model/transformer_block.py](../../../model/transformer_block.py)

| Class | Method | Line Numbers | Function |
|-------|--------|--------------|----------|
| `TransformerBlock` | `__init__` | 61-86 | Initialize sub-modules |
| `TransformerBlock` | `forward` | 88-155 | Implement Pre-LN Transformer Block |

---

## Exercises

### Requirements for the Lab Report

- Complete it using Markdown format, primarily text-based.
- Fill in the required report contents for each basic exercise.
- List the knowledge points you consider important in this lab, align them with the corresponding LLM principles, and briefly explain your understanding of their meanings, relationships, and differences.

### Exercise 1: Understanding Basic FFN vs GLU Variants

Read `model/feedforward.py`, and combine it with the algorithm principles to answer the following questions:

1. What is the core difference between Basic FFN and GLU variants (GeGLU/SwiGLU)? Why can GLU variants enhance the model's expressive capabilities?
2. How does the "gating" mechanism in GLU work? Why can gating signals control information flow?
3. What would happen if the SiLU activation in SwiGLU were replaced with ReLU? Please analyze this from both a mathematical formula perspective and practical effects.

### Exercise 2: Understanding Pre-LN vs Post-LN

Think about and answer:

1. What problems does the Post-LN Transformer encounter in deep network training? Why are these problems more severe in deep networks?
2. How does the Pre-LN structure solve these problems? Please explain from the perspectives of gradient flow and normalization.
3. Why do modern LLMs (like LLaMA, Qwen) all adopt the Pre-LN structure? What impact does this have on model training and inference?

### Exercise 3: Understanding Residual Connections

Think about and answer:

1. What is the core idea of a residual connection? Why can it mitigate the vanishing gradient problem?
2. In the Pre-LN Transformer Block, where are residual connections applied? What would be the impact on model training if residual connections were removed?
3. The output of a residual connection is $y = F(x) + x$. If the output dimension of $F(x)$ is inconsistent with $x$, how should this be handled?

### Exercise 4: Verifying Your Implementation

Run the following test code to verify if your implementation is correct:

```python
# Test Basic FFN
import torch
from model.feedforward import BasicFFN
from model.config import ModelConfig

config = ModelConfig(
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
)

basic_ffn = BasicFFN(config)
x = torch.randn(2, 10, config.hidden_size)
output = basic_ffn(x)

assert output.shape == x.shape, f"Shape should be {x.shape}, but got {output.shape}"
print("BasicFFN test passed!")

# Test SwiGLU
from model.feedforward import SwiGLU

swiglu = SwiGLU(config)
output = swiglu(x)

assert output.shape == x.shape
print("SwiGLU test passed!")

# Test GeGLU
from model.feedforward import FeedForward

geglu = FeedForward(config)
output = geglu(x)

assert output.shape == x.shape
print("GeGLU test passed!")

# Test TransformerBlock
from model.transformer_block import TransformerBlock

full_config = ModelConfig(
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

block = TransformerBlock(full_config)
output, present_kv = block(x)

assert output.shape == x.shape, f"Shape should be {x.shape}, but got {output.shape}"
print("TransformerBlock test passed!")

print("\nAll tests passed!")

```

---

## Frequently Asked Questions (FAQ)

**Q1: Why do we need two projection layers (gate and up) instead of one?**
A: The core idea of GLU is to control information flow through a gating mechanism. The gating signal generated by `gate_proj` (activated via sigmoid/silu) determines how much of the values generated by `up_proj` can "pass through." This design is more flexible than a single projection and allows the model to adaptively choose which dimensions should be activated.

**Q2: What advantages does SiLU in SwiGLU have compared to ReLU?**
A: SiLU is a smooth, non-monotonic activation function that has non-zero outputs in the negative region. This allows the model to learn to "suppress" certain features. ReLU is consistently 0 in the negative region and cannot learn negative representations. Experiments show that SiLU performs better in Transformers.

**Q3: How does placing normalization before or after the sub-layer in Pre-LN affect training?**
A: Pre-LN (normalization before the sub-layer) ensures that the input to each layer is normalized, resulting in more stable gradients. Post-LN (normalization after the sub-layer) can cause gradient explosion in deep networks and requires learning rate warm-up. Pre-LN is the standard choice for modern LLMs.

**Q4: How many times larger is the `intermediate_size` of the FeedForward Network compared to the `hidden_size`?**
A: It is usually set to 2-4 times larger. The original Transformer used 4 times (LLaMA uses 8/3 ≈ 2.67 times, Qwen uses 2.75 times). A larger `intermediate_size` can increase model capacity but also increases computation and memory overhead.

**Q5: Can LayerNorm and RMSNorm be mixed in a Transformer Block?**
A: Modern LLMs typically use RMSNorm uniformly (because it's faster). However, technically speaking, the two normalizations can be mixed (some early models did this). The current standard practice is to choose uniformly based on `config.use_rms_norm`.

---

## Further Reading

### Original Papers

1. **GLU Variants Improve Transformer** - Shazeer, 2020
* https://arxiv.org/abs/2002.05202
* First proposed the application of GLU variants in Transformers.


2. **Swish: A Self-Gated Activation Function** - Ramachandran et al., 2017
* https://arxiv.org/abs/1710.05941
* First proposed the Swish/SiLU activation function.


3. **On Layer Normalization in the Transformer Architecture** - Xiong et al., 2020
* https://arxiv.org/abs/2002.04745
* Theoretical analysis of Pre-LN Transformers.


4. **PaLM: Scaling Language Modeling with Pathways** - Chowdhery et al., 2022
* https://arxiv.org/abs/2204.02311
* PaLM uses GeGLU.



### Practical References

* **LLaMA**: https://github.com/meta-llama/llama - Uses SwiGLU + Pre-LN
* **Qwen**: https://github.com/QwenLM/Qwen - Uses SwiGLU + Pre-LN
* **Mistral**: https://github.com/mistralai/mistral-src - Uses SwiGLU + Pre-LN