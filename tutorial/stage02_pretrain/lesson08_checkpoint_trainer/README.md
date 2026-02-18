# L08: 检查点、训练器与技巧

## 学习目标

1. **掌握** 检查点保存与加载
2. **掌握** 训练器实现 (梯度累积、梯度裁剪)
3. **了解** WandB 日志集成

---

## 理论背景

### 1. 检查点 (Checkpoint)

#### 1.1 保存检查点的原因

- **容错恢复**: 训练中断后可以从最近检查点恢复
- **模型选择**: 保存不同阶段的模型用于选择
- **分布式训练**: 支持从检查点恢复训练

#### 1.2 保存内容

一个完整的检查点通常包含:

```python
checkpoint = {
    'epoch': epoch,
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': config,
    'random_state': random.getstate(),
    'torch_state': torch.get_rng_state(),
    'cuda_state': torch.cuda.get_rng_state_all(),
}
```

#### 1.3 保存策略

- **固定间隔**: 每 N 步保存一次
- **最佳模型**: 保存验证损失最低的模型
- **保留最近 N 个**: 避免磁盘空间耗尽
- **分片保存**: 大模型使用分片检查点

### 2. 梯度累积 (Gradient Accumulation)

#### 2.1 动机

当 GPU 显存不足以支持大 batch size 时，可以使用梯度累积模拟大 batch 训练。

#### 2.2 原理

梯度累积将多个小 batch 的梯度累积起来，一次性更新参数:

```
for batch in data:
    # 前向传播
    loss = model(batch)
    # 缩放损失 (除以累积步数)
    loss = loss / accumulation_steps
    # 反向传播 (累积梯度)
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        # 梯度裁剪 (如果需要)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # 参数更新
        optimizer.step()
        optimizer.zero_grad()
```

#### 2.3 数学等价性

梯度累积在数学上等价于使用更大的 batch size:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{K}\sum_{i=1}^{K} \nabla L_i$$

其中 K 是累积步数。

### 3. 梯度裁剪 (Gradient Clipping)

#### 3.1 目的

防止梯度爆炸，确保训练稳定性。

#### 3.2 方法

**Gradient Clipping by Norm**:
$$\text{if } \|g\| > \text{max\_norm}:$$
$$g = g \cdot \frac{\text{max\_norm}}{\|g\|}$$

**Gradient Clipping by Value**:
$$g = \text{clamp}(g, -\text{clip\_value}, \text{clip\_value})$$

#### 3.3 典型值

- `max_norm`: 1.0 (Transformer) 或 5.0 (RNN)
- `clip_value`: 1.0 或 5.0

### 4. WandB 日志集成

Weights & Biases (WandB) 是常用的训练可视化工具:

- 损失曲线、学习率曲线
- 梯度统计 (均值、范数)
- GPU/CPU 内存使用
- 硬件监控

---

## 代码实现

### 项目结构

```
training/
├── checkpoint.py  # 检查点保存/加载
├── trainer.py     # 训练器实现
└── logger.py      # 日志记录
```

---

## 实践练习

### 练习 1: 实现训练器类

打开 `training/trainer.py`，实现基础训练器:

```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, config):
        """
        初始化训练器。

        Args:
            model: 语言模型
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 训练配置，包含:
                - accumulation_steps: 梯度累积步数
                - max_grad_norm: 梯度裁剪范数
                - log_interval: 日志输出间隔
                - save_interval: 检查点保存间隔
        """
        pass

    def save_checkpoint(self, filepath):
        """
        保存检查点。

        保存内容:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - epoch, step, config
        """
        # 实现: 使用 torch.save 保存状态
        pass

    def load_checkpoint(self, filepath):
        """
        加载检查点。

        Args:
            filepath: 检查点路径

        Returns:
            epoch, step
        """
        # 实现: 加载并恢复状态
        pass

    def train_step(self, batch):
        """
        执行单步训练。

        流程:
        1. 前向传播计算损失
        2. 梯度累积 (必要时缩放损失)
        3. 梯度裁剪 (达到累积步数时)
        4. 参数更新 (达到累积步数时)
        5. 学习率调度更新
        """
        pass

    def train_epoch(self, dataloader):
        """
        训练一个 epoch。
        """
        pass
```

### 练习 2: 实现梯度累积和裁剪

在 `train_step` 中实现:
- 梯度累积: 多个小 batch 累积后更新
- 梯度裁剪: `torch.nn.utils.clip_grad_norm_`

```python
def train_step(self, batch):
    # 前向传播
    loss = self.model(batch)
    loss = loss / self.accumulation_steps

    # 反向传播
    loss.backward()

    # 梯度裁剪
    if self.global_step % self.accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### 练习 3: 实现日志记录

```python
class TrainerLogger:
    def __init__(self, log_dir="runs", use_wandb=False):
        # 初始化日志目录和 WandB (如果启用)
        pass

    def log(self, metrics, step):
        # 记录损失、学习率、梯度范数等
        pass
```

---

## 测试验证

```bash
# 基本功能测试
pytest tutorial/stage02_pretrain/lesson08_checkpoint_trainer/testcases/basic_test.py -v

# 进阶测试
pytest tutorial/stage02_pretrain/lesson08_checkpoint_trainer/testcases/advanced_test.py -v
```

---

## 延伸阅读

- **分布式检查点**: 了解 FSDP 的分片检查点保存
- **最优检查点选择**: 了解 SWA (Stochastic Weight Averaging)
- **WandB 文档**: 了解更多高级功能
