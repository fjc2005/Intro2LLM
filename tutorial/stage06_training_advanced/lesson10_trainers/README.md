# 课时10：训练器体系实现

## 学习目标

1. 理解Trainer抽象类的设计原则
2. 掌握训练循环的完整结构：forward → backward → step
3. 实现梯度累积、混合精度训练、梯度裁剪
4. 掌握检查点保存与恢复、日志记录
5. 实现PretrainTrainer、SFTTrainer等完整训练器

---

## 1. 基础Trainer设计

### 1.1 训练循环核心结构

```
通用训练循环:
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 1. 前向传播
            loss = model(batch)

            # 2. 反向传播
            loss.backward()

            # 3. 梯度更新
            optimizer.step()
            optimizer.zero_grad()

            # 4. 日志记录
            log_metrics(loss, ...)
```

### 1.2 Trainer抽象基类

```python
class BaseTrainer:
    """
    训练器基类

    提供通用训练功能:
    - 训练循环
    - 检查点管理
    - 日志记录
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional = None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        pass

    def training_step(self, batch: Dict) -> Dict[str, float]:
        """单步训练，子类实现"""
        raise NotImplementedError

    def train(self, dataloader, num_epochs: int):
        """主训练循环"""
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                metrics = self.training_step(batch)

                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
```

---

## 2. 混合精度训练

### 2.1 为什么需要混合精度？

```
问题: FP32训练占用显存大、计算慢

解决方案: 混合精度训练 (FP16/BF16 + FP32)

优势:
    - 显存占用减半
    - 计算速度提升 (Tensor Core加速)
    - 保持FP32精度进行关键计算
```

### 2.2 AMP自动混合精度

```python
# 导入混合精度相关模块
from torch.cuda.amp import autocast, GradScaler

# Step 1: 创建梯度缩放器
# GradScaler用于自动缩放损失值，防止FP16梯度下溢
# 初始缩放因子通常设为动态值

# Step 2: 自动混合精度前向传播
# autocast上下文管理器自动决定哪些操作使用FP16
# 前向传播使用FP16可以减少显存和加速计算

# Step 3: 缩放反向传播
# 使用scaler.scale()对损失进行缩放后再反向传播
# 缩放后的梯度值更大，避免下溢问题

# Step 4: 梯度裁剪(可选)
# 对缩放后的梯度进行裁剪

# Step 5: 参数更新
# 使用scaler.step()更新参数
# scaler会自动反缩放梯度

# Step 6: 更新缩放因子
# 根据梯度状态更新缩放因子
```

---

## 3. 梯度累积与裁剪

### 3.1 梯度累积

```
目的: 模拟大batch size训练

原理:
    实际batch_size = per_device_batch_size * gradient_accumulation_steps

实现:
    for i, batch in enumerate(dataloader):
        loss = model(batch) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### 3.2 梯度裁剪

```python
# 防止梯度爆炸
# 使用torch.nn.utils.clip_grad_norm_函数

# 原理:
# 计算所有参数梯度的总范数
# 如果总范数超过max_norm，按比例缩放所有梯度
# 使得缩放后的总范数等于max_norm

# 参数:
# - model.parameters(): 模型所有参数
# - max_norm: 最大梯度范数(通常1.0)
# 返回值: 梯度范数(用于日志记录)
```

---

## 4. 检查点管理

### 4.1 保存检查点

```python
# 检查点包含的内容:
# 1. epoch: 当前训练轮数
# 2. model_state_dict: 模型的参数字典
# 3. optimizer_state_dict: 优化器状态(动量等)
# 4. scheduler_state_dict: 学习率调度器状态
# 5. loss: 当前损失值

# 实现思路:
# Step 1: 创建字典，保存所有需要恢复的信息
# Step 2: 使用torch.save保存到文件
# 建议命名: checkpoint-{step}.pt 或 checkpoint-{epoch}.pt
```

### 4.2 加载检查点

```python
# 实现思路:
# Step 1: 使用torch.load加载检查点文件
# Step 2: 从检查点恢复模型参数
# Step 3: (可选)恢复优化器状态
# Step 4: (可选)恢复调度器状态
# Step 5: 返回epoch用于继续训练
```

---

## 5. 各阶段Trainer实现

### 5.1 PretrainTrainer

```python
class PretrainTrainer(BaseTrainer):
    """预训练Trainer"""

    def training_step(self, batch):
        input_ids = batch['input_ids']
        labels = batch['labels']

        with autocast():
            outputs = self.model(input_ids)
            logits = outputs['logits']
            loss = self.criterion(logits.view(-1, vocab_size), labels.view(-1))

        return {'loss': loss.item()}
```

### 5.2 SFTTrainer

```python
class SFTTrainer(BaseTrainer):
    """监督微调Trainer"""

    def training_step(self, batch):
        # SFT使用loss masking
        input_ids = batch['input_ids']
        labels = batch['labels']  # prompt部分为-100

        with autocast():
            outputs = self.model(input_ids)
            loss = self.criterion(
                outputs['logits'].view(-1, vocab_size),
                labels.view(-1)
            )

        return {'loss': loss.item()}
```

### 5.3 DPOTrainer

```python
class DPOTrainer(BaseTrainer):
    """DPO训练Trainer"""

    def training_step(self, batch):
        # 同时计算policy和reference的logprobs
        chosen_logps = self.compute_logprobs(self.model, batch['chosen'])
        rejected_logps = self.compute_logprobs(self.model, batch['rejected'])

        with torch.no_grad():
            ref_chosen_logps = self.compute_logprobs(self.ref_model, batch['chosen'])
            ref_rejected_logps = self.compute_logprobs(self.ref_model, batch['rejected'])

        loss = self.dpo_loss(chosen_logps, rejected_logps,
                             ref_chosen_logps, ref_rejected_logps)

        return {'loss': loss.item()}
```

---

## 6. 实现指引

### 6.1 training/trainer.py

```python
class BaseTrainer:
    def __init__(self, model, optimizer, ...):
        # 初始化模型、优化器、调度器
        pass

    def training_step(self, batch):
        # Step 1: 混合精度前向
        # 使用自动混合精度上下文管理器
        # 计算损失值

        # Step 2: 缩放反向传播
        # 使用GradScaler对损失进行缩放后执行反向传播

        # Step 3: 梯度裁剪
        # 先对梯度进行反缩放
        # 使用clip_grad_norm_对模型参数梯度进行裁剪

        # Step 4: 优化器步骤
        # 调用scaler的step方法执行优化器更新
        # 更新scaler的缩放因子

        pass
```

---

## 7. 常见陷阱

1. **梯度累积的loss缩放**: 记得除以accumulation_steps
2. **检查点一致性**: 同时保存model、optimizer、scheduler状态
3. **混合精度的上溢**: loss scaling处理梯度下溢
4. **分布式训练**: 使用DistributedDataParallel

---

## 8. 课后练习

1. 实现完整的训练循环，包含梯度累积和裁剪
2. 对比FP32和FP16训练的显存和速度
3. 实现检查点恢复功能
4. 添加训练指标监控（如困惑度）
