# 分割任务无法输出<seg>标记的解决方案

## 问题分析

在混合训练imgconv任务（9万图文对）和各种分割任务时，模型可能更倾向于生成文本回答而不是`<seg>`标记，导致无法触发segmentor。

### 根本原因

1. **数据不平衡**：imgconv任务数据量（9万）远大于分割任务，模型更倾向于学习生成文本
2. **Loss权重相同**：`llm_loss_weight=1.0`和`seg_loss_weight=1.0`，但数据量差异导致实际loss贡献不平衡
3. **训练策略**：模型可能没有充分学习到分割任务需要输出`<seg>`标记

## 解决方案

### 方案1：调整Loss权重（推荐）

在配置文件中增加分割任务的loss权重，确保分割任务得到充分训练：

```python
model = dict(
    type=XSamModel,
    # ... 其他配置 ...
    llm_loss_weight=1.0,      # 保持LLM loss权重
    seg_loss_weight=2.0,      # 增加分割loss权重（从1.0增加到2.0）
    # ... 其他配置 ...
)
```

**原理**：通过增加`seg_loss_weight`，即使分割任务数据量较少，也能确保分割损失在总损失中占更大比重，强制模型学习输出`<seg>`标记。

### 方案2：调整数据采样比例

在配置文件中调整`repeats_scale`，增加分割任务的采样频率：

```python
# 当前配置
geochat_imgconv_dataset = dict(
    # ...
    repeats_scale=1,  # imgconv: 99,740样本
)

sota_genseg_dataset = dict(
    # ...
    repeats_scale=5,  # genseg: 15,732 * 5 ≈ 78,660样本
)

# 建议调整
geochat_imgconv_dataset = dict(
    # ...
    repeats_scale=0.8,  # 减少imgconv采样，99,740 * 0.8 ≈ 79,792样本
)

sota_genseg_dataset = dict(
    # ...
    repeats_scale=8,  # 增加genseg采样，15,732 * 8 ≈ 125,856样本
)
```

**原理**：通过调整采样比例，使分割任务在每个epoch中出现更频繁，模型有更多机会学习输出`<seg>`标记。

### 方案3：使用课程学习策略

分阶段训练，先训练分割任务，再混合训练：

**阶段1**：只训练分割任务（1-2个epoch）
```python
combined_train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        # 暂时注释掉imgconv
        # geochat_imgconv_dataset,
        sota_genseg_dataset,
        sota_ovseg_dataset,
        remotesam_refseg_dataset,
    ],
)
```

**阶段2**：混合训练所有任务（继续训练）
```python
combined_train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        geochat_imgconv_dataset,  # 重新加入
        sota_genseg_dataset,
        sota_ovseg_dataset,
        remotesam_refseg_dataset,
    ],
)
```

**原理**：先让模型充分学习分割任务输出`<seg>`标记，再混合训练时模型已经掌握了这个能力。

### 方案4：增强训练时的监督信号

确保训练数据中分割任务的输出模板正确包含`<seg>`标记。检查`dataset_map_fn`的输出：

```python
# 在generic_seg_map_fn.py中，确保MASK_ANSWER_LIST包含<SEG>
MASK_ANSWER_LIST = [
    f"{DEFAULT_SEG_TOKEN}.",  # ✅ 正确
    f"It is {DEFAULT_SEG_TOKEN}.",
    f"Sure, {DEFAULT_SEG_TOKEN}.",
    # ...
]
```

### 方案5：使用延迟停止条件（已实现，需确认启用）

确保在推理时使用延迟停止条件，强制模型生成`<seg>`后才停止：

```python
# 在xsam/utils/config.py中，DelayedStopWordStoppingCriteria已实现
# 确保在evaluation时使用：
stop_criteria = setup_model_config(model, cfg)
```

## 推荐组合方案

**最佳实践**：同时使用方案1和方案2

1. **增加seg_loss_weight到2.0-3.0**
2. **调整数据采样比例，使分割任务数据量接近imgconv**
3. **确保延迟停止条件已启用**

## 验证方法

训练过程中监控以下指标：

1. **Loss比例**：检查`loss_seg / loss_llm`的比例，应该接近1:1或更高
2. **生成检查**：定期在验证集上检查分割任务是否输出了`<seg>`标记
3. **Mask生成率**：统计推理时成功生成mask的比例

## 代码修改示例

### 修改配置文件

```python
# 在配置文件中修改
model = dict(
    type=XSamModel,
    # ... 其他配置 ...
    llm_loss_weight=1.0,
    seg_loss_weight=2.5,  # 从1.0增加到2.5
    # ... 其他配置 ...
)

# 调整数据采样
geochat_imgconv_dataset = dict(
    # ...
    repeats_scale=0.7,  # 从1.0减少到0.7
)

sota_genseg_dataset = dict(
    # ...
    repeats_scale=10,  # 从5增加到10
)
```

## 注意事项

1. **不要过度增加seg_loss_weight**：过大的权重可能导致训练不稳定，建议在2.0-3.0之间
2. **保持数据多样性**：虽然要平衡数据量，但不要完全忽略imgconv任务，它有助于提升模型的通用能力
3. **监控训练稳定性**：调整后密切监控loss曲线，确保训练稳定

