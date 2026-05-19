# OVSeg 任务完整流程说明

## 📋 任务概述

**OVSeg (Open-Vocabulary Segmentation)** 是开放词汇分割任务，可以分割任意类别，不限于预定义的类别列表。

## 🔄 完整流程

### 1. 输入处理（前处理）

**用户输入**：
```
prompt = "person, car, tree, building"
```

**步骤1：解析类别** (`_get_classes_from_prompt`)
```python
# 从prompt中解析类别（用逗号分隔）
classes = [x.strip() for x in prompt.split(",") if len(x.strip()) > 0]
# 结果: ["person", "car", "tree", "building"]
# 保持用户输入的顺序！
```

**步骤2：构建example** (`_process_prompt`)
```python
example = {
    "sampled_cats": ["person", "car", "tree", "building"],  # 保持顺序
    "caption": None
}
```

**步骤3：Map函数处理** (`generic_seg_map_fn`)
```python
# 将sampled_cats转换为prompt格式
tag_categories(["person", "car", "tree", "building"])
# 生成: <p>person</p>, <p>car</p>, <p>tree</p>, <p>building</p>

# 构建完整prompt
question = "Can you segment the image based on the following categories: <p>person</p>, <p>car</p>, <p>tree</p>, <p>building</p>? Please output the segmentation masks."
```

**步骤4：设置Metadata** (`_set_metadata`)
```python
# ovseg所有类别都作为stuff处理
# 关键：索引必须与sampled_cats的顺序一致！
metadata.set(
    dataset_id_to_contiguous_id={0: 0, 1: 1, 2: 2, 3: 3},
    thing_dataset_id_to_contiguous_id={},  # ovseg没有thing类别
    stuff_dataset_id_to_contiguous_id={0: 0, 1: 1, 2: 2, 3: 3},
    thing_classes={},
    stuff_classes={0: "person", 1: "car", 2: "tree", 3: "building"},  # 索引对应类别名称
)
```

### 2. 模型推理

模型接收prompt，输出：
- `class_queries_logits`: [batch_size, num_queries, num_classes+1]
  - `num_classes = len(sampled_cats) = 4`
  - 最后一维是背景类
- `masks_queries_logits`: [batch_size, num_queries, height, width]

**类别索引对应关系**（关键！）：
- 索引 0 → "person" (第一个类别)
- 索引 1 → "car" (第二个类别)
- 索引 2 → "tree" (第三个类别)
- 索引 3 → "building" (第四个类别)

### 3. 后处理 (`generic_seg_postprocess_fn`)

**步骤1：处理mask** (`_panoptic_genseg_postprocess`)
```python
# 获取预测的类别索引
scores = F.softmax(mask_cls, dim=-1)[:, :-1]  # 去掉背景类
pred_score, pred_label = scores.max(-1)  # 每个query预测的类别索引

# 生成segments_info
segments_info = [
    {"category_id": 0, "score": 0.9, "isthing": False, ...},  # person
    {"category_id": 1, "score": 0.8, "isthing": False, ...},  # car
    {"category_id": 2, "score": 0.7, "isthing": False, ...},  # tree
    {"category_id": 3, "score": 0.6, "isthing": False, ...},  # building
]
```

**关键点**：
- `category_id` 是类别索引（0, 1, 2, 3）
- 必须与 `sampled_cats` 的顺序一致
- 必须与 `metadata.stuff_classes` 的索引对应

### 4. 可视化 (`draw_pan_seg`)

```python
# 从segments_info中获取category_id
for mask, sinfo in pred.semantic_masks():
    category_idx = sinfo["category_id"]  # 例如：0, 1, 2, 3
    
    # 从metadata中查找对应的类别名称
    text = self.metadata.stuff_classes[category_idx]
    # 例如：stuff_classes[0] = "person"
    #      stuff_classes[1] = "car"
    #      stuff_classes[2] = "tree"
    #      stuff_classes[3] = "building"
```

## ⚠️ 关键问题

### 问题：类别顺序不对应

**原因**：
1. 用户输入的类别顺序：`["person", "car", "tree", "building"]`
2. 模型输出的 `category_id` 必须按照这个顺序：0→person, 1→car, 2→tree, 3→building
3. Metadata 中的 `stuff_classes` 必须按照这个顺序设置

**解决方案**：
1. ✅ 保持用户输入的类别顺序（不要打乱）
2. ✅ Metadata 中的索引必须与类别顺序一致
3. ✅ 确保模型输出的 `category_id` 与 metadata 中的索引对应

## 🔍 调试方法

如果发现类别对应不上，检查：

1. **输入顺序**：
   ```python
   print("Input classes:", classes)  # 应该是 ["person", "car", "tree", "building"]
   ```

2. **Metadata设置**：
   ```python
   print("Metadata stuff_classes:", metadata.stuff_classes)
   # 应该是 {0: "person", 1: "car", 2: "tree", 3: "building"}
   ```

3. **模型输出**：
   ```python
   print("Segments info:", segments_info)
   # 检查 category_id 是否与输入顺序对应
   ```

4. **可视化**：
   ```python
   print("Category name:", metadata.stuff_classes[category_id])
   # 应该与输入prompt中的类别名称一致
   ```

## 📝 总结

OVSeg 任务的关键是**保持类别顺序的一致性**：
- 输入prompt的类别顺序
- Metadata中的索引顺序
- 模型输出的category_id顺序
- 可视化时的类别名称查找

所有环节必须保持一致，才能确保文本和mask的对应关系正确。

