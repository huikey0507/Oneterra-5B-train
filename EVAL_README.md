# X-SAM 验证测试说明

## 概述

本脚本用于在当前服务器上测试正在训练的X-SAM模型性能。增强版的`eval.py`会显示所有验证结果，包括：
- **语言输出**：LLM生成的问题和答案
- **图片可视化**：分割结果的可视化图片
- **评估指标**：所有数据集的评估指标

## 使用方法

### 方法1: 使用提供的脚本（推荐）

```bash
cd /mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM
bash run_eval_test.sh
```

### 方法2: 直接运行Python命令

#### 使用训练中的检查点（iter_*.pth目录）

```bash
cd /mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM

# 方式1: 指定iter_*.pth目录（推荐，脚本会自动查找模型文件）
python xsam/xsam/tools/eval.py \
    xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py \
    --work-dir ./eval_results/s3_mixed_finetune_test \
    --pth_model ./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_28000.pth

# 方式2: 直接指定模型状态文件（如果方式1不工作）
python xsam/xsam/tools/eval.py \
    xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py \
    --work-dir ./eval_results/s3_mixed_finetune_test \
    --pth_model ./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/iter_28000.pth/mp_rank_00_model_states.pt
```

#### 使用最终模型（pytorch_model.bin，训练完成后）

```bash
python xsam/xsam/tools/eval.py \
    xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3.py \
    --work-dir ./eval_results/s3_mixed_finetune_test \
    --pth_model ./wkdrs/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune3/pytorch_model.bin
```

## 参数说明

- `config`: 配置文件路径
- `--work-dir`: 结果保存目录（新建一个目录避免覆盖训练结果）
- `--pth_model`: 模型检查点路径，支持以下格式：
  - **训练中的检查点**：`iter_28000.pth` 目录（DeepSpeed格式）
    - 脚本会自动查找 `mp_rank_00_model_states.pt` 文件
    - 或直接指定：`iter_28000.pth/mp_rank_00_model_states.pt`
  - **最终模型**：`pytorch_model.bin` 文件（训练完成后）
  - **最新检查点**：使用 `"latest"` 自动查找最新检查点

## 输出结果

验证完成后，结果会保存在`--work-dir`指定的目录下：

```
eval_results/s3_mixed_finetune_test/
├── pred_data/                    # 预测数据（用于评估指标计算）
│   ├── sota_panoptic_genseg_val/
│   ├── sota_panoptic_ovseg_val/
│   ├── remotesam_val_refseg/
│   └── remotesam_test_refseg/
├── visualizations/               # 可视化图片（前50个样本）
│   ├── sota_panoptic_genseg_val/
│   │   ├── sample_00000.png
│   │   ├── sample_00001.png
│   │   └── ...
│   ├── sota_panoptic_ovseg_val/
│   ├── remotesam_val_refseg/
│   └── remotesam_test_refseg/
└── llm_outputs/                  # LLM输出（JSON格式）
    ├── sota_panoptic_genseg_val_llm_outputs.json
    ├── sota_panoptic_ovseg_val_llm_outputs.json
    ├── remotesam_val_refseg_llm_outputs.json
    └── remotesam_test_refseg_llm_outputs.json
```

## 控制台输出

验证过程中，控制台会显示：

1. **每个样本的LLM输出**：
   ```
   ================================================================================
   Sample 1 - sota_panoptic_genseg_val
   Image: xxx.jpg
   LLM Question: Can you segment...
   LLM Answer: <SEG>person</SEG>...
   ================================================================================
   ```

2. **评估指标**：
   ```
   sota_panoptic_genseg_val evaluation results:
   [评估指标表格]
   ```

3. **进度信息**：
   - 每个数据集的评估进度
   - 可视化图片保存进度
   - 失败样本数量

## 注意事项

1. **路径修改**：根据实际情况修改脚本中的路径
   - 模型路径：`--pth_model`参数
   - 配置文件路径：第一个参数
   - 结果保存路径：`--work-dir`参数

2. **检查点格式**：
   - **训练中**：使用 `iter_28000.pth` 目录（DeepSpeed格式）
   - **训练完成**：使用 `pytorch_model.bin` 文件
   - 脚本会自动检测并处理不同的检查点格式

3. **显存要求**：确保有足够的GPU显存运行模型

4. **可视化数量**：默认保存前50个样本的可视化图片，可在代码中修改`max_vis_samples`参数

5. **多GPU**：如果使用多GPU，脚本会自动处理分布式评估

## 故障排除

### 1. 模型文件不存在
```
错误: 模型文件或目录不存在: xxx/iter_28000.pth
```
**解决**：
- 检查模型路径是否正确
- 如果使用 `iter_*.pth` 目录，确保目录存在且包含 `mp_rank_00_model_states.pt` 文件
- 如果使用 `pytorch_model.bin`，确保文件已从训练服务器复制过来
- 可以尝试直接指定完整路径：`iter_28000.pth/mp_rank_00_model_states.pt`

### 2. 配置文件不存在
```
错误: 配置文件不存在: xxx.py
```
**解决**：检查配置文件路径是否正确

### 3. 可视化失败
如果某些样本的可视化失败，不会影响整体评估，错误信息会记录在日志中

### 4. 显存不足
如果遇到OOM错误，可以：
- 减小batch size（在配置文件中）
- 减少可视化样本数量（修改`max_vis_samples`参数）

## 查看结果

### 查看评估指标
评估指标会在控制台输出，也会保存在`pred_data`目录下的各个子目录中

### 查看可视化图片
```bash
# 查看某个数据集的可视化结果
ls eval_results/s3_mixed_finetune_test/visualizations/sota_panoptic_genseg_val/

# 使用图片查看器打开
eog eval_results/s3_mixed_finetune_test/visualizations/sota_panoptic_genseg_val/sample_00000.png
```

### 查看LLM输出
```bash
# 查看JSON格式的LLM输出
cat eval_results/s3_mixed_finetune_test/llm_outputs/sota_panoptic_genseg_val_llm_outputs.json | jq '.[0]'
```

## 增强功能说明

相比原始`eval.py`，增强版添加了以下功能：

1. **语言输出显示**：实时显示每个样本的LLM问题和答案
2. **图片可视化**：自动保存分割结果的可视化图片
3. **结果保存**：将所有LLM输出保存为JSON文件，方便后续分析
4. **详细日志**：更详细的进度和错误信息

