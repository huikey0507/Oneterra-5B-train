# SAR 对齐训练说明（S3 → A → B∥B2 + SARDET）

本文记录从光学底座 **S3** 到 SAR 对齐（**A → 联合 B∥B2(+SARDET)**）的完整流程、模块冻结、数据配比与启动命令。

> 配置与代码在主仓 `xsam/`（可入库）；训练产物写在 `wkdrs_*`（`.gitignore`，不上 GitHub）。  
> **推荐路径**：S3 → A → **联合 B∥B2 + SARDET**（跳过单独的旧 B / 旧 B2）。  
> **历史路径**：S3 → A → B → B2_v2（仅 ParSAR），再可选从 B2 续训 SARDET。

---

## 1. 总览

```
S3 光学混合微调底座
  └─ Stage A: SAR 语言侧对齐（projectors + router）
       └─ 联合 B∥B2 + SARDET: 同时训 connector + cond_adapter
            （ParSAR 保 stuff + SARDET 扩多类 thing + 光学 replay）
```

| 阶段 | 配置文件（主仓） | 默认 workdir | 主要产出 |
|------|------------------|--------------|----------|
| S3 | `s3_mixed_fineture_base/...`（已完成） | `checkpoints/s3_mixed_fineture_v3.2/` | `pytorch_model.bin` |
| A | `s_sar_align/xsam_sar_A_projector_align_4xA40.py` | `wkdrs_sar_align_A/` | `iter_91000.pth` 等 |
| 联合 B∥B2+SARDET | `s_sar_align/xsam_sar_B2_joint_sardet_4xA40.py` | `wkdrs_sar_align_B2_joint_sardet/` | `pytorch_model.bin` |

配置根目录：

```text
xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/
├── s3_mixed_fineture_base/          # S3
└── s_sar_align/
    ├── xsam_sar_A_projector_align_4xA40.py
    ├── xsam_sar_B_connector_pano_4xA40.py          # 历史 Stage B
    ├── xsam_sar_B2_cond_adapter_pano_4xA40.py      # 历史 B2（仅 ParSAR）
    ├── xsam_sar_B2_cond_adapter_pano_sardet_4xA40.py  # 从 B2_v2 续训 SARDET（备选）
    └── xsam_sar_B2_joint_sardet_4xA40.py           # ★ 推荐：联合训
```

---

## 2. 各阶段在训什么 / 冻什么

### 2.1 S3（前置，一般已完成）

- **目标**：光学遥感开集分割 / 多任务能力底座。
- **产出路径（本实验默认）**：  
  `/mnt_llm_A100_V1/shui/LAE/OneTerra-train/checkpoints/s3_mixed_fineture_v3.2/pytorch_model.bin`
- 本说明不展开 S3 超参；后面所有 SAR 阶段都把它当冻结的光学真值底座加载。

### 2.2 Stage A — SAR 语言对齐

| | |
|--|--|
| **可训** | `sar_visual_projector`、`sar_seg_projector`、`modality_router` |
| **冻结** | LLM、SigLIP、SAM encoder、光学 projectors/connector |
| **初始化** | S3（`PREV_S3_CKPT`，支持 `.bin` 或 ZeRO `iter_*.pth` 目录） |
| **数据** | SAR pretrain imgconv（×1.0）+ SAR SFT（×0.25）+ SkyScript 光学 replay（×0.15） |
| **损失** | `loss_llm` + `0.1 * loss_gate` |
| **LR** | `5e-4`，warmup 3%，cosine |
| **目的** | 让 SAR 图文进入 LLM 空间；路由光学/SAR；**还不做全景分割主训** |

### 2.3 联合 B∥B2 + SARDET（推荐）

| | |
|--|--|
| **可训** | `sar_seg_connector`、`sar_cond_adapter`、`modality_router` |
| **冻结** | LLM、SigLIP、SAM、Mask2Former decoder、光学 adapters、`llm_projector`、**SAR projectors（A 已训好）** |
| **初始化** | ① S3 全量 → ② 只叠 Stage A；connector 从 optical→SAR 拷贝起步；`sar_cond_adapter` **零初始化** |
| **不要加载** | 旧 Stage B / B2_v2（本阶段重训 connector+cond） |
| **LR** | `5e-4`（冷启动，与历史 A/B/B2 一致） |

**数据配比（effective ≈ raw × repeats_scale）**

| 数据源 | 路径要点 | scale | effective（约） | 占比 | 作用 |
|--------|----------|------:|----------------:|-----:|------|
| ParSAR | `oneterra_data/sarseg/pansar2`（ship/sea/land） | **10.0** | 36420 | 42% | **stuff 锚点**，保 ParSAR 测试 |
| SARDET | `detection2mask_tool/output`（6 类 thing） | **0.35** | 33073 | 38% | 多类开集 thing |
| 光学 pano | `datas/pano` | **0.15** | 16660 | 19% | 防遗忘 + 同义词 |

- SAR 两侧均 `use_cat_synonym=True`。  
- 约 **2.1 万 iter / 4×A40**（约为旧 B2_v2 的 ~1.9× 时长）。  
- SARDET **只有 thing、无 stuff**；sea/land 只能靠 ParSAR。

### 2.4 模块对照（一眼看完）

| 模块 | A | 联合 B∥B2 |
|------|---|-----------|
| `sar_visual_projector` / `sar_seg_projector` | 训 | **冻** |
| `sar_seg_connector` | — | **训** |
| `sar_cond_adapter` | 无 | **训（零初始化）** |
| `modality_router` | 训 | **训** |
| 光学主干 + 光学 adapters | 冻 | 冻 |

---

## 3. 环境与通用约定

```bash
cd /mnt_llm_A100_V1/shui/LAE/OneTerra-train

# 常用环境变量
export ROOT_DIR=$PWD/
export DATA_DIR=$PWD/datas/
export INIT_DIR=$PWD/inits/
export CODE_DIR=$PWD/xsam/
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

- 硬件默认：**4×NVIDIA A40**，DeepSpeed ZeRO-2，`batch=1`，`accum=8`。  
- 权重：优先用合并好的 `pytorch_model.bin`；Stage A 的 ZeRO 目录（如 `iter_91000.pth/`）也可被 `sar_adapter_pth` 加载。  
- **加载顺序禁忌**：联合阶段不要把「只有 adapter 的小 bin」当成整模 `s2`；必须 **S3 全量 + 再叠 A**。

---

## 4. 启动命令

### 4.1 Stage A（从 S3）

配置：`xsam/xsam/configs/.../s_sar_align/xsam_sar_A_projector_align_4xA40.py`

可用通用多卡脚本（与历史一致）：

```bash
cd /mnt_llm_A100_V1/shui/LAE/OneTerra-train

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WORK_DIR=$PWD/wkdrs_sar_align_A \
PREV_S3_CKPT=$PWD/checkpoints/s3_mixed_fineture_v3.2/pytorch_model.bin \
bash runs/run3.sh \
  --modes train \
  --config xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s_sar_align/xsam_sar_A_projector_align_4xA40.py \
  --work-dir $PWD/wkdrs_sar_align_A
```

或（若 `PREV_S3_CKPT` 指向 ZeRO 目录）：

```bash
PREV_S3_CKPT=$PWD/checkpoints/s3_mixed_fineture_v3.2/iter_70504.pth \
... # 同上
```

**检查点**：训练结束后确认存在  
`wkdrs_sar_align_A/iter_*.pth`（本流程后续默认用 `iter_91000.pth`）。

### 4.2 联合 B∥B2 + SARDET（推荐，从 S3 + A）

专用脚本（主仓）：

```bash
cd /mnt_llm_A100_V1/shui/LAE/OneTerra-train

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash runs/run_B2_joint_sardet_4xA40.sh
```

可选覆盖：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WORK_DIR=$PWD/wkdrs_sar_align_B2_joint_sardet \
PREV_S3_CKPT=$PWD/checkpoints/s3_mixed_fineture_v3.2/pytorch_model.bin \
PREV_SAR_A_CKPT=$PWD/wkdrs_sar_align_A/iter_91000.pth \
MASTER_PORT=29534 \
bash runs/run_B2_joint_sardet_4xA40.sh
```

**期望日志要点**：

- `Load s2_pretrained_pth from .../s3_mixed_fineture_v3.2/...`
- `Load sar_adapter_pth from .../wkdrs_sar_align_A/...`
- `Froze SAR visual/seg projectors (keep training sar_seg_connector/router)`
- `SarOVSegDataset: loaded 3642 ... sar_parsar_pano_ovseg_train`
- `SarOVSegDataset: loaded 94493 ... sar_sardet_pano_ovseg_train`
- 可训参数出现在 `sar_seg_connector` / `sar_cond_adapter` / `modality_router`

产物：`wkdrs_sar_align_B2_joint_sardet/pytorch_model.bin`

---

## 5. 历史路径（备查，非推荐新开）

若需复现旧实验：

| 阶段 | 配置 | 初始化 | 数据 |
|------|------|--------|------|
| B | `xsam_sar_B_connector_pano_4xA40.py` | S3 + A | ParSAR×8 + 光学×0.15 |
| B2_v2 | `xsam_sar_B2_cond_adapter_pano_4xA40.py` | S3 + A + B，cond 零初始化 | 同 B |
| 从 B2 续训 SARDET | `xsam_sar_B2_cond_adapter_pano_sardet_4xA40.py` + `runs/run_B2_sardet_4xA40.sh` | **B2_v2 bin**，`lr=2e-4` | ParSAR×10 + SARDET×0.35 + 光学×0.15 |

旧 B2_v2 实测：4×A40，约 **11456 iter**，workdir `wkdrs_sar_align_B2_cond_v2/`。

---

## 6. 数据路径速查

| 名称 | 路径 |
|------|------|
| S3 权重 | `checkpoints/s3_mixed_fineture_v3.2/pytorch_model.bin` |
| Stage A 权重 | `wkdrs_sar_align_A/iter_91000.pth` |
| ParSAR | `/mnt_llm_A100_V1/shui/oneterra_data/sarseg/pansar2/` |
| SARDET panoptic | `/mnt_llm_A100_V1/shui/LAE/detection2mask_tool/output/` |
| 光学 pano | `datas/pano/` |
| 模型初始化 | `inits/`（Phi-3 / SigLIP / SAM / Mask2Former） |

**SARDET 注意**：

- 训练用 `annotations/panoptic_train.json` + `annotations/panoptic_train/` + `train/`。  
- 6 类全为 thing：ship / aircraft / car / tank / bridge / harbor。  
- 类别极度不平衡（ship 为主）；无 sea/land。  
- 目前无 `panoptic_test.json`（仅有 `panoptic_test/` 图）。

---

## 7. 调参备忘

| 现象 | 建议 |
|------|------|
| ParSAR 上 sea/land 掉点 | ParSAR `repeats_scale` ↑（如 12）或 SARDET ↓（如 0.25） |
| 多类 thing 仍弱、stuff 仍稳 | SARDET 略升（如 0.4）或加长 epoch |
| 联合训早期 loss 很晃 | 可临时先冻 `sar_cond_adapter` 一小段再解冻（一般不必） |
| 从已收敛 B2 微调 | 用续训配置，`lr=2e-4`，不要用联合冷启动的 `5e-4` |

---

## 8. 一键推荐流程（最短）

```bash
# 0) 确认 S3 与 A 已就绪
ls checkpoints/s3_mixed_fineture_v3.2/pytorch_model.bin
ls -d wkdrs_sar_align_A/iter_91000.pth

# 1) 若还没有 A：先跑 4.1 Stage A

# 2) 联合 B∥B2 + SARDET
CUDA_VISIBLE_DEVICES=0,1,2,3 bash runs/run_B2_joint_sardet_4xA40.sh
```

---

## 9. 相关代码（须入库，勿只放在 wkdrs）

- `xsam/xsam/dataset/sar_ov_seg_dataset.py`
- `xsam/xsam/model/modules/sar_cond_adapter.py`
- `xsam/xsam/model/modules/modality_router.py`
- `xsam/xsam/model/xsam.py`（SAR 加载 / freeze 逻辑）
- `xsam/xsam/dataset/utils/cat_synonyms.py`（含 sea/land/aircraft/car/tank/harbor 等）
- `runs/run_B2_joint_sardet_4xA40.sh`

文档版本：与联合配置 `xsam_sar_B2_joint_sardet_4xA40.py` 一致。
