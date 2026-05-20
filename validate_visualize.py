#!/usr/bin/env python3
"""
第一阶段训练可视化验证脚本
基于训练配置，使用PanoSegDataset进行验证和可视化
支持自定义可视化结果路径
"""

import os
import sys
import argparse
from pathlib import Path
from os import getenv

# 添加xsam模块路径
# 从 validate_visualize.py 到项目根目录需要向上4级
# wkdrs/s1_seg_finetune/xsam_sota_s1_finetune/validate_visualize.py -> X-SAM/X-SAM/
xsam_path = Path(__file__).parent.parent.parent.parent / "xsam"
if not xsam_path.exists():
    # 如果上面的路径不存在，尝试直接使用绝对路径
    xsam_path = Path("/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/xsam")
    if not xsam_path.exists():
        print(f"❌ 错误：找不到xsam模块路径")
        print(f"   尝试的路径1: {Path(__file__).parent.parent.parent.parent / 'xsam'}")
        print(f"   尝试的路径2: {xsam_path}")
        sys.exit(1)

# 确保路径在sys.path中
xsam_path_str = str(xsam_path)
if xsam_path_str not in sys.path:
    sys.path.insert(0, xsam_path_str)
print(f"🔍 添加xsam模块路径: {xsam_path}")
print(f"🔍 Python路径: {sys.path[:3]}...")  # 只显示前3个路径

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mplc

# 配置matplotlib支持中文显示
# 尝试使用支持中文的字体，如果系统没有则使用默认字体
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置支持中文的字体"""
    # 常见的中文字体列表（按优先级排序）
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                     'Arial Unicode MS', 'STHeiti', 'STSong', 'Noto Sans CJK SC',
                     'Source Han Sans CN', 'Droid Sans Fallback']
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts + ['DejaVu Sans']
        print(f"✅ 已配置matplotlib中文字体: {selected_font}")
    else:
        # 如果没有找到中文字体，尝试使用DejaVu Sans并抑制警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️  未找到中文字体，将使用DejaVu Sans（中文可能显示为方块）")
        print("   提示：可以安装中文字体（如SimHei、Microsoft YaHei）以正确显示中文")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 抑制字体缺失警告（如果确实没有中文字体，警告是预期的）
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

setup_chinese_font()

# 导入必要的模块
try:
    from xsam.dataset.pano_seg_dataset import PanoSegDataset
    from xsam.dataset.collate_fns import xsam_collate_fn
    from xsam.model import XSamModel
    print("✅ 成功导入xsam模块")
except ImportError as e:
    print(f"❌ 导入xsam模块失败: {e}")
    print(f"   当前Python路径: {sys.path[:5]}")
    print(f"   xsam路径是否存在: {xsam_path.exists()}")
    if xsam_path.exists():
        print(f"   xsam目录内容: {list(xsam_path.iterdir())[:5]}")
    import traceback
    traceback.print_exc()
    print("\n💡 提示：如果缺少依赖包，请确保已安装所有必需的依赖")
    sys.exit(1)


def convert_checkpoint_to_pytorch(checkpoint_dir):
    """将DeepSpeed检查点转换为PyTorch格式"""
    print(f"🔄 转换检查点: {checkpoint_dir}")
    
    converted_file = Path(checkpoint_dir) / "pytorch_model.bin"
    if converted_file.exists():
        print(f"✅ 已存在转换后的权重文件: {converted_file}")
        return str(converted_file)
    
    model_files = list(Path(checkpoint_dir).glob("*_model_states.pt"))
    if not model_files:
        print("❌ 未找到模型状态文件")
        return None
    
    model_file = model_files[0]
    print(f"🎯 使用模型文件: {model_file}")
    
    try:
        checkpoint = torch.load(str(model_file), map_location='cpu')
        
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        output_file = Path(checkpoint_dir) / "pytorch_model.bin"
        torch.save(new_state_dict, output_file)
        
        print(f"✅ 成功转换为PyTorch格式: {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_trained_model(checkpoint_path, device='cuda'):
    """
    加载训练好的 X-SAM 模型
    基于训练配置文件中的模型配置
    """
    print("🔨 加载训练好的X-SAM模型...")
    print(f"🔧 使用设备: {device}")
    print(f"🔧 使用数据类型: torch.bfloat16 (与训练时一致)")
    
    try:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            print("🔍 检测到DeepSpeed检查点目录，正在转换...")
            converted_path = convert_checkpoint_to_pytorch(checkpoint_path)
            if not converted_path:
                print("❌ 检查点转换失败")
                return None
            checkpoint_path = converted_path

        # 从训练配置中读取路径（使用绝对路径）
        root_dir = '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/'
        init_dir = root_dir + 'inits/'
        
        seg_encoder_name_or_path = init_dir + "sam-vit-large"
        seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"
        s1_pretrained_pth = root_dir + "checkpoints/s1_seg_finetune/xsam_sam_large_m2f_e36_gpu16_seg_finetune/pytorch_model.bin"
        
        # 第一步：使用预训练参数初始化模型结构
        print("🔧 第一步：使用预训练参数初始化模型结构...")
        model = XSamModel(
            llm=None,
            tokenizer=None,
            visual_encoder=None,
            freeze_segmentor_encoder=False,
            use_activation_checkpointing=False,
            postprocess_fn="xsam.dataset.process_fns.generic_seg_postprocess_fn",
            connector_type="conv",
            seg_select_layers=[6, 12, 18, 24],
            connector_hidden_dim=512,
            connector_scale_factor=[4, 2, 1, 0.5],
            extract_seg_embeds=True,
            s1_pretrained_pth=s1_pretrained_pth,
            segmentor=dict(
                type="xsam.model.segmentors.XSegmentor",
                encoder=dict(
                    type="xsam.model.segmentors.sam.SamModel.from_pretrained",
                    pretrained_model_name_or_path=seg_encoder_name_or_path,
                    trust_remote_code=True,
                    torch_dtype="torch.bfloat16",
                ),
                decoder=dict(
                    type="xsam.model.segmentors.mask2former.Mask2FormerModel._from_config",
                    config=dict(
                        type="xsam.model.segmentors.mask2former.Mask2FormerConfig.from_pretrained",
                        pretrained_model_name_or_path=seg_decoder_name_or_path,
                        use_backbone=False,
                        feature_channels=[512, 1024, 2048],
                        num_feature_levels=3,
                        num_labels=41,  # Pano数据集：0-40，背景类为41
                        trust_remote_code=True,
                    ),
                    torch_dtype="torch.bfloat16",
                ),
                torch_dtype="torch.bfloat16",
                reinit_decoder=True,
                close_cls=True,
            ),
        )
        model = model.to(device)
        print("✅ 第一步完成：预训练参数加载成功")

        # 第二步：加载训练好的参数
        print("🔧 第二步：加载训练好的参数...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 加载训练好的参数
        model.load_state_dict(checkpoint, strict=False)
        print("✅ 第二步完成：训练好的参数加载成功")

        # 设置为评估模式
        model.eval()
        print("✅ 模型加载成功")
        return model

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_category_names_from_json(json_path):
    """从COCO标注文件中读取类别名称（英文）"""
    import json
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 从categories中提取类别ID到名称的映射
        category_id_to_name = {}
        for cat in coco_data.get('categories', []):
            category_id_to_name[cat['id']] = cat['name']
        
        print(f"✅ 从标注文件加载了 {len(category_id_to_name)} 个类别名称")
        return category_id_to_name
    except Exception as e:
        print(f"⚠️  无法从标注文件加载类别名称: {e}")
        return {}

def create_validation_dataset():
    """创建验证数据集 - 使用PanoSegDataset（与训练配置一致）"""
    print("📊 创建验证数据集...")
    
    try:
        # 从训练配置中读取路径（使用绝对路径）
        root_dir = '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/'
        data_dir = root_dir + 'datas/'
        init_dir = root_dir + 'inits/'
        
        pano_data_root = data_dir + "pano/"
        # 使用验证集路径（如果存在），否则使用训练集路径
        val_data_path = pano_data_root + "annotations_val.json"
        val_image_folder = pano_data_root + "val/images"
        val_panseg_map_folder = pano_data_root + "val/panoptic_labels"
        
        # 如果验证集不存在，使用训练集
        if not os.path.exists(val_data_path):
            print("⚠️  验证集不存在，使用训练集进行验证")
            val_data_path = pano_data_root + "annotations_train.json"
            val_image_folder = pano_data_root + "train/images"
            val_panseg_map_folder = pano_data_root + "train/panoptic_labels"
        
        # 创建验证数据集 - 使用PanoSegDataset
        val_dataset = PanoSegDataset(
            data_path=val_data_path,
            image_folder=val_image_folder,
            panseg_map_folder=val_panseg_map_folder,
            task_name="genseg",
            data_name="pano_panoptic_genseg_val",
            pad_image_to_square=True,
            extra_image_processor=dict(
                type="xsam.dataset.processors.SamImageProcessor.from_pretrained",
                pretrained_model_name_or_path=init_dir + "sam-vit-large",
                trust_remote_code=True,
                ignore_index=0,
                size={"min_scale": 0.1, "max_scale": 2.0, "target_size": 1024},
                do_crop=True,
                crop_size={"height": 1024, "width": 1024},
            )
        )
        
        # 将标注文件路径保存到数据集对象，以便后续使用
        val_dataset._annotation_file_path = val_data_path
        
        print(f"✅ 验证数据集创建成功，包含 {len(val_dataset)} 个样本")
        return val_dataset
        
    except Exception as e:
        print(f"❌ 创建验证数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _change_color_brightness(color, brightness_factor):
    """
    根据brightness_factor调整颜色亮度，生成更亮或更暗的颜色。
    参考：/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main/xsam/xsam/utils/visualize.py
    
    Args:
        color: RGB颜色，可以是numpy数组或tuple，值范围[0, 1]
        brightness_factor (float): 亮度调整因子，范围[-1.0, 1.0]
            - 0: 不改变
            - 正值: 更亮
            - 负值: 更暗
    
    Returns:
        modified_color (tuple): RGB颜色元组，值范围[0.0, 1.0]
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    # 确保color是numpy数组格式，值范围[0, 1]
    if isinstance(color, np.ndarray):
        color = tuple(color)
    color = mplc.to_rgb(color)
    # 转换到HLS色彩空间
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    # 调整亮度
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    # 转换回RGB
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    return tuple(np.clip(modified_color, 0.0, 1.0))


def save_visualization_results(original_img, mask_logits, class_logits, batch_idx, vis_dir,
                               dataset_metadata=None, dataset=None, class_thresh=0.1, mask_thresh=0.1,
                               original_image_size=None, original_image_path=None, scaled_size=None):
    """
    Panoptic 风格可视化（跳过背景）
    - 每个类别固定颜色
    - 实例用黑边勾勒
    - 背景（cls_id == 40）不绘制（Pano数据集：0-40为类别，40为背景）
    - 直接从原始图片文件加载，显示完整原始尺寸
    - 只显示原图+mask覆盖，无空白区域
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    import torch.nn.functional as F

    # 1. 加载原始图片（未padding的真实尺寸）
    if original_image_path is not None and os.path.exists(original_image_path):
        print(f"🔍 从原始图片文件加载: {original_image_path}")
        pil_img = Image.open(original_image_path).convert('RGB')
        original_img_np = np.array(pil_img).astype(np.float32) / 255.0  # [H, W, 3], 值范围[0, 1]
        true_img_h, true_img_w = original_img_np.shape[:2]
        print(f"🔍 原始图片尺寸: {true_img_h}x{true_img_w} (HxW)")
    else:
        # 如果没有原始图片路径，使用处理后的图片tensor
        print(f"⚠️  未提供原始图片路径，使用处理后的图片tensor")
        img_tensor = original_img.permute(1, 2, 0).cpu().numpy()
        loaded_img_h, loaded_img_w = img_tensor.shape[:2]
        print(f"🔍 使用处理后的图片tensor: {loaded_img_h}x{loaded_img_w} (HxW)")
        
        # 归一化图片
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        
        # 确定原始图片的真实尺寸
        if original_image_size is not None:
            if isinstance(original_image_size, (list, tuple)) and len(original_image_size) == 2:
                true_img_h, true_img_w = int(original_image_size[0]), int(original_image_size[1])
                print(f"🔍 使用original_image_size: {true_img_h}x{true_img_w} (HxW)")
            else:
                true_img_h, true_img_w = loaded_img_h, loaded_img_w
                print(f"⚠️  original_image_size格式不正确，使用加载的图片尺寸")
        elif scaled_size is not None:
            if isinstance(scaled_size, (list, tuple)) and len(scaled_size) == 2:
                true_img_h, true_img_w = int(scaled_size[0]), int(scaled_size[1])
                print(f"🔍 使用scaled_size: {true_img_h}x{true_img_w} (HxW)")
            else:
                true_img_h, true_img_w = loaded_img_h, loaded_img_w
                print(f"⚠️  scaled_size格式不正确，使用加载的图片尺寸")
        else:
            true_img_h, true_img_w = loaded_img_h, loaded_img_w
            print(f"⚠️  未提供原始尺寸信息，假设无padding: {true_img_h}x{true_img_w}")
        
        # 如果tensor尺寸大于真实尺寸，说明tensor被padding了，需要从tensor中提取原图部分
        if loaded_img_h > true_img_h or loaded_img_w > true_img_w:
            print(f"🔍 从tensor中提取原图部分...")
            # expand2square是居中padding
            h_start = (loaded_img_h - true_img_h) // 2
            w_start = (loaded_img_w - true_img_w) // 2
            h_end = h_start + true_img_h
            w_end = w_start + true_img_w
            original_img_np = img_tensor[h_start:h_end, w_start:w_end, :]
            print(f"   提取后尺寸: {original_img_np.shape[0]}x{original_img_np.shape[1]}")
        else:
            original_img_np = img_tensor
    
    # 1.1 将原图padding到1024（匹配mask尺寸，使用expand2square的逻辑）
    # expand2square: 如果width > height，在上下padding；如果height > width，在左右padding
    target_size = 1024
    orig_h, orig_w = original_img_np.shape[:2]
    
    # 先按expand2square逻辑padding成正方形
    if orig_w > orig_h:
        # 宽度大于高度，在上下padding成正方形
        square_size = orig_w
        square_img_np = np.ones((square_size, square_size, 3), dtype=np.float32) * 0.5  # 灰色背景
        h_offset = (square_size - orig_h) // 2
        square_img_np[h_offset:h_offset + orig_h, :orig_w, :] = original_img_np
        crop_h_start = h_offset
        crop_w_start = 0
        crop_h_end = h_offset + orig_h
        crop_w_end = orig_w
    elif orig_h > orig_w:
        # 高度大于宽度，在左右padding成正方形
        square_size = orig_h
        square_img_np = np.ones((square_size, square_size, 3), dtype=np.float32) * 0.5  # 灰色背景
        w_offset = (square_size - orig_w) // 2
        square_img_np[:orig_h, w_offset:w_offset + orig_w, :] = original_img_np
        crop_h_start = 0
        crop_w_start = w_offset
        crop_h_end = orig_h
        crop_w_end = w_offset + orig_w
    else:
        # 已经是正方形
        square_size = orig_h
        square_img_np = original_img_np.copy()
        crop_h_start = 0
        crop_w_start = 0
        crop_h_end = orig_h
        crop_w_end = orig_w
    
    # 如果正方形尺寸不是1024，resize到1024
    if square_size != target_size:
        from PIL import Image
        square_pil = Image.fromarray((square_img_np * 255).astype(np.uint8))
        square_pil = square_pil.resize((target_size, target_size), Image.BILINEAR)
        padded_img_np = np.array(square_pil).astype(np.float32) / 255.0
        
        # 更新裁剪坐标（按比例缩放）
        scale = target_size / square_size
        crop_h_start = int(crop_h_start * scale)
        crop_w_start = int(crop_w_start * scale)
        crop_h_end = int(crop_h_end * scale)
        crop_w_end = int(crop_w_end * scale)
        print(f"🔍 原图padding成正方形{square_size}x{square_size}，然后resize到{target_size}x{target_size}")
    else:
        padded_img_np = square_img_np
        print(f"🔍 原图padding成正方形{target_size}x{target_size}")
    
    print(f"   原图在padding图中的位置: ({crop_h_start}, {crop_w_start}) 到 ({crop_h_end}, {crop_w_end})")
    
    final_img_h, final_img_w = padded_img_np.shape[:2]

    # 2. logits → 概率
    mask_probs = torch.sigmoid(mask_logits).cpu().numpy()   # [num_queries, H, W]
    mask_current_h, mask_current_w = mask_probs.shape[1], mask_probs.shape[2]
    print(f"🔍 输入mask尺寸: {mask_current_h}x{mask_current_w} (HxW)")
    print(f"🔍 padding后图片尺寸: {final_img_h}x{final_img_w} (HxW)")
    
    # 3. 将mask resize到1024（匹配padding后的图片尺寸）
    if mask_probs.shape[1] != target_size or mask_probs.shape[2] != target_size:
        print(f"⚠️  mask尺寸 {mask_probs.shape[1]}x{mask_probs.shape[2]} 与目标尺寸 {target_size}x{target_size} 不匹配，进行resize")
        mask_tensor = torch.from_numpy(mask_probs).unsqueeze(0)  # [1, num_queries, H, W]
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        mask_probs = mask_tensor.squeeze(0).numpy()
        print(f"🔍 mask resize后尺寸: {mask_probs.shape[1]}x{mask_probs.shape[2]} (HxW)")
    else:
        print(f"✅ mask尺寸已经是{target_size}x{target_size}")
    
    class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()  # [num_queries, num_classes]

    # 3. 类别 & 分数
    class_ids = class_probs.argmax(axis=-1)
    class_scores = class_probs.max(axis=-1)      # 分类置信度
    mask_scores = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)  # mask 最大值

    # 4. 阈值过滤
    keep = (class_scores > class_thresh) & (mask_scores > mask_thresh)
    mask_probs = mask_probs[keep]
    class_ids = class_ids[keep]
    class_scores = class_scores[keep]
    mask_scores = mask_scores[keep]

    if len(mask_probs) == 0:
        print(f"⚠️ 批次 {batch_idx+1} 没有高置信度的有效实例")
        return

    # 5. 类别映射
    # 注意：PanoSegDataset将原始类别ID映射到模型类别ID（0-39）
    # 但metadata中的dataset_classes使用原始类别ID作为key
    # 需要获取reverse_mapping来转换模型类别ID -> 原始类别ID
    id2name, id2color = {}, {}
    reverse_mapping = None  # 模型类别ID -> 原始类别ID的映射
    
    if dataset_metadata is not None:
        if hasattr(dataset_metadata, "dataset_classes"):
            id2name = dataset_metadata.dataset_classes
        if hasattr(dataset_metadata, "dataset_colors"):
            id2color = dataset_metadata.dataset_colors
    
    # 尝试从数据集获取reverse_mapping（PanoSegDataset特有）
    # 如果数据集有reverse_mapping属性，使用它来转换类别ID
    if dataset is not None and hasattr(dataset, "reverse_mapping"):
        reverse_mapping = dataset.reverse_mapping
    elif hasattr(dataset_metadata, "_dataset") and hasattr(dataset_metadata._dataset, "reverse_mapping"):
        reverse_mapping = dataset_metadata._dataset.reverse_mapping
    elif hasattr(dataset_metadata, "reverse_mapping"):
        reverse_mapping = dataset_metadata.reverse_mapping
    
    # 如果没有reverse_mapping，尝试从dataset_id_to_contiguous_id构建反向映射
    if reverse_mapping is None and hasattr(dataset_metadata, "dataset_id_to_contiguous_id"):
        # 构建反向映射：contiguous_id -> dataset_id
        contiguous_to_dataset = {v: k for k, v in dataset_metadata.dataset_id_to_contiguous_id.items()}
        reverse_mapping = contiguous_to_dataset

    # 固定调色板（20 种循环使用）
    fixed_palette = (plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(np.uint8)

    def get_color(original_cat_id: int):
        """固定类别颜色，使用原始类别ID"""
        if original_cat_id in id2color:
            return np.array(id2color[original_cat_id]) / 255.0
        return fixed_palette[original_cat_id % len(fixed_palette)] / 255.0
    
    # 从标注文件中读取英文类别名称
    english_category_names = {}
    if dataset is not None and hasattr(dataset, '_annotation_file_path'):
        english_category_names = load_category_names_from_json(dataset._annotation_file_path)
        print(f"🔍 从标注文件加载了 {len(english_category_names)} 个类别名称")
        if len(english_category_names) > 0:
            print(f"   示例类别名称: {list(english_category_names.items())[:5]}")
    
    def get_class_name(model_cls_id: int):
        """获取类别名称（英文），从COCO标注文件中读取"""
        # 如果有reverse_mapping，将模型类别ID转换为原始类别ID
        if reverse_mapping is not None and model_cls_id in reverse_mapping:
            original_cat_id = reverse_mapping[model_cls_id]
            # 优先从标注文件中获取英文名称
            if original_cat_id in english_category_names:
                name = english_category_names[original_cat_id]
                # 确保返回的是字符串
                return str(name) if name is not None else f"class_{model_cls_id}"
            # 如果标注文件中没有，尝试从metadata获取
            if original_cat_id in id2name:
                name = id2name[original_cat_id]
                return str(name) if name is not None else f"class_{model_cls_id}"
        else:
            # 如果没有reverse_mapping，直接使用模型类别ID查找
            if model_cls_id in english_category_names:
                name = english_category_names[model_cls_id]
                return str(name) if name is not None else f"class_{model_cls_id}"
            if model_cls_id in id2name:
                name = id2name[model_cls_id]
                return str(name) if name is not None else f"class_{model_cls_id}"
        
        # 如果找不到类别名称，使用类别ID
        return f"class_{model_cls_id}"

    # 6. 验证mask和图片尺寸是否匹配（此时图片已经被padding到1024）
    H, W = padded_img_np.shape[:2]
    if mask_probs.shape[1] != H or mask_probs.shape[2] != W:
        print(f"❌ 错误：mask尺寸 {mask_probs.shape[1]}x{mask_probs.shape[2]} 与图片尺寸 {H}x{W} 不匹配！")
        print(f"   强制调整mask尺寸以匹配图片...")
        mask_tensor = torch.from_numpy(mask_probs).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        mask_probs = mask_tensor.squeeze(0).numpy()
        print(f"   调整后mask尺寸: {mask_probs.shape[1]}x{mask_probs.shape[2]}")
    
    # 7. 创建可视化结果：将mask盖到padding后的原图上
    # 直接使用numpy数组创建结果，不经过matplotlib，避免添加边框和空白
    result_img = padded_img_np.copy()  # [H, W, 3], 值范围[0, 1]，1024x1024
    
    # 用于存储需要添加文字标签的信息（mask中心点和类别名称，坐标是1024x1024中的）
    text_labels = []  # [(center_y, center_x, class_name, color)]
    
    # 绘制实例（跳过背景）
    drawn_classes = set()
    for i in range(len(mask_probs)):
        mask = mask_probs[i] > 0.5
        cls_id = int(class_ids[i])

        # 跳过背景类（索引40，Pano数据集背景类）
        # 注意：如果num_labels=41，那么0-39是实际类别，40是背景类
        if cls_id >= 40:  # 40及以上都是背景类
            continue
        if mask.sum() == 0:
            continue

        # 将模型类别ID转换为原始类别ID（用于查找类别名称和颜色）
        if reverse_mapping is not None and cls_id in reverse_mapping:
            original_cat_id = reverse_mapping[cls_id]
        else:
            # 如果没有reverse_mapping，假设模型类别ID就是原始类别ID（向后兼容）
            original_cat_id = cls_id
        
        color = get_color(original_cat_id)  # [R, G, B], 值范围[0, 1]
        
        # 获取类别名称
        class_name = get_class_name(cls_id)

        # 在padding后的原图上叠加mask颜色（半透明）
        alpha = 0.5
        result_img[mask] = result_img[mask] * (1 - alpha) + np.array(color) * alpha
        
        # 添加黑边勾勒（使用形态学操作找到边缘）
        from scipy import ndimage
        # 膨胀mask，然后减去原始mask得到边缘
        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        edge_mask = dilated_mask & (~mask)
        # 在边缘处绘制黑色
        result_img[edge_mask] = [0, 0, 0]
        
        # 计算mask的中心点（用于放置文字标签，坐标在1024x1024中）
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            text_labels.append((center_y, center_x, class_name, color))
    
    # 7.1 截取原图尺寸部分（只保留原图有效区域，去掉padding的无效部分）
    print(f"🔍 截取原图有效区域...")
    print(f"   截取前尺寸: {result_img.shape[0]}x{result_img.shape[1]}")
    
    # 保存截取后的原图部分（没有mask的，用于raw_save_path）
    original_cropped = padded_img_np[crop_h_start:crop_h_end, crop_w_start:crop_w_end, :]
    
    # 截取result_img（有mask的）
    result_img = result_img[crop_h_start:crop_h_end, crop_w_start:crop_w_end, :]
    # 更新文字标签坐标（减去截取偏移，转换到原图坐标系）
    text_labels = [(y - crop_h_start, x - crop_w_start, name, color) 
                   for y, x, name, color in text_labels
                   if crop_h_start <= y < crop_h_end and crop_w_start <= x < crop_w_end]
    final_img_h, final_img_w = result_img.shape[:2]
    print(f"   截取后尺寸: {final_img_h}x{final_img_w} (原图真实尺寸)")
    
    # 7.2 添加文字标签（使用PIL的ImageDraw）
    # 先将numpy数组转换为PIL Image
    result_img_uint8 = (np.clip(result_img, 0, 1) * 255).astype(np.uint8)
    result_pil = Image.fromarray(result_img_uint8)
    
    # 使用ImageDraw添加文字
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result_pil)
    
    # 尝试加载字体（支持中文）
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            # 尝试其他常见字体路径
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
        except:
            # 如果都失败，使用默认字体
            font = ImageFont.load_default()
            print("⚠️  无法加载系统字体，使用默认字体（可能不支持中文）")
    
    # 为每个mask添加文字标签
    # 参考：/mnt_llm_A100_V1/shui/LAE/XSAM-public/RS-Xsam-main/xsam/xsam/utils/visualize.py
    # 使用更亮的mask颜色作为文字颜色，而不是简单的黑白对比色
    img_width, img_height = result_pil.size
    for center_y, center_x, class_name, color in text_labels:
        # 确保坐标在图片范围内
        center_x = max(0, min(center_x, img_width - 1))
        center_y = max(0, min(center_y, img_height - 1))
        
        # 使用_change_color_brightness生成更亮的颜色用于文字标签
        # brightness_factor=0.7 表示将颜色调亮70%，确保文字清晰可见
        lighter_color = _change_color_brightness(color, brightness_factor=0.7)
        # 转换为PIL需要的格式（0-255范围的整数元组）
        text_color = tuple(int(c * 255) for c in lighter_color)
        
        # 添加文字背景（半透明黑色矩形，提高可读性）
        try:
            # 获取文字尺寸
            bbox = draw.textbbox((0, 0), class_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算背景矩形位置
            padding = 4
            bg_x1 = max(0, center_x - text_width // 2 - padding)
            bg_y1 = max(0, center_y - text_height // 2 - padding)
            bg_x2 = min(img_width, center_x + text_width // 2 + padding)
            bg_y2 = min(img_height, center_y + text_height // 2 + padding)
            
            # 确保背景矩形有效
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                # 绘制半透明背景（降低不透明度，让文字颜色更突出）
                bg_img = Image.new('RGBA', (bg_x2 - bg_x1, bg_y2 - bg_y1), (0, 0, 0, 150))
                result_pil.paste(bg_img, (bg_x1, bg_y1), bg_img)
            
            # 绘制文字
            text_x = max(0, center_x - text_width // 2)
            text_y = max(0, center_y - text_height // 2)
            draw.text((text_x, text_y), class_name, fill=text_color, font=font)
        except Exception as e:
            # 如果文字绘制失败，至少尝试绘制简单文字
            try:
                draw.text((center_x, center_y), class_name, fill=text_color)
            except:
                print(f"⚠️  无法在位置 ({center_x}, {center_y}) 绘制文字: {class_name}, 错误: {e}")
    
    # 将PIL Image转换回numpy数组
    result_img = np.array(result_pil).astype(np.float32) / 255.0

    # 8. 保存结果
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_save_path = vis_dir / f"batch_{batch_idx + 1}_panoptic.png"
    raw_save_path = vis_dir / f"batch_{batch_idx + 1}_raw.png"

    # 保存分割结果（原图+mask覆盖+文字标签）
    # result_img从PIL转换回来是float32格式（0-1范围），需要转换为uint8
    if result_img.dtype == np.uint8:
        result_to_save = result_img
    else:
        result_to_save = np.clip(result_img, 0, 1)
        result_to_save = (result_to_save * 255).astype(np.uint8)
    
    # 使用PIL保存，避免matplotlib添加边框
    result_pil_final = Image.fromarray(result_to_save)
    result_pil_final.save(mask_save_path)

    # 保存原图（保存截取后的原图，没有mask的）
    # 确保值在[0,1]范围内
    img_to_save = np.clip(original_cropped, 0, 1)
    # 转换为uint8
    img_to_save = (img_to_save * 255).astype(np.uint8)
    # 使用PIL保存，避免matplotlib添加边框
    img_pil = Image.fromarray(img_to_save)
    img_pil.save(raw_save_path)

    print(f"✅ 原图保存到 {raw_save_path}")
    print(f"✅ 分割图保存到 {mask_save_path}")


def run_validation(model, dataset, vis_output_dir, device='cuda', max_images=50):
    """
    运行验证 - 使用完整的X-SAM模型进行推理和可视化
    """
    print("🔍 开始验证流程...")
    print(f"🔧 使用设备: {device}")
    print(f"🔧 使用数据类型: torch.bfloat16 (与训练时一致)")
    print(f"📁 可视化结果保存路径: {vis_output_dir}")
    
    # 创建数据加载器
    def custom_collate_fn(batch):
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None
        return xsam_collate_fn(valid_batch)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 可视化结果保存目录（使用用户指定的路径）
    vis_dir = Path(vis_output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        "total_samples": 0,
        "processed_samples": 0,
        "successful_samples": 0,
        "failed_samples": 0,
        "errors": []
    }
    
    # 打印 metadata 信息
    dataset_metadata = getattr(dataset, "_metadata", None)
    print("🔍 dataset._metadata =", dataset_metadata)
    if hasattr(dataset_metadata, "categories"):
        cats = dataset_metadata.categories
        print("🔍 categories 类型:", type(cats))
        if isinstance(cats, (list, tuple)):
            print("🔍 categories 示例:", cats[:3])
        elif isinstance(cats, dict):
            print("🔍 categories keys 示例:", list(cats.keys())[:5])
    
    print(f"🔍 开始处理图片，限制数量: {max_images}")
    
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= max_images:
            print(f"🔧 已达到最大处理数量 {max_images}，停止处理")
            break
            
        if data is None:
            print(f"⚠️  批次 {batch_idx + 1} 数据为None，跳过")
            results_summary["failed_samples"] += 1
            continue
            
        try:
            print(f"\n{'='*60}")
            print(f"🔍 处理批次 {batch_idx + 1}/{min(max_images, len(dataloader))}")
            print(f"{'='*60}")
            
            results_summary["total_samples"] += 1
            
            # 获取图像数据
            # xsam_collate_fn 返回格式: {"data_dict": {...}, "data_samples": ...}
            img = None
            if isinstance(data, dict):
                # 首先检查是否有 data_dict 键（xsam_collate_fn 的标准格式）
                if 'data_dict' in data:
                    data_dict = data['data_dict']
                    # 在 data_dict 中查找图像数据
                    for key in ['seg_pixel_values', 'pixel_values']:
                        if key in data_dict and isinstance(data_dict[key], torch.Tensor) and len(data_dict[key].shape) == 4:
                            img = data_dict[key]
                            break
                else:
                    # 如果没有 data_dict，直接在当前字典中查找（向后兼容）
                    for key in ['seg_pixel_values', 'pixel_values', 'img', 'image', 'inputs']:
                        if key in data and isinstance(data[key], torch.Tensor) and len(data[key].shape) == 4:
                            img = data[key]
                            break
            
            if img is None:
                print(f"❌ 批次 {batch_idx + 1} 无法找到图像数据")
                if isinstance(data, dict):
                    if 'data_dict' in data:
                        print(f"   data_dict 键: {list(data['data_dict'].keys())}")
                    print(f"   数据键: {list(data.keys())}")
                else:
                    print(f"   数据类型: {type(data)}")
                results_summary["failed_samples"] += 1
                continue
            
            # 确保图像格式正确
            if len(img.shape) != 4 or img.shape[1] != 3:
                print(f"⚠️  图像数据格式不正确: shape={img.shape}")
                results_summary["failed_samples"] += 1
                continue
            
            # 移动到正确设备并转换为正确数据类型
            img = img.to(device)
            if img.dtype != torch.bfloat16:
                img = img.to(torch.bfloat16)
            
            print(f"🔍 使用图像数据: shape={img.shape}, dtype={img.dtype}, device={img.device}")
            
            # 使用X-SAM模型进行推理
            print("🔍 使用X-SAM模型进行推理...")
            
            try:
                # 准备输入数据 - 使用与训练时相同的格式
                # 如果原始数据有 data_dict，使用它；否则创建新的
                if isinstance(data, dict) and 'data_dict' in data:
                    # 创建 data_dict 的副本并更新图像数据
                    data_dict = dict(data['data_dict'])
                    # 确保图像数据在正确的设备和数据类型上
                    if 'seg_pixel_values' in data_dict:
                        data_dict['seg_pixel_values'] = img
                    elif 'pixel_values' in data_dict:
                        data_dict['pixel_values'] = img
                else:
                    # 如果没有 data_dict，创建新的
                    data_dict = {
                        'seg_pixel_values': img,  # 使用训练时的数据格式
                    }
                
                # 调用模型的forward方法进行推理
                print("🔍 开始调用 model.forward 进行推理...")
                with torch.no_grad():
                    llm_outputs, seg_outputs = model.forward(data_dict, mode="tensor")
                
                print(f"✅ 推理成功，LLM输出类型: {type(llm_outputs)}, 分割输出类型: {type(seg_outputs)}")
                
                # 处理输出结果
                if seg_outputs is not None and hasattr(seg_outputs, 'class_queries_logits') and hasattr(seg_outputs, 'masks_queries_logits'):
                    print("🔍 处理分割结果...")
                    
                    class_logits = seg_outputs.class_queries_logits # [1, num_queries, num_classes]
                    mask_logits = seg_outputs.masks_queries_logits   # [1, num_queries, H, W]
                    
                    # 获取图片处理信息 - 从data_dict中获取scaled_size和image_size
                    original_image_size = None
                    scaled_size = None
                    original_image_path = None
                    
                    # 方法1: 从data_dict中获取（最准确）
                    if isinstance(data, dict) and 'data_dict' in data:
                        data_dict = data['data_dict']
                        if 'image_size' in data_dict:
                            original_image_size = data_dict['image_size']
                            if isinstance(original_image_size, (list, tuple)) and len(original_image_size) == 2:
                                # 确保是 (height, width) 格式
                                if original_image_size[0] > original_image_size[1]:  # 可能是 (width, height)
                                    original_image_size = (original_image_size[1], original_image_size[0])
                                print(f"🔍 从data_dict获取original_image_size: {original_image_size}")
                        if 'scaled_size' in data_dict:
                            scaled_size = data_dict['scaled_size']
                            if isinstance(scaled_size, (list, tuple)) and len(scaled_size) == 2:
                                # 确保是 (height, width) 格式
                                if scaled_size[0] > scaled_size[1]:  # 可能是 (width, height)
                                    scaled_size = (scaled_size[1], scaled_size[0])
                                print(f"🔍 从data_dict获取scaled_size: {scaled_size}")
                    
                    # 方法2: 从data_samples中获取
                    if original_image_size is None and isinstance(data, dict) and 'data_samples' in data:
                        data_samples = data['data_samples']
                        if hasattr(data_samples, 'image_sizes') and len(data_samples.image_sizes) > 0:
                            original_image_size = data_samples.image_sizes[0]
                            print(f"🔍 从data_samples.image_sizes获取: {original_image_size}")
                    
                    # 方法3: 从原始图片文件读取（作为备用）
                    if original_image_size is None:
                        try:
                            dataset_item = dataset[batch_idx]
                            if isinstance(dataset_item, dict) and 'image_file' in dataset_item:
                                image_file = dataset_item['image_file']
                                root_dir = '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/'
                                data_dir = root_dir + 'datas/'
                                pano_data_root = data_dir + "pano/"
                                val_image_folder = pano_data_root + "val/images"
                                if not os.path.exists(val_image_folder):
                                    val_image_folder = pano_data_root + "train/images"
                                image_path = os.path.join(val_image_folder, image_file)
                                if os.path.exists(image_path):
                                    original_image_path = image_path
                                    from PIL import Image
                                    pil_img = Image.open(image_path).convert('RGB')
                                    original_image_size = (pil_img.size[1], pil_img.size[0])  # (height, width)
                                    print(f"🔍 从图片文件获取: {image_file} -> {original_image_size} (HxW)")
                        except Exception as e:
                            print(f"⚠️  无法从图片文件获取原始尺寸: {e}")
                    
                    # 打印调试信息
                    print(f"🔍 原始图片尺寸: {original_image_size}")
                    print(f"🔍 缩放后尺寸: {scaled_size}")
                    print(f"🔍 当前图片tensor尺寸: {img.shape} (BCHW)")
                    if len(img.shape) == 4:
                        current_h, current_w = img.shape[2], img.shape[3]
                        print(f"🔍 当前图片尺寸: {current_h}x{current_w} (HxW)")
                    
                    # 可视化前转换：GPU(bfloat16/float32) → CPU(float32)
                    orig_img_vis = img[0].float().cpu()
                    mask_logits_vis = mask_logits[0].float().cpu()
                    class_logits_vis = class_logits[0].float().cpu()

                    save_visualization_results(
                        orig_img_vis, mask_logits_vis, class_logits_vis, 
                        batch_idx, vis_dir, 
                        dataset_metadata=getattr(dataset, '_metadata', None),
                        dataset=dataset,  # 传递数据集对象以访问reverse_mapping和标注文件路径
                        original_image_size=original_image_size,  # 传递原始图片大小
                        original_image_path=original_image_path,  # 传递原始图片路径
                        scaled_size=scaled_size  # 传递缩放后尺寸（crop前的尺寸）
                    )

                    print(f"✅ 批次 {batch_idx + 1} 处理完成")
                    results_summary["successful_samples"] += 1
                else:
                    print(f"⚠️  分割输出格式异常: {type(seg_outputs)}")
                    if seg_outputs is not None:
                        print(f"   分割输出属性: {dir(seg_outputs)}")
                    results_summary["failed_samples"] += 1
                    
            except Exception as e:
                print(f"❌ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                results_summary["failed_samples"] += 1
                continue
                
            results_summary["processed_samples"] += 1
            
        except Exception as e:
            print(f"❌ 处理批次 {batch_idx + 1} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results_summary["failed_samples"] += 1
            continue
    
    print("\n" + "="*60)
    print("🎯 验证完成总结")
    print("="*60)
    print(f"📊 总样本数: {results_summary['total_samples']}")
    print(f"✅ 成功处理: {results_summary['successful_samples']}")
    print(f"❌ 处理失败: {results_summary['failed_samples']}")
    print(f"📁 可视化结果保存在: {vis_dir}")
    
    return results_summary


def main(checkpoint_path, vis_output_dir, device='cuda', max_images=50):
    """主函数"""
    try:
        # 检查设备可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA不可用，切换到CPU")
            device = 'cpu'
        
        print(f"🔧 使用设备: {device}")
        
        # 加载训练好的模型
        model = load_trained_model(checkpoint_path, device)
        if not model:
            print("❌ 模型加载失败")
            return
        
        # 创建验证数据集
        dataset = create_validation_dataset()
        if not dataset:
            print("❌ 验证数据集创建失败")
            return
        
        # 运行验证
        results = run_validation(model, dataset, vis_output_dir, device, max_images)
        
        return results
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='第一阶段训练可视化验证脚本')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to checkpoint file or directory')
    parser.add_argument('--vis_output_dir', type=str, required=True,
                       help='Path to save visualization results (自定义可视化结果路径)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to process (default: 50)')
    
    args = parser.parse_args()
    
    try:
        results = main(args.checkpoint_path, args.vis_output_dir, args.device, args.max_images)
        
        if results:
            print(f"\n🎉 验证成功完成!")
            print(f"📁 可视化结果保存在: {args.vis_output_dir}")
        else:
            print("❌ 验证过程中出现错误")
            
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

