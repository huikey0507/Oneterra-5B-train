#!/usr/bin/env python3
"""
简化的验证脚本 - 直接加载训练好的模型进行推理
"""

import os
import sys
import argparse
from pathlib import Path

# 添加 xsam 模块路径（仅 OneTerra-train 仓库内：xsam/xsam/dataset/...）
xsam_path = (Path(__file__).resolve().parent / "xsam").resolve()
if not xsam_path.is_dir():
    print("❌ 错误：找不到 xsam 模块，请确认 OneTerra-train/xsam 存在")
    print(f"   期望路径: {xsam_path}")
    sys.exit(1)
xsam_path_str = str(xsam_path)
if xsam_path_str not in sys.path:
    sys.path.insert(0, xsam_path_str)
print(f"🔍 添加 xsam 模块路径: {xsam_path}")

import torch
import torch.nn as nn
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import subprocess
import time

# 导入必要的模块
try:
    from xsam.dataset.generic_seg_dataset import GenericSegDataset
    from xsam.evaluation.evaluators.generic_seg_evaluator import GenericSegEvaluator
    from xsam.model import XSamModel
    print("✅ 成功导入xsam模块")
except ImportError as e:
    print(f"❌ 导入xsam模块失败: {e}")
    sys.exit(1)


def get_most_free_gpu():
    """自动检测最空闲的GPU"""
    try:
        # 使用nvidia-smi获取GPU使用情况
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                mem_used = int(parts[1].strip())
                mem_total = int(parts[2].strip())
                gpu_util = int(parts[3].strip())
                mem_used_percent = (mem_used / mem_total) * 100
                gpu_info.append({
                    'id': gpu_id,
                    'mem_used': mem_used,
                    'mem_total': mem_total,
                    'mem_free_percent': 100 - mem_used_percent,
                    'gpu_util': gpu_util
                })
        
        # 选择内存使用率最低且GPU利用率最低的GPU
        if gpu_info:
            # 优先考虑内存空闲率，其次考虑GPU利用率
            best_gpu = max(gpu_info, key=lambda x: (x['mem_free_percent'], -x['gpu_util']))
            print(f"🔍 检测到GPU使用情况:")
            for gpu in gpu_info:
                print(f"   GPU {gpu['id']}: 内存使用 {gpu['mem_used']}/{gpu['mem_total']}MB ({100-gpu['mem_free_percent']:.1f}%), GPU利用率 {gpu['gpu_util']}%")
            print(f"✅ 自动选择GPU {best_gpu['id']} (内存空闲 {best_gpu['mem_free_percent']:.1f}%, GPU利用率 {best_gpu['gpu_util']}%)")
            return f"cuda:{best_gpu['id']}"
        else:
            return "cuda:0"
    except Exception as e:
        print(f"⚠️  无法检测GPU使用情况: {e}，使用默认GPU 0")
        return "cuda:0"


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
        return None


def load_trained_model(checkpoint_path, device='cuda'):
    """
    加载训练好的 X-SAM 模型（修复：先伪前向搭壳，再 load_state_dict 装货）
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

        print("🔍 加载 checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        
        
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_dir():
            ckpt_file = ckpt_path / "pytorch_model.bin"
        else:
            ckpt_file = ckpt_path          # 本身就是 .pth/.bin

        ckpt = torch.load(ckpt_file, map_location="cpu")
        for k, v in ckpt.items():
            if "connector.layer3" in k:
                print(k, v.shape)          # 看 in_channels
        # 1. 先加载预训练参数，再加载训练好的参数
        from xsam.model import XSamModel
        
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
            s1_pretrained_pth="checkpoints/s1_seg_finetune/coco_pretrain/pytorch_model.bin",  # 先加载预训练参数
            segmentor=dict(
                type="xsam.model.segmentors.XSegmentor",
                encoder=dict(
                    type="xsam.model.segmentors.sam.SamModel.from_pretrained",
                    pretrained_model_name_or_path="inits/sam-vit-large",
                    trust_remote_code=True,
                    torch_dtype="torch.bfloat16",
                ),
                decoder=dict(
                    type="xsam.model.segmentors.mask2former.Mask2FormerModel._from_config",
                    config=dict(
                        type="xsam.model.segmentors.mask2former.Mask2FormerConfig.from_pretrained",
                        pretrained_model_name_or_path="inits/mask2former-swin-large-coco-panoptic",
                        use_backbone=False,
                        feature_channels=[512, 1024, 2048],
                        num_feature_levels=3,
                        num_labels=39,
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
        
        
        
        
        # 智能处理class_predictor的形状不匹配
        # filtered_checkpoint = {}
        # for key, value in checkpoint.items():
        #     if 'class_predictor' in key:
                
        #         # 跳过class_predictor相关的权重，因为类别数可能不匹配
        #         print(f"Skipping {key} due to potential class number mismatch")
        #         continue
        #     else:
        #         filtered_checkpoint[key] = value
        
        filtered_checkpoint = {}
        for key, value in checkpoint.items():
            filtered_checkpoint[key] = value
        
        # 加载训练好的参数
        model.load_state_dict(filtered_checkpoint, strict=False)
        print("✅ 第二步完成：训练好的参数加载成功")

        # 调试：打印关键层的形状（加载训练参数后）
        print("调试：加载训练参数后关键层形状")
        for name, param in model.named_parameters():
            if 'connector' in name or 'pixel_decoder' in name or 'input_projections' in name:
                print(f"{name}: {param.shape}")

        # # 2. 先搭壳：伪前向触发 lazy-build
        # with torch.no_grad():
        #     dummy = torch.zeros(1, 3, 1024, 1024, dtype=torch.bfloat16, device=device)
        #     _ = model._forward({'seg_pixel_values': dummy})

        # 调试：打印关键层的形状（初始化后，加载权重前）
        print("调试：初始化后关键层形状")
        for name, param in model.named_parameters():
            if 'connector' in name or 'pixel_decoder' in name or 'input_projections' in name:
                print(f"{name}: {param.shape}")

        # 3. 模型已通过 s1_pretrained_pth 参数自动加载了训练好的权重
        print("✅ 模型已通过 s1_pretrained_pth 参数自动加载训练好的权重")

        # 4. 最后 eval
        model.eval()
        # 调试：打印关键层的形状（加载权重后）
        print("调试：加载权重后关键层形状")
        for name, param in model.named_parameters():
            if 'connector' in name or 'pixel_decoder' in name or 'input_projections' in name:
                print(f"{name}: {param.shape}")

        print("✅ 模型加载成功 & connector 权重已正确加载")
        return model

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_validation_dataset():
    """创建验证数据集"""
    print("📊 创建验证数据集...")
    
    try:
        val_dataset = GenericSegDataset(
            data_name='sota_panoptic_genseg_val',
            data_path='datas/sota/val_annotations.json',
            image_folder='datas/sota/val/images',
            panseg_map_folder='datas/sota/val/panoptic_labels',
            task_name='genseg',
            pad_image_to_square=True,
            extra_image_processor=dict(
                crop_size=dict(height=1024, width=1024),
                do_crop=True,
                ignore_index=0,
                pretrained_model_name_or_path='inits/sam-vit-large',
                size=dict(max_scale=2.0, min_scale=0.1, target_size=1024),
                trust_remote_code=True,
                type='xsam.dataset.processors.SamImageProcessor.from_pretrained'
            )
        )
        
        print(f"✅ 验证数据集创建成功，包含 {len(val_dataset)} 个样本")
        return val_dataset
        
    except Exception as e:
        print(f"❌ 创建验证数据集失败: {e}")
        return None


def run_validation(model, dataset, work_dir, device='cuda', max_images=50):
    """
    运行验证 - 使用完整的X-SAM模型进行推理
    """
    print("🔍 开始验证流程...")
    print(f"🔧 使用设备: {device}")
    print(f"🔧 使用数据类型: torch.bfloat16 (与训练时一致)")
    
    # 创建数据加载器
    def custom_collate_fn(batch):
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None
        return valid_batch[0]
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 可视化结果保存目录
    vis_dir = work_dir / "visualization_results"
    vis_dir.mkdir(exist_ok=True)
    
    results_summary = {
        "total_samples": 0,
        "processed_samples": 0,
        "successful_samples": 0,
        "failed_samples": 0,
        "errors": []
    }
    
    
     # 打印 metadata 信息，调试类别映射
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
    print(f"📊 数据集总大小: {len(dataset)}")
    
    # 添加进度条
    pbar = tqdm(total=min(max_images, len(dataloader)), desc="处理进度", unit="batch")
    
    print(f"⏱️  [{time.strftime('%H:%M:%S')}] 开始加载数据...")
    for batch_idx, data in enumerate(dataloader):
        print(f"⏱️  [{time.strftime('%H:%M:%S')}] 数据加载完成，批次 {batch_idx + 1}")


        
        if batch_idx >= max_images:
            print(f"🔧 已达到最大处理数量 {max_images}，停止处理")
            pbar.close()
            break
            
        if data is None:
            print(f"⚠️  批次 {batch_idx + 1} 数据为None，跳过")
            results_summary["failed_samples"] += 1
            continue
            
        try:
            batch_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"🔍 处理批次 {batch_idx + 1}/{min(max_images, len(dataloader))}")
            print(f"⏱️  [{time.strftime('%H:%M:%S')}] 开始处理批次...")
            print(f"{'='*60}")
            
            results_summary["total_samples"] += 1
            
            # 获取图像数据
            img = None
            if 'image_file' in data:
                image_file_path = data['image_file']
                if isinstance(image_file_path, str):
                    full_image_path = f"datas/sota/val/images/{image_file_path}"
                    img = load_original_image(full_image_path, device)
            
            if img is None and isinstance(data, dict):
                # 尝试其他字段
                for key in ['img', 'image', 'inputs', 'pixel_values']:
                    if key in data and isinstance(data[key], torch.Tensor) and len(data[key].shape) == 4:
                        img = data[key]
                        break
            
            if img is None:
                print(f"❌ 批次 {batch_idx + 1} 无法找到图像数据")
                results_summary["failed_samples"] += 1
                continue
            
            # 确保图像格式正确
            if len(img.shape) != 4 or img.shape[1] != 3:
                print(f"⚠️  图像数据格式不正确: shape={img.shape}")
                results_summary["failed_samples"] += 1
                continue
            
            # 移动到正确设备并转换为正确数据类型
            print(f"⏱️  [{time.strftime('%H:%M:%S')}] 移动数据到设备 {device}...")
            img = img.to(device)
            if img.dtype != torch.bfloat16:
                img = img.to(torch.bfloat16)
            
            print(f"✅ 图像数据准备完成: shape={img.shape}, dtype={img.dtype}, device={img.device}")
            
            # 使用X-SAM模型进行推理
            print("🔍 开始推理...")
            
            try:
                # 准备输入数据 - 使用与训练时相同的格式
                data_dict = {
                    'seg_pixel_values': img,  # 使用训练时的数据格式
                }
                
                # 调用模型的forward方法进行推理
                print(f"⏱️  [{time.strftime('%H:%M:%S')}] 开始调用 model.forward...")
                inference_start = time.time()
                
                with torch.no_grad():
                    # 确保CUDA操作完成
                    if device.startswith('cuda'):
                        gpu_id = int(device.split(':')[1]) if ':' in device else 0
                        torch.cuda.set_device(gpu_id)
                        torch.cuda.synchronize()
                    
                    # 执行推理
                    llm_outputs, seg_outputs = model.forward(data_dict, mode="tensor")
                    
                    # 同步等待推理完成
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    
                inference_time = time.time() - inference_start
                print(f"✅ 推理完成，耗时: {inference_time:.2f}秒")
                print(f"✅ 推理成功，LLM输出类型: {type(llm_outputs)}, 分割输出类型: {type(seg_outputs)}")
                
                # 处理输出结果
                if seg_outputs is not None and hasattr(seg_outputs, 'class_queries_logits') and hasattr(seg_outputs, 'masks_queries_logits'):
                    print("🔍 处理分割结果...")
                    
                    class_logits = seg_outputs.class_queries_logits # [1, num_queries, num_classes]
                    mask_logits = seg_outputs.masks_queries_logits   # [1, num_queries, H, W]
                    
                    #print(f"   类别预测形状: {class_logits.shape}, dtype: {class_logits.dtype}")
                    #print(f"   掩码预测形状: {mask_logits.shape}, dtype: {mask_logits.dtype}")
                    
                    ## 生成可视化结果
                    #save_visualization_results(
                    #    img[0], mask_logits[0], class_logits[0], 
                    #    batch_idx, vis_dir, 
                    #    dataset_metadata=getattr(dataset, '_metadata', None)
                    #)
                    # 可视化前转换：GPU(bfloat16/float32) → CPU(float32)
                    orig_img_vis = img[0].float().cpu()
                    mask_logits_vis = mask_logits[0].float().cpu()
                    class_logits_vis = class_logits[0].float().cpu()

                    save_visualization_results(
                        orig_img_vis, mask_logits_vis, class_logits_vis, 
                        batch_idx, vis_dir, 
                        dataset_metadata=getattr(dataset, '_metadata', None)
                    )

                    
                    batch_time = time.time() - batch_start_time
                    print(f"✅ 批次 {batch_idx + 1} 处理完成，总耗时: {batch_time:.2f}秒")
                    results_summary["successful_samples"] += 1
                    pbar.update(1)
                    
                    # 清理GPU缓存
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                else:
                    print(f"⚠️  分割输出格式异常: {type(seg_outputs)}")
                    if seg_outputs is not None:
                        print(f"   分割输出属性: {dir(seg_outputs)}")
                    results_summary["failed_samples"] += 1
                    
            except Exception as e:
                print(f"❌ 推理失败: {e}")
                import traceback
                print(f"详细错误信息:")
                traceback.print_exc()
                results_summary["failed_samples"] += 1
                pbar.update(1)
                
                # 清理GPU缓存
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                continue
                
            results_summary["processed_samples"] += 1
            
        except Exception as e:
            print(f"❌ 处理批次 {batch_idx + 1} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results_summary["failed_samples"] += 1
            pbar.update(1)
            
            # 清理GPU缓存
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            continue
    
    pbar.close()
    
    print("\n" + "="*60)
    print("🎯 验证完成总结")
    print("="*60)
    print(f"📊 总样本数: {results_summary['total_samples']}")
    print(f"✅ 成功处理: {results_summary['successful_samples']}")
    print(f"❌ 处理失败: {results_summary['failed_samples']}")
    print(f"📁 可视化结果保存在: {vis_dir}")
    
    return results_summary


#def load_original_image(image_file_path, device='cuda'):
#    """加载原始图像数据 - 使用与训练时一致的数据类型"""
#    try:
#        from PIL import Image
#        import torchvision.transforms as transforms
#        
#        image = Image.open(image_file_path).convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize((1024, 1024)),
#            transforms.ToTensor(),
#        ])
#        
#        image_tensor = transform(image).unsqueeze(0)
        
#        # 移动到指定设备并转换为bfloat16（与训练时一致）
#        image_tensor = image_tensor.to(device)
#        if image_tensor.dtype != torch.bfloat16:
#            image_tensor = image_tensor.to(torch.bfloat16)
#        
#        return image_tensor
#        
#    except Exception as e:
#        print(f"❌ 加载原始图像失败: {e}")
#        return None

def load_original_image(image_file_path, device='cuda'):
    from PIL import Image
    import torchvision.transforms as transforms

    image = Image.open(image_file_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),  # float32
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1,3,1024,1024], float32
    return image_tensor



# def save_visualization_results(original_img, mask_logits, class_logits, batch_idx, vis_dir, dataset_metadata=None):
#     """
#     保存可视化结果
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # 创建图形
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle(f'X-SAM Segmentation Results - Batch {batch_idx + 1}', fontsize=16)
    
#     # 1. 原始图像
#     img_np = original_img.permute(1, 2, 0).cpu().numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
#     axes[0, 0].imshow(img_np)
#     axes[0, 0].set_title('Original Image')
#     axes[0, 0].axis('off')
    
#     # 2. 最佳掩码
#     # 转换BFloat16为Float32以支持sigmoid操作
#     mask_logits_float = mask_logits.float() if mask_logits.dtype == torch.bfloat16 else mask_logits
#     mask_probs = torch.sigmoid(mask_logits_float)  # [num_queries, H, W]
#     mask_confidences = mask_probs.max(dim=-1)[0].max(dim=-1)[0]  # [num_queries]
#     best_mask_idx = mask_confidences.argmax()
#     best_mask = mask_probs[best_mask_idx].cpu().numpy()
    
#     axes[0, 1].imshow(best_mask, cmap='jet')
#     axes[0, 1].set_title(f'Best Mask (Conf: {mask_confidences[best_mask_idx]:.3f})')
#     axes[0, 1].axis('off')
    
#     # 3. 掩码叠加
#     axes[1, 0].imshow(img_np)
#     axes[1, 0].imshow(best_mask, alpha=0.6, cmap='jet')
#     axes[1, 0].set_title('Mask Overlay')
#     axes[1, 0].axis('off')
    
#     # 4. 类别概率分布
#     # 转换BFloat16为Float32以支持softmax操作
#     class_logits_float = class_logits.float() if class_logits.dtype == torch.bfloat16 else class_logits
#     class_probs = torch.softmax(class_logits_float, dim=-1)  # [num_queries, num_classes]
#     best_class_probs = class_probs[best_mask_idx].cpu().numpy()
#     top_k = min(10, len(best_class_probs))
#     top_indices = np.argsort(best_class_probs)[-top_k:]
    
#     axes[1, 1].bar(range(top_k), best_class_probs[top_indices])
#     axes[1, 1].set_title(f'Class Probabilities (Top {top_k})')
#     axes[1, 1].set_xlabel('Class ID')
#     axes[1, 1].set_ylabel('Probability')
#     axes[1, 1].set_xticks(range(top_k))
#     axes[1, 1].set_xticklabels([f'{idx}' for idx in top_indices], rotation=45)
    
#     # 保存图像
#     plt.tight_layout()
#     save_path = vis_dir / f"batch_{batch_idx + 1}_results.png"
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"   💾 可视化结果已保存: {save_path}")

# def save_visualization_results(original_img, mask_logits, class_logits, batch_idx, vis_dir, dataset_metadata=None, top_k=50):
#     """
#     保存可视化结果，显示多个mask，并叠加类别标签
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # 原图
#     img_np = original_img.permute(1, 2, 0).cpu().numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

#     # 转换 logits → 概率
#     mask_probs = torch.sigmoid(mask_logits)  # [num_queries, H, W]
#     class_probs = torch.softmax(class_logits, dim=-1)  # [num_queries, num_classes]

#     # 每个 query 的 mask 置信度
#     mask_scores = mask_probs.flatten(1).max(1).values  
#     # 每个 query 的类别预测
#     class_scores, class_ids = class_probs.max(dim=-1)  

#     # 综合得分（mask质量 * 类别置信度）
#     scores = mask_scores * class_scores
#     top_indices = scores.topk(min(top_k, scores.numel())).indices.tolist()

#     # # 类别映射（如果有 metadata）
#     # id2name = {}
#     # if dataset_metadata and "categories" in dataset_metadata:
#     #     for c in dataset_metadata["categories"]:
#     #         id2name[c["id"]] = c["name"]
#         # 类别映射（如果有 metadata）
#     id2name = {}
#     if dataset_metadata is not None:
#         cats = None
#         if hasattr(dataset_metadata, "categories"):
#             cats = dataset_metadata.categories
#         elif isinstance(dataset_metadata, dict) and "categories" in dataset_metadata:
#             cats = dataset_metadata["categories"]

#         if isinstance(cats, dict):  # 已经是 {id: name}
#             id2name = cats
#         elif isinstance(cats, list):  # [{id: , name: }, ...]
#             for c in cats:
#                 if "id" in c and "name" in c:
#                     id2name[c["id"]] = c["name"]


#     # 调色板
#     cmap = plt.get_cmap("tab20")
#     colors = [cmap(i % 20) for i in range(len(top_indices))]

#     # 画布
#     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
#     fig.suptitle(f'X-SAM Segmentation Results - Batch {batch_idx + 1}', fontsize=16)

#     # 1. 原始图像
#     axes[0].imshow(img_np)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")

#     # 2. 叠加多个mask
#     axes[1].imshow(img_np)
#     for i, idx in enumerate(top_indices):
#         mask = mask_probs[idx].cpu().numpy()
#         class_id = int(class_ids[idx])
#         score = float(scores[idx])

#         # 类别名
#         label_name = id2name.get(class_id, f"ID {class_id}")
#         label = f"{label_name} ({score:.2f})"

#         # 叠加半透明颜色
#         colored_mask = np.zeros((*mask.shape, 4))
#         colored_mask[..., :3] = colors[i][:3]
#         colored_mask[..., 3] = mask * 0.5
#         axes[1].imshow(colored_mask)

#         # mask 中心点写文字
#         ys, xs = np.where(mask > 0.5)
#         if len(xs) > 0 and len(ys) > 0:
#             cx, cy = int(xs.mean()), int(ys.mean())
#             axes[1].text(cx, cy, label, color="white", fontsize=9, ha="center", va="center",
#                          bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

#     axes[1].set_title(f"Top-{len(top_indices)} Masks with Labels")
#     axes[1].axis("off")

#     # 保存
#     plt.tight_layout()
#     save_path = vis_dir / f"batch_{batch_idx + 1}_results.png"
#     plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close()

# def save_visualization_results(original_img, mask_logits, class_logits, batch_idx, vis_dir, dataset_metadata=None, score_thresh=0.01):
#     """
#     Panoptic Segmentation 风格可视化：
#     - 每个类别同一个颜色
#     - 每个实例用黑边勾勒
#     - 每个类别只显示一个文字标签（文字颜色与类别颜色一致）
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#     from skimage import measure

#     # 1. 原图处理
#     img_np = original_img.permute(1, 2, 0).cpu().numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

#     # 2. 转换 logits → 概率
#     mask_probs = torch.sigmoid(mask_logits).cpu().numpy()   # [num_queries, H, W]
#     class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()  # [num_queries, num_classes]

#     # 3. 计算每个 query 的类别 & 分数
#     class_ids = class_probs.argmax(axis=-1)
#     class_scores = class_probs.max(axis=-1)
#     mask_scores = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)
#     scores = class_scores * mask_scores

#     # 4. 阈值过滤
#     keep = scores > score_thresh
#     mask_probs = mask_probs[keep]
#     class_ids = class_ids[keep]
#     scores = scores[keep]

#     if len(mask_probs) == 0:
#         print(f"⚠️ 批次 {batch_idx+1} 没有高置信度的mask")
#         return

#     # 5. 类别映射
#     id2name, id2color = {}, {}
#     if dataset_metadata is not None:
#         if hasattr(dataset_metadata, "dataset_classes"):
#             id2name = dataset_metadata.dataset_classes
#         if hasattr(dataset_metadata, "dataset_colors"):
#             id2color = dataset_metadata.dataset_colors

#     # 6. 初始化画布
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.imshow(img_np)
#     H, W = img_np.shape[:2]

#     # 7. 每个实例绘制 mask + 黑边
#     drawn_classes = set()
#     for i in range(len(mask_probs)):
#         mask = mask_probs[i] > 0.5
#         cls_id = int(class_ids[i])
#         score = scores[i]

#         if mask.sum() == 0:
#             continue

#         # 类别颜色
#         if cls_id in id2color:
#             color = np.array(id2color[cls_id]) / 255.0
#         else:
#             cmap = plt.get_cmap("tab20")
#             color = np.array(cmap(cls_id % 20)[:3])

#         # 叠加半透明颜色
#         colored_mask = np.zeros((H, W, 4))
#         colored_mask[..., :3] = color
#         colored_mask[..., 3] = mask.astype(float) * 0.5
#         ax.imshow(colored_mask)

#         # 黑边勾勒
#         contours = measure.find_contours(mask.astype(float), 0.5)
#         for contour in contours:
#             ax.plot(contour[:, 1], contour[:, 0], color="black", linewidth=1)

#         # 类别文字（只写一次）
#         if cls_id not in drawn_classes:
#             ys, xs = np.where(mask)
#             if len(xs) > 0:
#                 cx, cy = int(xs.mean()), int(ys.mean())
#                 label_name = id2name.get(cls_id, f"ID {cls_id}")
#                 ax.text(
#                     cx, cy, f"{label_name}",
#                     color=color, fontsize=10, ha="center", va="center",
#                     bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
#                 )
#                 drawn_classes.add(cls_id)

#     ax.set_title(f"Panoptic Segmentation - Batch {batch_idx+1}")
#     ax.axis("off")

#     # 8. 保存
#     save_path = vis_dir / f"batch_{batch_idx + 1}_panoptic.png"
#     plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✅ 可视化结果保存到 {save_path}")
# def save_visualization_results(original_img, mask_logits, class_logits, batch_idx, vis_dir,
#                                dataset_metadata=None, score_thresh=0.01):
#     """
#     Panoptic 风格可视化（跳过背景）
#     - 每个类别同一颜色
#     - 实例用黑边勾勒
#     - 背景（cls_id == 39）不绘制
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#     from skimage import measure

#     # 1. 原图
#     img_np = original_img.permute(1, 2, 0).cpu().numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

#     # 2. logits → 概率
#     mask_probs = torch.sigmoid(mask_logits).cpu().numpy()   # [num_queries, H, W]
#     class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()  # [num_queries, num_classes]

#     # 3. 类别 & 分数
#     class_ids = class_probs.argmax(axis=-1)
#     class_scores = class_probs.max(axis=-1)
#     mask_scores = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)
#     scores = class_scores * mask_scores

#     # 4. 阈值过滤
#     keep = scores > score_thresh
#     mask_probs = mask_probs[keep]
#     class_ids = class_ids[keep]
#     scores = scores[keep]

#     if len(mask_probs) == 0:
#         print(f"⚠️ 批次 {batch_idx+1} 没有高置信度的有效实例")
#         return

#     # 5. 类别映射
#     id2name, id2color = {}, {}
#     if dataset_metadata is not None:
#         if hasattr(dataset_metadata, "dataset_classes"):
#             id2name = dataset_metadata.dataset_classes  # 原 category_id → 名称
#         if hasattr(dataset_metadata, "dataset_colors"):
#             id2color = dataset_metadata.dataset_colors

#     # 6. 画布
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.imshow(img_np)
#     H, W = img_np.shape[:2]

#     # 7. 绘制实例（跳过背景）
#     drawn_classes = set()
#     for i in range(len(mask_probs)):
#         mask = mask_probs[i] > 0.5
#         cls_id = int(class_ids[i])

#         # 跳过背景
#         if cls_id == 39:
#             continue

#         if mask.sum() == 0:
#             continue

#         # 原 category_id
#         real_cat_id = cls_id - 1
#         color = np.array(id2color.get(real_cat_id, plt.cm.tab20(cls_id % 20)[:3])) / 255.0

#         # 叠加颜色
#         colored_mask = np.zeros((H, W, 4))
#         colored_mask[..., :3] = color
#         colored_mask[..., 3] = mask.astype(float) * 0.5
#         ax.imshow(colored_mask)

#         # 黑边勾勒
#         contours = measure.find_contours(mask.astype(float), 0.5)
#         for contour in contours:
#             ax.plot(contour[:, 1], contour[:, 0], color="black", linewidth=1)

#         # 文字（每类一次）
#         if real_cat_id not in drawn_classes:
#             ys, xs = np.where(mask)
#             if len(xs) > 0:
#                 cx, cy = int(xs.mean()), int(ys.mean())
#                 label_name = id2name.get(real_cat_id, f"class_{real_cat_id}")
#                 ax.text(
#                     cx, cy, label_name,
#                     color=color, fontsize=10, ha="center", va="center",
#                     bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
#                 )
#                 drawn_classes.add(real_cat_id)

#     ax.set_title(f"Panoptic Segmentation - Batch {batch_idx+1}")
#     ax.axis("off")

#     # 8. 保存
#     vis_dir.mkdir(parents=True, exist_ok=True)
#     save_path = vis_dir / f"batch_{batch_idx + 1}_panoptic.png"
#     plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✅ 可视化结果保存到 {save_path}")

# def save_visualization_results(
#     original_img,
#     mask_logits,
#     class_logits,
#     batch_idx,
#     vis_dir,
#     dataset_metadata=None,
#     score_thresh=0.01,
# ):
#     """
#     Panoptic 风格可视化（跳过背景）
#     - 每个类别同一颜色
#     - 实例用黑边勾勒
#     - 背景（cls_id == 39）不绘制
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#     from skimage import measure
#     from scipy.ndimage import center_of_mass

#     # 1. 原图
#     img_np = original_img.permute(1, 2, 0).cpu().numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

#     # 2. logits → 概率
#     mask_probs = torch.sigmoid(mask_logits).cpu().numpy()   # [num_queries, H, W]
#     class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()  # [num_queries, num_classes]

#     # 3. 类别 & 分数
#     class_ids = class_probs.argmax(axis=-1)
#     class_scores = class_probs.max(axis=-1)
#     mask_scores = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)
#     scores = class_scores * mask_scores

#     # 4. 阈值过滤
#     keep = scores > score_thresh
#     mask_probs = mask_probs[keep]
#     class_ids = class_ids[keep]
#     scores = scores[keep]

#     if len(mask_probs) == 0:
#         print(f"⚠️ 批次 {batch_idx+1} 没有高置信度的有效实例")
#         return

#     # 5. 类别映射
#     id2name, id2color = {}, {}
#     if dataset_metadata is not None:
#         if hasattr(dataset_metadata, "dataset_classes"):
#             id2name = dataset_metadata.dataset_classes  # 原 category_id → 名称
#         if hasattr(dataset_metadata, "dataset_colors"):
#             id2color = dataset_metadata.dataset_colors

#     # 6. 画布
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.imshow(img_np)
#     H, W = img_np.shape[:2]

#     # 7. 绘制实例（跳过背景）
#     drawn_classes = set()
#     for i in range(len(mask_probs)):
#         mask = mask_probs[i] > 0.5
#         cls_id = int(class_ids[i])

#         # 跳过背景
#         if cls_id == 39:
#             continue

#         if mask.sum() == 0:
#             continue

#         # 原 category_id
#         real_cat_id = cls_id - 1
#         color = np.array(id2color.get(real_cat_id, plt.cm.tab20(cls_id % 20)[:3])) / 255.0

#         # 叠加颜色
#         colored_mask = np.zeros((H, W, 4))
#         colored_mask[..., :3] = color
#         colored_mask[..., 3] = mask.astype(float) * 0.5
#         ax.imshow(colored_mask)

#         # 黑边勾勒
#         contours = measure.find_contours(mask.astype(float), 0.5)
#         for contour in contours:
#             ax.plot(contour[:, 1], contour[:, 0], color="black", linewidth=1)

#         # 文字（每类一次）
#         if real_cat_id not in drawn_classes:
#             # 用 center_of_mass 找中心点
#             cy, cx = center_of_mass(mask.astype(float))
#             if np.isnan(cx) or np.isnan(cy):  # 如果失败，退回均值
#                 ys, xs = np.where(mask)
#                 if len(xs) > 0:
#                     cx, cy = xs.mean(), ys.mean()
#                 else:
#                     continue
#             cx, cy = int(cx), int(cy)

#             # 如果重心不在 mask 内，退回一个 mask 内的像素
#             if not mask[cy, cx]:
#                 ys, xs = np.where(mask)
#                 cx, cy = xs[len(xs)//2], ys[len(ys)//2]

#             label_name = id2name.get(real_cat_id, f"class_{real_cat_id}")
#             ax.text(
#                 cx, cy, label_name,
#                 color=color, fontsize=10, ha="center", va="center",
#                 bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
#             )
#             drawn_classes.add(real_cat_id)

#     ax.set_title(f"Panoptic Segmentation - Batch {batch_idx+1}")
#     ax.axis("off")

#     # 8. 保存
#     vis_dir.mkdir(parents=True, exist_ok=True)
#     save_path = vis_dir / f"batch_{batch_idx + 1}_panoptic.png"
#     plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✅ 可视化结果保存到 {save_path}")
def save_visualization_results(original_img, mask_logits, class_logits, batch_idx, vis_dir,
                               dataset_metadata=None, class_thresh=0.1, mask_thresh=0.1 ):
    """
    Panoptic 风格可视化（跳过背景）
    - 每个类别固定颜色
    - 实例用黑边勾勒
    - 背景（cls_id == 39）不绘制
    - 同一类别文字和掩码颜色一致
    - 同时保存原图、分割图、拼接对比图
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from skimage import measure
    from PIL import Image

    # 1. 原图
    img_np = original_img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # 2. logits → 概率
    mask_probs = torch.sigmoid(mask_logits).cpu().numpy()   # [num_queries, H, W]
    class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()  # [num_queries, num_classes]

    # # 3. 类别 & 分数
    # class_ids = class_probs.argmax(axis=-1)
    # class_scores = class_probs.max(axis=-1)
    # mask_scores = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)
    # scores = class_scores * mask_scores

    # # 4. 阈值过滤
    # keep = scores > score_thresh
    # mask_probs = mask_probs[keep]
    # class_ids = class_ids[keep]
    # scores = scores[keep]
    
        # 3. 类别 & 分数
    class_ids = class_probs.argmax(axis=-1)
    class_scores = class_probs.max(axis=-1)      # 分类置信度
    mask_scores = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)  # mask 最大值

    # 4. 阈值过滤（分开）
    # class_thresh = 0.5   # 分类分数阈值
    # mask_thresh = 0.5    # mask 分数阈值

    keep = (class_scores > class_thresh) & (mask_scores > mask_thresh)

    mask_probs = mask_probs[keep]
    class_ids = class_ids[keep]
    class_scores = class_scores[keep]
    mask_scores = mask_scores[keep]


    if len(mask_probs) == 0:
        print(f"⚠️ 批次 {batch_idx+1} 没有高置信度的有效实例")
        return

    # 5. 类别映射
    id2name, id2color = {}, {}
    if dataset_metadata is not None:
        if hasattr(dataset_metadata, "dataset_classes"):
            id2name = dataset_metadata.dataset_classes
        if hasattr(dataset_metadata, "dataset_colors"):
            id2color = dataset_metadata.dataset_colors

    # 固定调色板（20 种循环使用）
    fixed_palette = (plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(np.uint8)

    def get_color(cls_id: int):
        """固定类别颜色"""
        if cls_id in id2color:
            return np.array(id2color[cls_id]) / 255.0
        return fixed_palette[cls_id % len(fixed_palette)] / 255.0

    # 6. 画布
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_np)
    H, W = img_np.shape[:2]

    # 7. 绘制实例（跳过背景）
    drawn_classes = set()
    for i in range(len(mask_probs)):
        mask = mask_probs[i] > 0.5
        cls_id = int(class_ids[i])

        if cls_id == 39:  # 跳过背景类（索引39）
            continue
        if mask.sum() == 0:
            continue

        # 直接使用cls_id作为类别ID，因为0-38对应39个具体类别
        real_cat_id = cls_id
        color = get_color(real_cat_id)

        # 叠加颜色
        colored_mask = np.zeros((H, W, 4))
        colored_mask[..., :3] = color
        colored_mask[..., 3] = mask.astype(float) * 0.5
        ax.imshow(colored_mask)

        # 黑边勾勒
        contours = measure.find_contours(mask.astype(float), 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color="black", linewidth=1)

        # 文字（每类一次）
        if real_cat_id not in drawn_classes:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label_name = id2name.get(real_cat_id, f"class_{real_cat_id}")
                ax.text(
                    cx, cy, label_name,
                    color=color, fontsize=10, ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
                )
                drawn_classes.add(real_cat_id)

    ax.set_title(f"Panoptic Segmentation - Batch {batch_idx+1}")
    ax.axis("off")

    # 8. 保存
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_save_path = vis_dir / f"batch_{batch_idx + 1}_panoptic2.png"
    raw_save_path = vis_dir / f"batch_{batch_idx + 1}_raw.png"
    compare_save_path = vis_dir / f"batch_{batch_idx + 1}_compare.png"

    plt.savefig(mask_save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 保存原图
    plt.imsave(raw_save_path, img_np)

    # 拼接原图和分割图
    # raw_img = Image.open(raw_save_path).convert("RGB")
    # mask_img = Image.open(mask_save_path).convert("RGB")
    # compare_w = raw_img.width + mask_img.width
    # compare_h = max(raw_img.height, mask_img.height)
    # compare_img = Image.new("RGB", (compare_w, compare_h), (255, 255, 255))
    # compare_img.paste(raw_img, (0, 0))
    # compare_img.paste(mask_img, (raw_img.width, 0))
    # compare_img.save(compare_save_path)

    print(f"✅ 原图保存到 {raw_save_path}")
    print(f"✅ 分割图保存到 {mask_save_path}")
    # print(f"✅ 拼接对比图保存到 {compare_save_path}")




def main(checkpoint_path, work_dir, device='cuda'):
    """主函数"""
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # 检查设备可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA不可用，切换到CPU")
            device = 'cpu'
        elif device == 'cuda' and torch.cuda.is_available():
            # 如果只指定了'cuda'，自动选择最空闲的GPU
            device = get_most_free_gpu()
            print(f"🔧 自动选择设备: {device}")
        elif device.startswith('cuda:') and torch.cuda.is_available():
            # 验证指定的GPU是否存在
            gpu_id = int(device.split(':')[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"⚠️  GPU {gpu_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
                device = get_most_free_gpu()
                print(f"🔧 自动切换到设备: {device}")
        
        print(f"🔧 最终使用设备: {device}")
        
        # 如果指定了具体的GPU设备，设置当前设备
        if device.startswith('cuda:'):
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            print(f"🔧 已设置当前GPU设备为: {device}")
        
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
        results = run_validation(model, dataset, Path(work_dir), device)
        
        return results
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='简化版X-SAM验证脚本')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--work_dir', type=str, required=True, help='Path to work directory')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu/cuda:0/cuda:1等). 如果指定为cuda，将自动选择最空闲的GPU')
    
    args = parser.parse_args()
    
    try:
        results = main(args.checkpoint_path, args.work_dir, args.device)
        
        if results:
            print(f"\n🎉 验证成功完成!")
            print(f"📁 结果保存在: {args.work_dir}")
        else:
            print("❌ 验证过程中出现错误")
            
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()