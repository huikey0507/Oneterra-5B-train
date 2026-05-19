#!/usr/bin/env python3
"""
SOTA训练数据金标准标注可视化脚本 - 带类别标签版本
基于X-SAM的数据流方法，在图像上标注英文类别名称
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import cv2

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_annotations(annotations_path):
    """加载标注文件"""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations

def load_panoptic_label(label_path):
    """加载全景分割标签 - 使用X-SAM的RGB解码方法"""
    if not os.path.exists(label_path):
        return None
    
    # 加载RGB格式的全景分割标签
    label_img = Image.open(label_path).convert('RGB')
    label_array = np.array(label_img)
    
    # 使用X-SAM的RGB解码方法
    panoptic_id = label_array[:, :, 0].astype(np.uint32) + \
                  label_array[:, :, 1].astype(np.uint32) * 256 + \
                  label_array[:, :, 2].astype(np.uint32) * 256 * 256
    
    return panoptic_id

def create_color_map(num_classes):
    """创建颜色映射 - 为SOTA数据集的39个类别创建独特颜色"""
    colors = [
        # Thing类别 (0-28) - 实例对象
        [255, 0, 0],     # 0: buildings - 红色
        [0, 255, 0],     # 1: large-vehicle - 绿色
        [0, 0, 255],     # 2: swimming-pool - 蓝色
        [255, 255, 0],   # 3: helicopter - 黄色
        [255, 0, 255],   # 4: bridge - 洋红
        [0, 255, 255],   # 5: plane - 青色
        [128, 0, 0],     # 6: ship - 深红
        [0, 128, 0],     # 7: soccer-ball-field - 深绿
        [0, 0, 128],     # 8: basketball-court - 深蓝
        [128, 128, 0],   # 9: ground-track-field - 橄榄色
        [128, 0, 128],   # 10: small-vehicle - 紫色
        [0, 128, 128],   # 11: baseball-diamond - 青绿色
        [192, 192, 192], # 12: tennis-court - 银色
        [128, 128, 128], # 13: roundabout - 灰色
        [255, 165, 0],   # 14: storage-tank - 橙色
        [255, 192, 203], # 15: harbor - 粉色
        [165, 42, 42],   # 16: container-crane - 棕色
        [0, 100, 0],     # 17: airport - 深绿
        [70, 130, 180],  # 18: helipad - 钢蓝
        [255, 20, 147],  # 19: chimney - 深粉红
        [50, 205, 50],   # 20: expressway service area - 酸橙绿
        [255, 69, 0],    # 21: expressway toll station - 红橙色
        [138, 43, 226],  # 22: dam - 蓝紫色
        [0, 191, 255],   # 23: golf field - 深天蓝
        [255, 215, 0],   # 24: overpass - 金色
        [220, 20, 60],   # 25: stadium - 深红
        [0, 250, 154],   # 26: train station - 中春绿
        [255, 105, 180], # 27: vehicle - 热粉色
        [30, 144, 255],  # 28: windmill - 道奇蓝
        
        # Stuff类别 (29-38) - 材质区域
        [139, 69, 19],   # 29: bare land - 马鞍棕
        [34, 139, 34],   # 30: grass - 森林绿
        [105, 105, 105], # 31: pavement - 暗灰
        [64, 64, 64],    # 32: road - 深灰
        [0, 128, 0],     # 33: tree - 绿色
        [0, 0, 139],     # 34: water - 深蓝
        [154, 205, 50],  # 35: agriculture land - 黄绿
        [34, 139, 34],   # 36: forest land - 森林绿
        [160, 82, 45],   # 37: barren land - 马鞍棕
        [105, 105, 105], # 38: urban land - 暗灰
    ]
    
    return colors[:num_classes]

def get_centroid(mask):
    """获取掩码的质心坐标"""
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0:
        return None
    return (int(np.mean(x_coords)), int(np.mean(y_coords)))

def add_labels_to_image(image, segment_info, font_size=20):
    """在图像上添加类别标签"""
    # 转换为PIL图像
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # 为每个segment添加标签
    for seg in segment_info:
        if seg['pixel_count'] < 100:  # 只标注较大的区域
            continue
            
        # 获取质心坐标
        centroid = seg.get('centroid')
        if centroid is None:
            continue
            
        x, y = centroid
        label = seg['category']
        
        # 绘制标签背景
        bbox = draw.textbbox((x, y), label, font=font)
        padding = 5
        draw.rectangle([
            bbox[0] - padding, bbox[1] - padding,
            bbox[2] + padding, bbox[3] + padding
        ], fill=(255, 255, 255, 200), outline=(0, 0, 0))
        
        # 绘制文本
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    
    return pil_image

def visualize_training_sample_with_labels(image_path, panoptic_id, categories, output_path=None):
    """可视化单张训练样本，带类别标签"""
    
    # 加载原图
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # 创建颜色映射
    colors = create_color_map(len(categories))
    
    # 创建可视化图像
    height, width = panoptic_id.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 获取所有唯一的segment ID
    unique_ids = np.unique(panoptic_id)
    unique_ids = unique_ids[unique_ids > 0]  # 排除背景
    
    # 为每个segment分配颜色
    segment_info = []
    for unique_id in unique_ids:
        # 解码得到类别ID和实例ID
        class_id = unique_id >> 17
        instance_id = unique_id & ((1 << 17) - 1)
        
        # 确保类别ID在有效范围内
        if class_id < len(colors):
            color = colors[class_id]
            mask = (panoptic_id == unique_id)
            colored_mask[mask] = color
            
            # 获取质心坐标
            centroid = get_centroid(mask)
            
            # 记录segment信息
            category_name = categories[class_id]['name'] if class_id < len(categories) else f'Unknown_{class_id}'
            segment_info.append({
                'id': unique_id,
                'class_id': class_id,
                'instance_id': instance_id,
                'category': category_name,
                'color': color,
                'pixel_count': np.sum(mask),
                'isthing': categories[class_id]['isthing'] if class_id < len(categories) else 0,
                'centroid': centroid,
                'mask': mask
            })
    
    # 创建图像叠加
    alpha = 0.6
    overlay = (image_array * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    # 添加标签到叠加图像
    overlay_with_labels = add_labels_to_image(overlay, segment_info)
    overlay_with_labels_array = np.array(overlay_with_labels)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原图
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 全景分割掩码
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('Panoptic Segmentation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 叠加图像（无标签）
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay Result', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 叠加图像（带标签）
    axes[1, 1].imshow(overlay_with_labels_array)
    axes[1, 1].set_title('Overlay with Category Labels', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization with labels saved to: {output_path}")
    
    plt.show()
    
    return segment_info

def create_legend(categories, output_path=None):
    """创建类别图例"""
    colors = create_color_map(len(categories))
    
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(categories))
    ax.axis('off')
    
    # 添加标题
    ax.text(0.5, len(categories) - 0.5, 'SOTA Dataset Categories', 
            fontsize=16, fontweight='bold', ha='center')
    
    # 添加类别
    for i, (cat, color) in enumerate(zip(categories, colors)):
        y_pos = len(categories) - i - 1.5
        
        # 绘制颜色块
        rect = patches.Rectangle((0.1, y_pos), 0.1, 0.8, 
                               facecolor=[c/255.0 for c in color], 
                               edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # 添加类别名称
        category_type = "Thing" if cat['isthing'] else "Stuff"
        label = f"{cat['id']}: {cat['name']} ({category_type})"
        ax.text(0.25, y_pos + 0.4, label, fontsize=10, va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Legend saved to: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SOTA Training Data Visualization with Labels')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='labeled_visualization', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='datas/sota', help='Data directory')
    parser.add_argument('--create_legend', action='store_true', help='Create category legend')
    
    args = parser.parse_args()
    
    # 数据路径配置
    base_dir = args.data_dir
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "panoptic_labels")
    annotations_path = os.path.join(base_dir, "train_annotations.json")
    
    # 检查路径是否存在
    if not os.path.exists(annotations_path):
        print(f"Error: Annotation file not found: {annotations_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载标注数据
    print("Loading annotation data...")
    annotations = load_annotations(annotations_path)
    categories = annotations.get('categories', [])
    images = annotations.get('images', [])
    
    print(f"Dataset Information:")
    print(f"  - Total images: {len(images)}")
    print(f"  - Total categories: {len(categories)}")
    print(f"  - Thing categories: {sum(1 for cat in categories if cat['isthing'])}")
    print(f"  - Stuff categories: {sum(1 for cat in categories if not cat['isthing'])}")
    
    # 创建图例
    if args.create_legend:
        legend_path = os.path.join(args.output_dir, "category_legend.png")
        create_legend(categories, legend_path)
    
    # 随机选择图像
    selected_images = random.sample(images, min(args.num_samples, len(images)))
    
    print(f"\nVisualizing {len(selected_images)} training samples with labels...")
    
    # 可视化每张图像
    for i, image_info in enumerate(selected_images):
        image_filename = image_info['file_name']
        image_id = image_info['id']
        
        print(f"\n{'='*60}")
        print(f"Processing sample {i+1}/{len(selected_images)}: {image_filename}")
        print(f"{'='*60}")
        
        # 构建文件路径
        image_path = os.path.join(images_dir, image_filename)
        label_filename = image_filename
        label_path = os.path.join(labels_dir, label_filename)
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            continue
        
        if not os.path.exists(label_path):
            print(f"Error: Label file not found: {label_path}")
            continue
        
        # 加载全景分割标签
        panoptic_id = load_panoptic_label(label_path)
        
        if panoptic_id is None:
            print("Error: Failed to load panoptic label")
            continue
        
        print(f"Image size: {panoptic_id.shape}")
        print(f"Unique segments: {len(np.unique(panoptic_id))}")
        
        # 生成输出文件名
        base_name = os.path.splitext(image_filename)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_labeled.png")
        
        # 可视化
        segment_info = visualize_training_sample_with_labels(
            image_path, panoptic_id, categories, output_path
        )
        
        # 显示统计信息
        thing_count = sum(1 for seg in segment_info if seg['isthing'])
        stuff_count = sum(1 for seg in segment_info if not seg['isthing'])
        
        print(f"Segmentation Statistics:")
        print(f"  - Thing instances: {thing_count}")
        print(f"  - Stuff regions: {stuff_count}")
        print(f"  - Total segments: {len(segment_info)}")
        
        # 显示主要类别
        class_counts = {}
        for seg in segment_info:
            class_name = seg['category']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Main categories:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {class_name}: {count}")

if __name__ == "__main__":
    main()