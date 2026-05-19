#!/usr/bin/env python3
"""
SOTA训练数据金标准标注可视化脚本
基于X-SAM的数据流方法，可视化多张训练数据的全景分割标注
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from pathlib import Path
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
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
    # 为SOTA数据集优化的颜色调色板
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

def visualize_single_image(image_path, panoptic_id, categories, output_path=None, show_info=True):
    """可视化单张图像的全景分割结果"""
    
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
            
            # 记录segment信息
            category_name = categories[class_id]['name'] if class_id < len(categories) else f'Unknown_{class_id}'
            segment_info.append({
                'id': unique_id,
                'class_id': class_id,
                'instance_id': instance_id,
                'category': category_name,
                'color': color,
                'pixel_count': np.sum(mask),
                'isthing': categories[class_id]['isthing'] if class_id < len(categories) else 0
            })
    
    # 创建图像叠加
    alpha = 0.6
    overlay = (image_array * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 原图
    axes[0].imshow(image_array)
    axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 全景分割掩码
    axes[1].imshow(colored_mask)
    axes[1].set_title('全景分割标注', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 叠加图像
    axes[2].imshow(overlay)
    axes[2].set_title('叠加效果', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
    
    # 显示类别信息
    if show_info:
        print(f"\n图像中的类别信息:")
        print(f"总共发现 {len(segment_info)} 个分割区域")
        print("-" * 80)
        
        # 按类别分组显示
        class_groups = {}
        for seg in segment_info:
            class_name = seg['category']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(seg)
        
        # 分别显示Thing和Stuff类别
        thing_classes = {}
        stuff_classes = {}
        
        for class_name, segments in class_groups.items():
            if segments[0]['isthing']:
                thing_classes[class_name] = segments
            else:
                stuff_classes[class_name] = segments
        
        if thing_classes:
            print("Thing类别 (实例对象):")
            for class_name, segments in thing_classes.items():
                print(f"  {class_name}: {len(segments)} 个实例")
                for seg in segments:
                    print(f"    - 实例ID: {seg['instance_id']}, 像素数: {seg['pixel_count']}")
        
        if stuff_classes:
            print("\nStuff类别 (材质区域):")
            for class_name, segments in stuff_classes.items():
                print(f"  {class_name}: {len(segments)} 个区域")
                for seg in segments:
                    print(f"    - 区域ID: {seg['instance_id']}, 像素数: {seg['pixel_count']}")
    
    plt.show()
    
    return segment_info

def visualize_multiple_images(annotations, images_dir, labels_dir, num_samples=5, output_dir="visualization_results"):
    """可视化多张训练数据图像"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像
    images = annotations.get('images', [])
    categories = annotations.get('categories', [])
    
    if not images:
        print("错误: 没有找到图像数据")
        return
    
    # 随机选择图像
    selected_images = random.sample(images, min(num_samples, len(images)))
    
    print(f"数据集信息:")
    print(f"  - 总图像数: {len(images)}")
    print(f"  - 类别数: {len(categories)}")
    print(f"  - 选择可视化: {len(selected_images)} 张图像")
    
    # 显示类别信息
    print(f"\nSOTA数据集类别信息:")
    print("Thing类别 (实例对象):")
    for cat in categories:
        if cat['isthing']:
            print(f"  {cat['id']}: {cat['name']}")
    
    print("\nStuff类别 (材质区域):")
    for cat in categories:
        if not cat['isthing']:
            print(f"  {cat['id']}: {cat['name']}")
    
    # 可视化每张图像
    for i, image_info in enumerate(selected_images):
        image_filename = image_info['file_name']
        image_id = image_info['id']
        
        print(f"\n{'='*80}")
        print(f"正在处理图像 {i+1}/{len(selected_images)}: {image_filename} (ID: {image_id})")
        print(f"{'='*80}")
        
        # 构建文件路径
        image_path = os.path.join(images_dir, image_filename)
        label_filename = image_filename  # 假设标签文件名与图像文件名相同
        label_path = os.path.join(labels_dir, label_filename)
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 图像文件不存在: {image_path}")
            continue
        
        if not os.path.exists(label_path):
            print(f"错误: 标签文件不存在: {label_path}")
            continue
        
        # 加载全景分割标签
        print("正在加载全景分割标签...")
        panoptic_id = load_panoptic_label(label_path)
        
        if panoptic_id is None:
            print("错误: 无法加载全景分割标签")
            continue
        
        print(f"标签图像尺寸: {panoptic_id.shape}")
        print(f"唯一segment数: {len(np.unique(panoptic_id))}")
        
        # 生成输出文件名
        base_name = os.path.splitext(image_filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_panoptic_visualization.png")
        
        # 可视化
        print("正在生成可视化...")
        segment_info = visualize_single_image(
            image_path, panoptic_id, categories, output_path, show_info=True
        )
        
        print(f"可视化完成！输出文件: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SOTA训练数据金标准标注可视化')
    parser.add_argument('--num_samples', type=int, default=5, help='要可视化的图像数量')
    parser.add_argument('--output_dir', type=str, default='visualization_results', help='输出目录')
    parser.add_argument('--data_dir', type=str, default='datas/sota', help='数据目录')
    
    args = parser.parse_args()
    
    # 数据路径配置
    base_dir = args.data_dir
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "panoptic_labels")
    annotations_path = os.path.join(base_dir, "train_annotations.json")
    
    # 检查路径是否存在
    if not os.path.exists(annotations_path):
        print(f"错误: 标注文件不存在: {annotations_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"错误: 标签目录不存在: {labels_dir}")
        return
    
    # 加载标注数据
    print("正在加载标注数据...")
    annotations = load_annotations(annotations_path)
    
    # 可视化多张图像
    visualize_multiple_images(
        annotations, 
        images_dir, 
        labels_dir, 
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()