#!/usr/bin/env python3
"""
全景分割标注可视化脚本
用于可视化SOTA数据集的训练图片和对应的全景分割标注
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from pathlib import Path

def load_annotations(annotations_path):
    """加载标注文件"""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations

def load_panoptic_label(label_path):
    """加载全景分割标签"""
    if not os.path.exists(label_path):
        return None
    
    # 加载RGB格式的全景分割标签
    label_img = Image.open(label_path).convert('RGB')
    label_array = np.array(label_img)
    
    # 将RGB转换为ID
    panoptic_id = label_array[:, :, 0].astype(np.uint32) + \
                  label_array[:, :, 1].astype(np.uint32) * 256 + \
                  label_array[:, :, 2].astype(np.uint32) * 256 * 256
    
    return panoptic_id

def create_color_map(num_classes):
    """创建颜色映射"""
    # 使用固定的颜色调色板，确保每个类别都有独特的颜色
    colors = [
        [255, 0, 0],     # 红色
        [0, 255, 0],     # 绿色  
        [0, 0, 255],     # 蓝色
        [255, 255, 0],   # 黄色
        [255, 0, 255],   # 洋红
        [0, 255, 255],   # 青色
        [128, 0, 0],     # 深红
        [0, 128, 0],     # 深绿
        [0, 0, 128],     # 深蓝
        [128, 128, 0],   # 橄榄色
        [128, 0, 128],   # 紫色
        [0, 128, 128],   # 青绿色
        [192, 192, 192], # 银色
        [128, 128, 128], # 灰色
        [255, 165, 0],   # 橙色
        [255, 192, 203], # 粉色
        [165, 42, 42],   # 棕色
        [0, 100, 0],     # 深绿
        [70, 130, 180],  # 钢蓝
        [255, 20, 147],  # 深粉红
        [50, 205, 50],   # 酸橙绿
        [255, 69, 0],    # 红橙色
        [138, 43, 226],  # 蓝紫色
        [0, 191, 255],   # 深天蓝
        [255, 215, 0],   # 金色
        [220, 20, 60],   # 深红
        [0, 250, 154],   # 中春绿
        [255, 105, 180], # 热粉色
        [30, 144, 255],  # 道奇蓝
        [255, 140, 0],   # 深橙色
        [124, 252, 0],   # 草坪绿
        [255, 0, 255],   # 洋红
        [0, 255, 127],   # 春绿
        [255, 99, 71],   # 番茄色
        [64, 224, 208],  # 绿松石色
        [255, 127, 80],  # 珊瑚色
        [72, 209, 204],  # 中绿松石色
        [199, 21, 133],  # 中紫罗兰红
        [25, 25, 112],   # 午夜蓝
    ]
    
    # 如果类别数超过预定义颜色数，随机生成更多颜色
    while len(colors) < num_classes:
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        colors.append(color)
    
    return colors[:num_classes]

def visualize_panoptic_segmentation(image_path, panoptic_id, categories, output_path=None):
    """可视化全景分割结果"""
    
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
                'pixel_count': np.sum(mask)
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
    print(f"\n图像中的类别信息:")
    print(f"总共发现 {len(segment_info)} 个分割区域")
    print("-" * 60)
    
    # 按类别分组显示
    class_groups = {}
    for seg in segment_info:
        class_name = seg['category']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(seg)
    
    for class_name, segments in class_groups.items():
        print(f"类别: {class_name} (共 {len(segments)} 个实例)")
        for seg in segments:
            print(f"  - 实例ID: {seg['instance_id']}, 像素数: {seg['pixel_count']}")
    
    plt.show()
    
    return segment_info

def main():
    """主函数"""
    # 数据路径配置
    base_dir = "datas/sota"
    train_images_dir = os.path.join(base_dir, "train/images")
    train_labels_dir = os.path.join(base_dir, "train/panoptic_labels")
    annotations_path = os.path.join(base_dir, "train_annotations.json")
    
    # 检查路径是否存在
    if not os.path.exists(annotations_path):
        print(f"错误: 标注文件不存在: {annotations_path}")
        return
    
    if not os.path.exists(train_images_dir):
        print(f"错误: 训练图像目录不存在: {train_images_dir}")
        return
    
    if not os.path.exists(train_labels_dir):
        print(f"错误: 训练标签目录不存在: {train_labels_dir}")
        return
    
    # 加载标注数据
    print("正在加载标注数据...")
    annotations = load_annotations(annotations_path)
    categories = annotations.get('categories', [])
    
    print(f"数据集信息:")
    print(f"  - 总图像数: {len(annotations.get('images', []))}")
    print(f"  - 类别数: {len(categories)}")
    print(f"  - 标注数: {len(annotations.get('annotations', []))}")
    
    # 显示类别信息
    print(f"\n类别列表:")
    for i, cat in enumerate(categories):
        print(f"  {i}: {cat['name']} (id: {cat['id']})")
    
    # 随机选择一张图像进行可视化
    images = annotations.get('images', [])
    if not images:
        print("错误: 没有找到图像数据")
        return
    
    # 选择第一张图像
    selected_image = images[0]
    image_filename = selected_image['file_name']
    image_id = selected_image['id']
    
    print(f"\n选择图像: {image_filename} (ID: {image_id})")
    
    # 构建文件路径
    image_path = os.path.join(train_images_dir, image_filename)
    label_filename = image_filename  # 假设标签文件名与图像文件名相同
    label_path = os.path.join(train_labels_dir, label_filename)
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"错误: 标签文件不存在: {label_path}")
        return
    
    # 加载全景分割标签
    print("正在加载全景分割标签...")
    panoptic_id = load_panoptic_label(label_path)
    
    if panoptic_id is None:
        print("错误: 无法加载全景分割标签")
        return
    
    print(f"标签图像尺寸: {panoptic_id.shape}")
    print(f"唯一segment数: {len(np.unique(panoptic_id))}")
    
    # 创建输出目录
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(image_filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_panoptic_visualization.png")
    
    # 可视化
    print("正在生成可视化...")
    segment_info = visualize_panoptic_segmentation(
        image_path, panoptic_id, categories, output_path
    )
    
    print(f"\n可视化完成！")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    main()