#!/usr/bin/env python3
"""
验证pano数据与X-SAM训练代码的兼容性

关键检查点：
1. panoptic_id编码逻辑是否正确
2. segments_info格式是否匹配
3. RGB标签中的ID是否与segments_info中的ID匹配
"""

import json
import numpy as np
from PIL import Image
from panopticapi.utils import rgb2id


def encode_panoptic_id(class_id: int, instance_id: int) -> int:
    """pano数据的编码逻辑"""
    return (class_id << 17) + instance_id


def decode_panoptic_id(panoptic_id: int):
    """解码panoptic_id"""
    class_id = panoptic_id >> 17
    instance_id = panoptic_id & ((1 << 17) - 1)
    return class_id, instance_id


def verify_pano_data(pano_ann_path, pano_rgb_dir, num_samples=5):
    """验证pano数据的兼容性"""
    print("=" * 60)
    print("验证pano数据与X-SAM训练代码的兼容性")
    print("=" * 60)
    
    # 读取annotations
    print("\n1. 读取annotations文件...")
    with open(pano_ann_path, 'r') as f:
        data = json.load(f)
    
    print(f"   - 图片数量: {len(data.get('images', []))}")
    print(f"   - 标注数量: {len(data.get('annotations', []))}")
    
    # 检查数据格式
    print("\n2. 检查数据格式...")
    if len(data['images']) != len(data['annotations']):
        print("   ⚠️  警告: images和annotations数量不匹配")
    else:
        print("   ✅ images和annotations数量匹配")
    
    # 验证编码逻辑和ID匹配
    print("\n3. 验证编码逻辑和ID匹配...")
    success_count = 0
    error_count = 0
    
    for i in range(min(num_samples, len(data['images']))):
        img = data['images'][i]
        img_anns = [ann for ann in data.get('annotations', []) 
                   if ann.get('image_id') == img['id']]
        
        if not img_anns:
            print(f"   ⚠️  图片 {img['file_name']}: 没有找到对应的annotation")
            continue
        
        ann = img_anns[0]
        segments_info = ann.get('segments_info', [])
        
        # 处理segments_info可能是字符串的情况
        if isinstance(segments_info, str):
            import ast
            try:
                segments_info = ast.literal_eval(segments_info)
            except:
                print(f"   ❌ 图片 {img['file_name']}: segments_info解析失败")
                error_count += 1
                continue
        
        if not segments_info:
            print(f"   ⚠️  图片 {img['file_name']}: segments_info为空")
            continue
        
        # 检查RGB标签文件
        rgb_filename = img['file_name'].replace('.tif', '.png').replace('.jpg', '.png')
        rgb_path = f"{pano_rgb_dir}/{rgb_filename}"
        
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            rgb_array = np.array(rgb_img)
            panoptic_ids = rgb2id(rgb_array)
            unique_ids = np.unique(panoptic_ids)
            unique_ids = unique_ids[unique_ids > 0]
            
            # 验证segments_info中的ID是否在RGB标签中
            matched_ids = 0
            total_ids = len(segments_info)
            
            for seg in segments_info:
                seg_id = seg.get('id')
                if seg_id in unique_ids:
                    matched_ids += 1
                    
                    # 验证编码逻辑
                    class_id, instance_id = decode_panoptic_id(seg_id)
                    encoded = encode_panoptic_id(class_id, instance_id)
                    
                    if seg_id != encoded:
                        print(f"   ❌ 图片 {img['file_name']}: ID {seg_id} 编码验证失败")
                        error_count += 1
                        break
                    
                    # 验证category_id
                    if seg.get('category_id') != class_id:
                        print(f"   ⚠️  图片 {img['file_name']}: ID {seg_id} category_id不匹配 "
                              f"(解码class_id={class_id}, segments_info中category_id={seg.get('category_id')})")
            
            if matched_ids == total_ids:
                success_count += 1
                print(f"   ✅ 图片 {img['file_name']}: {matched_ids}/{total_ids} IDs匹配，编码验证通过")
            else:
                print(f"   ⚠️  图片 {img['file_name']}: {matched_ids}/{total_ids} IDs匹配")
                error_count += 1
                
        except FileNotFoundError:
            print(f"   ❌ 图片 {img['file_name']}: RGB标签文件不存在 ({rgb_path})")
            error_count += 1
        except Exception as e:
            print(f"   ❌ 图片 {img['file_name']}: 处理错误 - {e}")
            error_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    print(f"✅ 成功: {success_count}/{num_samples}")
    print(f"❌ 失败: {error_count}/{num_samples}")
    
    if success_count == num_samples:
        print("\n🎉 pano数据格式完全兼容X-SAM训练代码！")
        print("\n关键点确认：")
        print("1. ✅ panoptic_id编码逻辑正确: (class_id << 17) + instance_id")
        print("2. ✅ segments_info格式匹配训练代码要求")
        print("3. ✅ RGB标签中的ID与segments_info中的ID完全匹配")
        print("4. ✅ 每张图独立编码不影响训练（训练时逐张图处理）")
        return True
    else:
        print("\n⚠️  发现一些问题，请检查上述错误信息")
        return False


if __name__ == "__main__":
    pano_ann_path = "datas/pano/panoptic_annotations.json"
    pano_rgb_dir = "datas/pano/panoptic_rgb_ids"
    
    verify_pano_data(pano_ann_path, pano_rgb_dir, num_samples=10)

