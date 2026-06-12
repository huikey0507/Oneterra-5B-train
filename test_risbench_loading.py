#!/usr/bin/env python3
import os
import sys
import pickle
import json

# 添加项目路径
sys.path.insert(0, '/mnt_llm_A100_V1/shui/LAE/OneTerra-train/xsam')

base_root = "/mnt_llm_A100_V1/"
data_root = base_root + "shui/oneterra_data/refseg/RISBench"

print("=" * 80)
print("RISBench 数据集检查")
print("=" * 80)

# 1. 检查文件是否存在
refs_file = os.path.join(data_root, "risbench/refs(unc).p")
instances_file = os.path.join(data_root, "risbench/instances.json")
image_dir = os.path.join(data_root, "RISBench_dataset/img_rgb")

print(f"\n1. 文件存在性检查:")
print(f"   refs(unc).p: {os.path.exists(refs_file)} - {refs_file}")
print(f"   instances.json: {os.path.exists(instances_file)} - {instances_file}")
print(f"   图片目录: {os.path.exists(image_dir)} - {image_dir}")

# 2. 检查 refs(unc).p 内容
print(f"\n2. refs(unc).p 内容检查:")
try:
    with open(refs_file, 'rb') as f:
        refs_data = pickle.load(f)
    print(f"   ✅ 成功加载，包含 {len(refs_data)} 条引用数据")
    if len(refs_data) > 0:
        sample_ref = refs_data[0]
        print(f"   样例数据键: {sample_ref.keys()}")
        print(f"   样例 ref_id: {sample_ref.get('ref_id')}")
        print(f"   样例 split: {sample_ref.get('split')}")
        print(f"   样例 sentences 数量: {len(sample_ref.get('sentences', []))}")
        
        # 统计各 split 的数量
        splits = {}
        for ref in refs_data:
            split = ref.get('split', 'unknown')
            splits[split] = splits.get(split, 0) + 1
        print(f"   Split 分布: {splits}")
except Exception as e:
    print(f"   ❌ 加载失败: {e}")

# 3. 检查 instances.json 内容
print(f"\n3. instances.json 内容检查:")
try:
    with open(instances_file, 'r') as f:
        instances_data = json.load(f)
    print(f"   ✅ 成功加载")
    print(f"   images 数量: {len(instances_data.get('images', []))}")
    print(f"   annotations 数量: {len(instances_data.get('annotations', []))}")
    print(f"   categories 数量: {len(instances_data.get('categories', []))}")
    
    if len(instances_data.get('images', [])) > 0:
        sample_img = instances_data['images'][0]
        print(f"   样例图片: {sample_img}")
        
    if len(instances_data.get('annotations', [])) > 0:
        sample_ann = instances_data['annotations'][0]
        print(f"   样例标注键: {sample_ann.keys()}")
        print(f"   样例 segmentation 类型: {type(sample_ann.get('segmentation'))}")
except Exception as e:
    print(f"   ❌ 加载失败: {e}")

# 4. 检查图片文件
print(f"\n4. 图片文件检查:")
try:
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"   图片数量: {len(image_files)}")
    if len(image_files) > 0:
        print(f"   样例图片: {image_files[:3]}")
except Exception as e:
    print(f"   ❌ 读取失败: {e}")

# 5. 测试 REFER 类加载
print(f"\n5. REFER 类加载测试:")
try:
    from xsam.dataset.utils.refer import REFER
    refer_api = REFER(data_root, "risbench")
    print(f"   ✅ REFER 初始化成功")
    print(f"   总图片数: {len(refer_api.Imgs)}")
    print(f"   总引用数: {len(refer_api.Refs)}")
    print(f"   总标注数: {len(refer_api.Anns)}")
    print(f"   总类别数: {len(refer_api.Cats)}")
    
    # 检查 train split
    train_ref_ids = refer_api.getRefIds(split='train')
    print(f"   Train split 引用数: {len(train_ref_ids)}")
    
    print(f"   IMAGE_DIR: {refer_api.IMAGE_DIR}")
    
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
