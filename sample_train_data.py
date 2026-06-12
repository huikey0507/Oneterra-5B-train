import json
import random
import os

# ==========================================
# 1. 配置根目录 (与原始环境保持一致)
# ==========================================
base_root = "/mnt_llm_A100_V1/"
data_dir = "./datas/"
oneterra_data_root = base_root + "shui/oneterra_data/"
yangsen_data_root = base_root + "yangsen/datasets/"

fitrs_path = oneterra_data_root + "imgconv/FIT-RS/raw_data/train_data_of_each_individual_task/"

# ==========================================
# 2. 抽样任务矩阵 (仅保留需要抽样的背景字典和文本提纯)
# 注意：Pano全景、RefSeg指代、ReaSeg推理以及打榜集(UCM/NWPU等)全部走原文件全量，不在这里抽样！
# 格式: (输入原始大文件路径, 输出_mini小文件路径, 目标抽样条数, 是否开启GeoChat长文本提纯)
# ==========================================
TASKS = [
    # 🔴 FIT-RS 抽出 17 万大军，充当世界知识字典防遗忘
    (fitrs_path + "train_instruction_complexcompre_708k_cleaned.json", fitrs_path + "complexcompre_mini_30k.json", 30000, False),
    (fitrs_path + "train_instruction_vqa_400k_cleaned.json", fitrs_path + "vqa_mini_30k.json", 30000, False),
    (fitrs_path + "train_instruction_imagecaption_65k_cleaned.json", fitrs_path + "imagecaption_mini_20k.json", 20000, False),
    (fitrs_path + "train_instruction_imageclassification_130k_cleaned.json", fitrs_path + "imageclassification_mini_20k.json", 20000, False),
    (fitrs_path + "train_instruction_multiturn_50k_cleaned.json", fitrs_path + "multiturn_mini_20k.json", 20000, False),
    (fitrs_path + "train_instruction_regioncaption_72k_cleaned.json", fitrs_path + "regioncaption_mini_20k.json", 20000, False),

    # 🟡 SAR 模态特征防遗忘 (抽出 3 万条)
    (yangsen_data_root + "sar_total/sft/train.json", yangsen_data_root + "sar_total/sft/train_mini_30k.json", 30000, False),

    # 🟢 GeoChat 长文本提纯 (开启过滤，只留优质长文本)
    (data_dir + "img_conv_data/geochat/geochat_llava.json", data_dir + "img_conv_data/geochat/geochat_mini_30k_PRO.json", 30000, True),
]

def sample_dataset(input_path, output_path, target_num, is_geochat):
    if not os.path.exists(input_path):
        print(f"❌ 找不到文件跳过: {input_path}")
        return

    print(f"\n⏳ 正在加载: {os.path.basename(input_path)} ...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 解析 JSON 失败: {e}")
        return

    # ==========================================
    # 🌟 特殊逻辑：GeoChat 提取长文本高质量对话
    # ==========================================
    if is_geochat:
        print("   🔍 [激活特权] 启动 GeoChat 专属提纯 (过滤回复字数 < 100 的低质量数据)...")
        high_quality_pool = []
        for item in data:
            conversations = item.get("conversations", [])
            gpt_responses = [turn["value"] for turn in conversations if turn.get("from") == "gpt"]
            if gpt_responses and max(len(resp) for resp in gpt_responses) >= 100:
                high_quality_pool.append(item)
                
        print(f"   🎯 提纯完成: 从 {len(data)} 条中挖出 {len(high_quality_pool)} 条优质长文本数据。")
        data = high_quality_pool

    # ==========================================
    # 常规全局随机抽样逻辑
    # ==========================================
    if isinstance(data, list):
        if len(data) <= target_num:
            print(f"   ↳ 数量不足或刚好 {target_num} 条，全量保留。")
            sampled_data = data
        else:
            print(f"   🎲 全局随机抽取 {target_num} 条 (List格式)...")
            sampled_data = random.sample(data, target_num)
    else:
        print(f"⚠️ 无法识别的格式结构，跳过抽样，直接全量复制。")
        sampled_data = data

    # 写入新文件
    print(f"   💾 正在写入本地硬盘...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f)
        
    print(f"✅ 成功保存至: {os.path.basename(output_path)} (最终条数: {len(sampled_data)})")

if __name__ == "__main__":
    print("="*70)
    print("🚀 开始 X-SAM 背景字典数据集精准抽样...")
    print("="*70)
    
    random.seed(42) # 固定随机种子，保证每次抽样可复现
    
    for src, dst, num, is_geochat in TASKS:
        sample_dataset(src, dst, num, is_geochat)
        
    print("\n" + "="*70)
    print("🎉 字典抽样完成！Pano、RefSeg、ReaSeg 等核心任务将直接使用全量原文件。")
    print("👉 下一步：运行 python validate_train_data.py --verbose 进行校验！")
    print("="*70)