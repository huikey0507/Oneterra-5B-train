#!/usr/bin/env python3
"""
测试X-SAM训练配置的完整性
"""

import sys
import os
sys.path.insert(0, '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/xsam')

def test_training_config():
    """测试训练配置"""
    try:
        from mmengine.config import Config
        
        # 加载配置文件
        config_path = '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s1_seg_finetune/xsam_sota_s1_finetune.py'
        cfg = Config.fromfile(config_path)
        
        print("🔍 测试X-SAM训练配置...")
        print("=" * 60)
        
        # 1. 检查基本配置
        print("📋 基本配置:")
        print(f"  ✅ 模型类型: {cfg.model.type}")
        print(f"  ✅ 训练轮数: {cfg.max_epochs}")
        print(f"  ✅ 批次大小: {cfg.batch_size}")
        print(f"  ✅ 累积步数: {cfg.accumulative_counts}")
        print(f"  ✅ 学习率: {cfg.lr}")
        
        # 2. 检查预训练权重
        print("\n🔗 预训练权重:")
        pretrained_path = cfg.model.s1_pretrained_pth
        print(f"  ✅ 预训练权重路径: {pretrained_path}")
        if os.path.exists(pretrained_path):
            file_size = os.path.getsize(pretrained_path) / (1024*1024)  # MB
            print(f"  ✅ 文件存在，大小: {file_size:.1f} MB")
        else:
            print(f"  ❌ 文件不存在: {pretrained_path}")
            return False
            
        # 3. 检查模型配置
        print("\n🏗️ 模型配置:")
        print(f"  ✅ 类别数量: {cfg.model.segmentor.decoder.config.num_labels}")
        print(f"  ✅ 编码器: {cfg.model.segmentor.encoder.type}")
        print(f"  ✅ 解码器: {cfg.model.segmentor.decoder.type}")
        print(f"  ✅ 重新初始化解码器: {cfg.model.segmentor.reinit_decoder}")
        print(f"  ✅ 关闭分类器: {cfg.model.segmentor.close_cls}")
        
        # 4. 检查数据配置
        print("\n📊 数据配置:")
        print(f"  ✅ 数据根目录: {cfg.data_root}")
        print(f"  ✅ 训练数据路径: {cfg.train_data_path}")
        print(f"  ✅ 图像文件夹: {cfg.train_image_folder}")
        print(f"  ✅ 标签文件夹: {cfg.train_panseg_map_folder}")
        
        # 检查数据文件是否存在
        if os.path.exists(cfg.train_data_path):
            print(f"  ✅ 训练标注文件存在")
        else:
            print(f"  ❌ 训练标注文件不存在: {cfg.train_data_path}")
            
        if os.path.exists(cfg.train_image_folder):
            image_count = len([f for f in os.listdir(cfg.train_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
            print(f"  ✅ 训练图像文件夹存在，包含 {image_count} 个图像文件")
        else:
            print(f"  ❌ 训练图像文件夹不存在: {cfg.train_image_folder}")
            
        if os.path.exists(cfg.train_panseg_map_folder):
            label_count = len([f for f in os.listdir(cfg.train_panseg_map_folder) if f.lower().endswith('.png')])
            print(f"  ✅ 训练标签文件夹存在，包含 {label_count} 个标签文件")
        else:
            print(f"  ❌ 训练标签文件夹不存在: {cfg.train_panseg_map_folder}")
        
        # 5. 检查优化器配置
        print("\n⚙️ 优化器配置:")
        print(f"  ✅ 优化器类型: {cfg.optim_wrapper.optimizer.type}")
        print(f"  ✅ 学习率: {cfg.optim_wrapper.optimizer.lr}")
        print(f"  ✅ 权重衰减: {cfg.optim_wrapper.optimizer.weight_decay}")
        print(f"  ✅ 梯度裁剪: {cfg.optim_wrapper.clip_grad.max_norm}")
        
        # 6. 检查学习率调度器
        print("\n📈 学习率调度器:")
        print(f"  ✅ 最大轮数: {cfg.max_epochs}")
        print(f"  ✅ Warmup比例: {cfg.warmup_ratio}")
        print(f"  ✅ Milestones: {cfg.param_scheduler[1].milestones}")
        
        # 7. 检查保存配置
        print("\n💾 保存配置:")
        print(f"  ✅ 保存步数: {cfg.save_steps}")
        print(f"  ✅ 最大保存数量: {cfg.save_total_limit}")
        print(f"  ✅ 工作目录: {cfg.work_dir}")
        
        print("\n" + "=" * 60)
        print("🎉 配置检查完成！")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_config()
    if success:
        print("✅ 训练配置验证通过！可以开始训练。")
    else:
        print("❌ 训练配置验证失败！请检查配置。")