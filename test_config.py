#!/usr/bin/env python3
"""
测试X-SAM配置是否正确加载预训练权重
"""

import sys
import os
sys.path.insert(0, '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/xsam')

def test_config():
    """测试配置文件"""
    try:
        from mmengine.config import Config
        
        # 加载配置文件
        config_path = '/mnt_llm_A100_V1/shui/LAE/X-SAM/X-SAM/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s1_seg_finetune/xsam_sota_s1_finetune.py'
        cfg = Config.fromfile(config_path)
        
        print("✅ 配置文件加载成功")
        print(f"✅ 模型类型: {cfg.model.type}")
        print(f"✅ 预训练权重路径: {cfg.model.s1_pretrained_pth}")
        print(f"✅ 类别数量: {cfg.model.segmentor.decoder.config.num_labels}")
        
        # 检查预训练权重文件是否存在
        pretrained_path = cfg.model.s1_pretrained_pth
        if os.path.exists(pretrained_path):
            print(f"✅ 预训练权重文件存在: {pretrained_path}")
            file_size = os.path.getsize(pretrained_path) / (1024*1024)  # MB
            print(f"✅ 文件大小: {file_size:.1f} MB")
        else:
            print(f"❌ 预训练权重文件不存在: {pretrained_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 测试X-SAM配置...")
    success = test_config()
    if success:
        print("🎉 配置测试通过！")
    else:
        print("💥 配置测试失败！")