#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 skyscript.json 文件中的 QA 对数量
"""
import json
import sys

def count_qa_pairs(json_file_path):
    """
    统计 JSON 文件中的 QA 对数量
    
    Args:
        json_file_path: JSON 文件路径
        
    Returns:
        QA 对的数量
    """
    print(f"正在读取文件: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式
        if isinstance(data, list):
            count = len(data)
            print(f"文件包含 {count} 条 QA 对")
            return count
        elif isinstance(data, dict):
            # 如果是字典，可能需要检查特定键
            print("文件是字典格式，检查是否有 'data' 或 'items' 键...")
            if 'data' in data and isinstance(data['data'], list):
                count = len(data['data'])
                print(f"文件包含 {count} 条 QA 对")
                return count
            else:
                print("无法确定 QA 对的数量，文件结构可能不同")
                return None
        else:
            print(f"未知的数据格式: {type(data)}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

if __name__ == "__main__":
    json_file = "/mnt_llm_A100_V1/shui/LAE/OneTerra-train/datas/img_conv_data/skyscript/skyscript.json"
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    count = count_qa_pairs(json_file)
    
    if count is not None:
        print(f"\n结果: {count} 条 QA 对")
        sys.exit(0)
    else:
        print("\n统计失败")
        sys.exit(1)




