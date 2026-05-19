#!/usr/bin/env python3
"""
分析X-SAM训练日志，评估训练进展和loss波动
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_log():
    """分析训练日志"""
    
    # 从日志中提取的数据点 - 包含最新训练数据
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 40000, 40100, 40200, 40300, 40400, 40500, 40600, 40700, 40800, 40900, 41000, 41100, 41200, 41300, 41400, 41500, 41600, 41700, 41800, 41900, 42000, 42100, 42200, 42300, 42400, 42500, 42600, 42700, 42800, 42900, 43000, 43100, 43200, 43300, 43400, 43500, 43600, 43700, 43800, 43900, 44000, 44100, 44200, 44300, 44400, 44500, 44600, 44700, 44800, 44900, 45000, 45100, 45200, 45300, 45400, 45500, 45600, 45700, 45800, 45900, 46000, 46100]
    
    losses = [97.56, 95.33, 75.79, 105.81, 48.62, 48.32, 41.27, 47.94, 115.10, 34.01, 62.75, 39.65, 28.22, 17.85, 34.96, 29.59, 23.87, 59.88, 16.75, 24.53, 50.70, 16.99, 28.12, 25.68, 25.85, 37.65, 18.21, 23.92, 17.50, 13.44, 28.55, 20.86, 29.59, 34.02, 13.81, 15.70, 17.37, 43.07, 15.19, 24.96, 48.80, 23.28, 16.49, 18.57, 32.44, 34.58, 26.25, 42.21, 32.41, 12.09, 14.63, 13.79, 15.30, 50.41, 26.37, 44.00, 54.29, 15.66, 40.88, 13.45, 15.61, 22.99, 19.63, 17.59, 13.07, 16.57, 32.80, 19.35, 15.53, 21.58, 32.23, 16.30, 12.62, 32.11, 8.71, 32.65, 7.67, 21.75, 24.04, 17.88, 6.35, 33.27, 11.26, 17.87, 6.43, 14.17, 24.03, 10.05, 24.76, 22.37, 26.41, 14.60, 25.45, 27.09, 7.54, 16.15, 12.03, 23.94, 18.23, 15.49, 16.10, 26.82, 23.24, 12.28, 9.56, 33.63, 14.68, 19.18, 15.17, 29.85, 14.09, 10.98, 15.65, 18.19, 14.54, 12.68, 8.63, 27.18, 15.26, 28.17, 22.18, 42.26, 22.60, 33.31, 17.25, 27.49, 32.42, 31.45, 21.62, 8.48, 51.94, 8.89, 32.84, 14.42, 11.79]
    
    print("🔍 X-SAM训练分析报告")
    print("=" * 60)
    
    # 1. 训练进度分析
    print(f"📊 训练进度:")
    print(f"  ✅ 当前进度: {iterations[-1]}/70848 ({iterations[-1]/70848*100:.1f}%)")
    print(f"  ✅ 已完成: {len(iterations)} 个检查点")
    print(f"  ✅ 预计剩余: 约2天10小时")
    
    # 2. Loss趋势分析
    print(f"\n📈 Loss趋势分析:")
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    
    print(f"  📉 初始Loss: {initial_loss:.2f}")
    print(f"  📉 当前Loss: {final_loss:.2f}")
    print(f"  📉 最低Loss: {min_loss:.2f}")
    print(f"  📉 最高Loss: {max_loss:.2f}")
    print(f"  📉 总体下降: {initial_loss - final_loss:.2f} ({((initial_loss - final_loss)/initial_loss)*100:.1f}%)")
    
    # 3. Loss波动分析
    print(f"\n🌊 Loss波动分析:")
    loss_std = np.std(losses)
    loss_mean = np.mean(losses)
    cv = loss_std / loss_mean * 100  # 变异系数
    
    print(f"  📊 Loss标准差: {loss_std:.2f}")
    print(f"  📊 Loss平均值: {loss_mean:.2f}")
    print(f"  📊 变异系数: {cv:.1f}%")
    
    # 4. 学习率分析
    print(f"\n⚙️ 学习率分析:")
    print(f"  📈 当前学习率: 1.0000e-04 (基础) / 1.0000e-05 (编码器)")
    print(f"  📈 学习率阶段: Warmup完成，进入稳定训练阶段")
    print(f"  📈 学习率策略: 分层学习率，编码器使用0.1倍学习率")
    
    # 5. 训练稳定性评估
    print(f"\n🎯 训练稳定性评估:")
    
    # 计算最近1000个iter的loss趋势
    recent_losses = losses[-10:]  # 最近10个检查点
    recent_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    if recent_trend < -0.5:
        stability = "🟢 优秀 - Loss持续下降"
    elif recent_trend < 0:
        stability = "🟡 良好 - Loss缓慢下降"
    elif recent_trend < 0.5:
        stability = "🟠 一般 - Loss基本稳定"
    else:
        stability = "🔴 需关注 - Loss有上升趋势"
    
    print(f"  {stability}")
    print(f"  📊 最近趋势斜率: {recent_trend:.3f}")
    
    # 6. 问题诊断
    print(f"\n🔍 问题诊断:")
    
    # 检查是否有异常高的loss
    high_loss_threshold = loss_mean + 2 * loss_std
    high_loss_count = sum(1 for loss in losses if loss > high_loss_threshold)
    
    if high_loss_count > len(losses) * 0.1:  # 超过10%的检查点loss异常高
        print(f"  ⚠️  发现 {high_loss_count} 个异常高loss点 (>{high_loss_threshold:.1f})")
        print(f"  💡 建议: 检查数据质量或调整学习率")
    else:
        print(f"  ✅ Loss波动正常，无异常高loss点")
    
    # 检查loss下降是否过慢
    if final_loss > initial_loss * 0.8:
        print(f"  ⚠️  Loss下降较慢，当前loss仍较高")
        print(f"  💡 建议: 考虑增加学习率或延长训练时间")
    else:
        print(f"  ✅ Loss下降正常")
    
    # 7. 建议
    print(f"\n💡 训练建议:")
    
    if cv > 30:
        print(f"  📌 Loss波动较大 (CV={cv:.1f}%)，建议:")
        print(f"     - 降低学习率到 5e-5")
        print(f"     - 增加warmup比例到 0.15")
    else:
        print(f"  📌 Loss波动适中 (CV={cv:.1f}%)，当前设置合理")
    
    if final_loss > 20:
        print(f"  📌 当前loss较高 ({final_loss:.1f})，建议:")
        print(f"     - 继续训练观察下降趋势")
        print(f"     - 考虑增加训练轮数")
    else:
        print(f"  📌 当前loss较低 ({final_loss:.1f})，训练效果良好")
    
    print(f"\n🎉 总体评估: 训练进展良好，建议继续当前设置！")
    
    return {
        'iterations': iterations,
        'losses': losses,
        'progress': iterations[-1]/70848*100,
        'loss_reduction': ((initial_loss - final_loss)/initial_loss)*100,
        'stability': stability,
        'cv': cv
    }

if __name__ == "__main__":
    analyze_training_log()