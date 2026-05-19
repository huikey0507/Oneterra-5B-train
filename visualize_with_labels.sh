#!/bin/bash
# SOTA训练数据可视化脚本 - 带类别标签版本
# 用于运行 visualize_with_labels.py

# 设置脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
NUM_SAMPLES=3
OUTPUT_DIR="labeled_visualization3"
DATA_DIR="datas/sota"
CREATE_LEGEND=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --create_legend)
            CREATE_LEGEND=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --num_samples NUM     要可视化的图像数量 (默认: 3)"
            echo "  --output_dir DIR      输出目录 (默认: labeled_visualization)"
            echo "  --data_dir DIR        数据目录 (默认: datas/sota)"
            echo "  --create_legend       创建类别图例 (默认: 不创建)"
            echo "  -h, --help           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --num_samples 10 --output_dir my_visualization"
            echo "  $0 --data_dir datas/sota --num_samples 5 --create_legend"
            echo "  $0 --output_dir labeled_visualization2 --num_samples 3"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查Python脚本是否存在
if [ ! -f "visualize_with_labels.py" ]; then
    echo "错误: 找不到 visualize_with_labels.py"
    exit 1
fi

# 构建Python命令
PYTHON_CMD="python visualize_with_labels.py \
    --num_samples $NUM_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR"

# 如果设置了创建图例标志，添加该参数
if [ "$CREATE_LEGEND" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --create_legend"
fi

# 运行Python脚本
echo "=========================================="
echo "SOTA训练数据可视化 - 带类别标签版本"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "样本数量: $NUM_SAMPLES"
if [ "$CREATE_LEGEND" = true ]; then
    echo "创建图例: 是"
else
    echo "创建图例: 否"
fi
echo "=========================================="
echo ""

eval $PYTHON_CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 可视化完成！"
    echo "输出目录: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 可视化失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

