#!/bin/bash
# 在源服务器上运行此脚本，导出 xsam 环境

echo "📦 导出 xsam 环境..."
echo ""

# 方法1: 使用 conda-pack（推荐，最可靠）
if command -v conda-pack &> /dev/null; then
    echo "使用方法1: conda-pack（推荐）"
    echo "运行以下命令："
    echo "  conda activate xsam"
    echo "  pip install conda-pack"
    echo "  conda-pack -n xsam -o xsam_env.tar.gz"
    echo ""
    echo "然后将 xsam_env.tar.gz 复制到目标服务器，运行："
    echo "  mkdir -p /path/to/conda/envs/xsam"
    echo "  cd /path/to/conda/envs/xsam"
    echo "  tar -xzf /path/to/xsam_env.tar.gz"
    echo "  source bin/activate"
    echo "  conda-unpack"
    echo ""
fi

# 方法2: 导出环境列表
echo "使用方法2: 导出环境配置"
echo "运行以下命令："
echo "  conda activate xsam"
echo "  conda env export > environment.yml"
echo "  conda list --explicit > spec-file.txt"
echo ""
echo "然后在目标服务器上："
echo "  conda env create -f environment.yml"
echo "  或"
echo "  conda create -n xsam --file spec-file.txt"
echo ""

# 方法3: 直接复制目录
echo "使用方法3: 直接复制（如果两台服务器网络好）"
echo "在源服务器上运行："
echo "  conda env list  # 查看 xsam 环境路径"
echo "  tar -czf xsam_env.tar.gz /path/to/xsam/env"
echo ""
echo "在目标服务器上："
echo "  解压到 conda 的 envs 目录"
echo "  修复路径："
echo "    find xsam/bin -type f -exec sed -i 's|/old/path|/new/path|g' {} \\;"

