#!/bin/bash

# X-SAM Docker镜像构建脚本
# 使用方法: ./build_docker.sh [镜像名称] [标签]

set -e

# 默认镜像名称和标签
IMAGE_NAME=${1:-"xsam"}
TAG=${2:-"cuda12.1"}

# 完整镜像名称
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "=========================================="
echo "构建X-SAM Docker镜像"
echo "镜像名称: ${FULL_IMAGE_NAME}"
echo "=========================================="

# 检查基础镜像是否存在
echo "检查基础镜像..."
if ! docker images | grep -q "ubuntu22.04-cuda12.1-conda.*diffuser"; then
    echo "警告: 未找到基础镜像 ubuntu22.04-cuda12.1-conda:diffuser"
    echo "请确保基础镜像已存在"
    exit 1
fi

# 导出conda环境文件（如果不存在）
if [ ! -f "xsam_environment.yml" ]; then
    echo "导出conda环境文件..."
    conda activate xsam && conda env export --no-builds > xsam_environment.yml
    echo "已生成 xsam_environment.yml"
fi

# 构建Docker镜像
echo "开始构建Docker镜像..."
docker build -t ${FULL_IMAGE_NAME} -f Dockerfile .

echo "=========================================="
echo "镜像构建完成!"
echo "镜像名称: ${FULL_IMAGE_NAME}"
echo ""
echo "运行镜像:"
echo "  docker run --gpus all -it ${FULL_IMAGE_NAME}"
echo ""
echo "或者使用 run_docker.sh 脚本运行"
echo "=========================================="

