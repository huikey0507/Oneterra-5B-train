#!/bin/bash

# X-SAM Docker容器运行脚本
# 使用方法: ./run_docker.sh [镜像名称] [标签] [其他docker run参数]

set -e

# 默认镜像名称和标签
IMAGE_NAME=${1:-"xsam"}
TAG=${2:-"cuda12.1"}

# 完整镜像名称
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# 检查镜像是否存在
if ! docker images | grep -q "${IMAGE_NAME}.*${TAG}"; then
    echo "错误: 镜像 ${FULL_IMAGE_NAME} 不存在"
    echo "请先运行 ./build_docker.sh 构建镜像"
    exit 1
fi

# 获取当前目录的绝对路径（用于挂载）
CURRENT_DIR=$(pwd)
CONTAINER_NAME="xsam_${TAG}_$(date +%s)"

echo "=========================================="
echo "运行X-SAM Docker容器"
echo "镜像名称: ${FULL_IMAGE_NAME}"
echo "容器名称: ${CONTAINER_NAME}"
echo "挂载目录: ${CURRENT_DIR} -> /workspace/X-SAM"
echo "=========================================="

# 运行Docker容器
# --gpus all: 启用所有GPU
# -it: 交互式终端
# --rm: 退出时自动删除容器
# -v: 挂载当前目录到容器
# --name: 容器名称
docker run --gpus all \
    -it \
    --rm \
    --name ${CONTAINER_NAME} \
    -v ${CURRENT_DIR}:/workspace/X-SAM \
    -v /mnt_llm_A100_V1:/mnt_llm_A100_V1 \
    -w /workspace/X-SAM \
    ${FULL_IMAGE_NAME} \
    ${@:3}

