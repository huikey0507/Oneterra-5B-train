#!/usr/bin/env bash

set -euo pipefail

LOCAL_DIR_DEFAULT="/mnt_shui/OneTerra-train"
NAS_DIR_DEFAULT="/mnt_llm_A100_V1/shui/LAE/OneTerra-train"

LOCAL_DIR="${LOCAL_DIR_DEFAULT}"
NAS_DIR="${NAS_DIR_DEFAULT}"
DRY_RUN=false
DELETE_EXTRA=false

print_usage() {
  cat <<'EOF'
用法:
  bash sync_to_nas.sh [选项]

选项:
  --dry-run              仅预览同步内容，不真正执行
  --delete               删除 NAS 目标目录中本地不存在的文件
  --local-dir PATH       指定本地项目目录
  --nas-dir PATH         指定 NAS 项目目录
  -h, --help             显示帮助

示例:
  bash sync_to_nas.sh
  bash sync_to_nas.sh --dry-run
  bash sync_to_nas.sh --delete
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --delete)
      DELETE_EXTRA=true
      shift
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --nas-dir)
      NAS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "错误: 未知参数 '$1'"
      echo
      print_usage
      exit 1
      ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "错误: 未找到 rsync，请先安装 rsync"
  exit 1
fi

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "错误: 本地项目目录不存在: ${LOCAL_DIR}"
  exit 1
fi

NAS_PARENT_DIR="$(dirname "${NAS_DIR}")"
if [[ ! -d "${NAS_PARENT_DIR}" ]]; then
  echo "错误: NAS 父目录不存在: ${NAS_PARENT_DIR}"
  exit 1
fi

mkdir -p "${NAS_DIR}"

RSYNC_ARGS=(
  -avh
  --info=progress2
  --exclude=.git
  --exclude=.gitignore
  --exclude=.cursorignore
  --exclude=.vscode
  --exclude=.idea
  --exclude=__pycache__
  --exclude=.pytest_cache
  --exclude=.mypy_cache
  --exclude=checkpoints
  --exclude=datas
  --exclude=inits
  --exclude=work_dirs
  --exclude=wkdrs
  --exclude=wkdrs*
  --exclude=wkdirs
  --exclude=output
  --exclude=outputs
  --exclude=demo_outputs
  --exclude=eval_results
  --exclude=validation_results*
  --exclude=labeled_visualization
  --exclude=labeled_visualization*
  --exclude=training_samples_visualization
  --exclude=test_images
  --exclude=logs
  --exclude=log
  --exclude=nohup.out
  --exclude=*.log
  --exclude=*.out
  --exclude=*.pth
  --exclude=*.pt
  --exclude=*.bin
  --exclude=*.safetensors
  --exclude=*.ckpt
  --exclude=*.png
  --exclude=*.jpg
  --exclude=*.jpeg
  --exclude=*.gif
  --exclude=*.bmp
  --exclude=*.mp4
  --exclude=.gradio
  --exclude=.DS_Store
  --exclude=*.swp
  --exclude=*.swo
  --exclude=*~
)

if [[ "${DRY_RUN}" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

if [[ "${DELETE_EXTRA}" == "true" ]]; then
  RSYNC_ARGS+=(--delete)
fi

echo "=========================================="
echo "同步本地代码到 NAS"
echo "=========================================="
echo "本地目录: ${LOCAL_DIR}"
echo "NAS 目录 : ${NAS_DIR}"
echo "预览模式: ${DRY_RUN}"
echo "删除多余: ${DELETE_EXTRA}"
echo

rsync "${RSYNC_ARGS[@]}" "${LOCAL_DIR}/" "${NAS_DIR}/"

echo
echo "=========================================="
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "预览完成，未实际写入 NAS"
else
  echo "同步完成"
fi
echo "=========================================="
echo "训练目录: ${NAS_DIR}"
