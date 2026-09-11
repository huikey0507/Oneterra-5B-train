#!/usr/bin/env bash
# 4xA40: joint B∥B2 + SARDET (from S3 + Stage A, not from B2_v2).
#
# Config:
#   xsam/xsam/configs/.../s_sar_align/xsam_sar_B2_joint_sardet_4xA40.py
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash runs/run_B2_joint_sardet_4xA40.sh
#
# Optional:
#   WORK_DIR=... PREV_S3_CKPT=... PREV_SAR_A_CKPT=... MASTER_PORT=29534

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${CODE_DIR:-${ROOT_DIR}/xsam}"
CONFIG_ABS="${ROOT_DIR}/xsam/xsam/configs/xsam/phi3_mini_4k_instruct_siglip2_so400m_p14_384/s_sar_align/xsam_sar_B2_joint_sardet_4xA40.py"

WORK_DIR="${WORK_DIR:-${ROOT_DIR}/wkdrs_sar_align_B2_joint_sardet}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/datas}"
INIT_DIR="${INIT_DIR:-${ROOT_DIR}/inits}"
PREV_S3_CKPT="${PREV_S3_CKPT:-${ROOT_DIR}/checkpoints/s3_mixed_fineture_v3.2/pytorch_model.bin}"
PREV_SAR_A_CKPT="${PREV_SAR_A_CKPT:-${ROOT_DIR}/wkdrs_sar_align_A/iter_91000.pth}"

export ROOT_DIR="${ROOT_DIR}/"
export CODE_DIR="${CODE_DIR}/"
export DATA_DIR="${DATA_DIR}/"
export INIT_DIR="${INIT_DIR}/"
export WORK_DIR="${WORK_DIR}/"
export PREV_S3_CKPT
export PREV_SAR_A_CKPT
export LMUData="${DATA_DIR}/LMUData"
export HF_HOME="${INIT_DIR}/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export XTUNER_DATASET_TIMEOUT=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_GDR_LEVEL=2
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-7200}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-7200000}"
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker0,lo}"
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"

if [[ ! -f "${CONFIG_ABS}" ]]; then
  echo "Config not found: ${CONFIG_ABS}" >&2
  exit 1
fi
if [[ ! -f "${PREV_S3_CKPT}" ]]; then
  echo "S3 checkpoint not found: ${PREV_S3_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${PREV_SAR_A_CKPT}" && ! -d "${PREV_SAR_A_CKPT}" ]]; then
  echo "Stage A checkpoint not found: ${PREV_SAR_A_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${CODE_DIR}/xsam/dataset/sar_ov_seg_dataset.py" ]]; then
  echo "SAR code missing under ${CODE_DIR}/xsam/dataset/" >&2
  exit 1
fi
if [[ ! -f "${CODE_DIR}/xsam/model/modules/sar_cond_adapter.py" ]]; then
  echo "sar_cond_adapter missing under ${CODE_DIR}/xsam/model/modules/" >&2
  exit 1
fi

nproc_per_node="${TQ_GPU_NUM:-${GPU_PER_NODE:-}}"
if [[ -z "${nproc_per_node}" ]]; then
  nproc_per_node=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [[ -z "${nproc_per_node}" || "${nproc_per_node}" -lt 1 ]] && nproc_per_node=4
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible_gpu_count=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
  if [[ "${visible_gpu_count}" -gt 0 && "${visible_gpu_count}" -lt "${nproc_per_node}" ]]; then
    nproc_per_node="${visible_gpu_count}"
  fi
fi

nnodes="${WORLD_SIZE:-1}"
node_rank="${RANK:-0}"
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-29534}"
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"

mkdir -p "${WORK_DIR}"
if [[ "${node_rank}" = "0" ]]; then
  cp -f "$(realpath "$0")" "${WORK_DIR}/"
  cp -f "${CONFIG_ABS}" "${WORK_DIR}/"
  if [[ ! -d "${WORK_DIR}/xsam" ]]; then
    cp -rf "${CODE_DIR}" "${WORK_DIR}/"
    find "${WORK_DIR}/xsam" -name "*.crc" -type f -delete 2>/dev/null || true
  fi
fi

time_tag=$(date "+%Y%m%d-%H%M%S")
echo "[INFO] ROOT=${ROOT_DIR}"
echo "[INFO] CODE=${CODE_DIR}"
echo "[INFO] WORK=${WORK_DIR}"
echo "[INFO] CONFIG=${CONFIG_ABS}"
echo "[INFO] S3=${PREV_S3_CKPT}"
echo "[INFO] A=${PREV_SAR_A_CKPT}"
echo "[INFO] nnodes=${nnodes} nproc=${nproc_per_node} rank=${node_rank} master=${master_addr}:${master_port}"

cd "${ROOT_DIR}"
PYTHONPATH="$(realpath "${CODE_DIR}"):${PYTHONPATH:-}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun \
  --nproc_per_node="${nproc_per_node}" \
  --nnodes="${nnodes}" \
  --node_rank="${node_rank}" \
  --master_addr="${master_addr}" \
  --master_port="${master_port}" \
  "${CODE_DIR}/xsam/tools/train.py" \
  "${CONFIG_ABS}" \
  --work-dir "${WORK_DIR}" \
  --launcher pytorch \
  --deepspeed deepspeed_zero2 \
  --seed 1024 \
  2>&1 | { [[ "${node_rank}" = "0" ]] && tee "${WORK_DIR}/train-${time_tag}.log" || cat; }
