#!/usr/bin/env bash
# Run V8 refined GT + candidate dual-image inference with Qwen3.5-27B checkpoint-1320 on four GPUs with a budget based on current free memory.
# Validate without loading model weights: DRY_RUN=1 bash "$0"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep dataset, runtime, output and log paths relative to the VLMEvalKit root.
if [[ "$(basename "${SCRIPT_DIR}")" == "dual_infer_scripts" ]]; then
    SCRIPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROMPT_VERSION=v8
export I2C_HALLU_V8_DUAL_IMAGE_PROMPT_PATH="${KIT_ROOT:-${SCRIPT_DIR}}/../prompt_v8_refined_dual_image.py"
DATASET_NAME="I2C_Hallu_Benchmark_Update_1015_${PROMPT_VERSION^^}_Dual_Image"
DATA_TSV="${PROJECT_ROOT}/data_pipline/i2c_hallu_benchmark_update_1015_dual_image.tsv"
RESOURCE_DIR="${PROJECT_ROOT}/benchmark_update_1015"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/disk1/data_group/zhoujiawei/i2c_hallu_train/training_outputs/qwen38_27b_full_merged_gpt6sol_20261004_32k_packed_gpu0_gpu1_gpu2_gpu3/v0-20261006-215412/checkpoint-1320}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
[[ "${GPU_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || { echo "GPU_IDS must contain four comma-separated GPU indices" >&2; exit 1; }
IFS=, read -r -a GPU_ARRAY <<< "${GPU_IDS}"
for (( i=0; i<4; i++ )); do
    for (( j=i+1; j<4; j++ )); do
        [[ "${GPU_ARRAY[i]}" != "${GPU_ARRAY[j]}" ]] || { echo "Require four distinct GPU IDs" >&2; exit 1; }
    done
done
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-32768}"
GPU_MEMORY_RESERVE_MIB="${GPU_MEMORY_RESERVE_MIB:-8192}"
GPU_UTILIZATION_CAP="${VLLM_GPU_UTILIZATION_CAP:-0.30}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/outputs/benchmark_update_1015_${PROMPT_VERSION}/local}"
MAX_NEW_TOKENS="${I2C_MAX_NEW_TOKENS:-16384}"
MAX_MODEL_LEN="${I2C_MAX_MODEL_LEN:-32768}"
LOCAL_BATCH_SIZE="${VLMEVAL_LOCAL_BATCH_SIZE:-4}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
GPU_UTILIZATION="${VLLM_GPU_UTILIZATION:-auto}"
MODEL_NAME="${I2C_MODEL_NAME:-qwen35_27b_merged_gpt6sol_ckpt1320_dual_image_tp4_${PROMPT_VERSION}}"
GDN_PREFILL_BACKEND="${I2C_GDN_PREFILL_BACKEND-triton}"
CONFIG_PATH="${WORK_DIR}/configs/${MODEL_NAME}_vllm.json"
MEMORY_PLAN_PATH="${WORK_DIR}/configs/${MODEL_NAME}_gpu_memory.json"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/log/i2c_hallu_update_1015_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log}"

cd "${SCRIPT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")" "${WORK_DIR}/configs"
exec > >(tee -a "${LOG_FILE}") 2>&1

eval "$(/root/miniforge3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-image2code}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python "${PROJECT_ROOT}/data_pipline/build_benchmark_update_1015_dual_image_tsv.py"

[[ -f "${DATA_TSV}" ]] || { echo "Missing TSV: ${DATA_TSV}" >&2; exit 1; }
[[ -d "${RESOURCE_DIR}" ]] || { echo "Missing resources: ${RESOURCE_DIR}" >&2; exit 1; }
[[ -f "${CHECKPOINT_PATH}/config.json" ]] || { echo "Invalid checkpoint: ${CHECKPOINT_PATH}" >&2; exit 1; }
[[ -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]] || {
    echo "Checkpoint weights are incomplete: ${CHECKPOINT_PATH}" >&2
    exit 1
}

python "${SCRIPT_DIR}/dual_infer_scripts/preflight_i2c_hallu_local_model.py" "${CHECKPOINT_PATH}"

# vLLM's utilization is a fraction of TOTAL memory, not free memory.
# Use the tightest GPU budget so every tensor-parallel worker has headroom.
GPU_UTILIZATION="$(python - "${GPU_IDS}" "${MIN_FREE_GPU_MIB}" \
    "${GPU_MEMORY_RESERVE_MIB}" "${GPU_UTILIZATION_CAP}" \
    "${GPU_UTILIZATION}" "${MEMORY_PLAN_PATH}" <<'PY_GPU'
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import json
from pathlib import Path
import subprocess
import sys

ids, minimum, reserve, cap, requested, plan_path = sys.argv[1:]
minimum, reserve = int(minimum), int(reserve)
cap = Decimal(cap)
if minimum <= 0 or reserve < 0 or not 0 < cap < 1:
    raise ValueError('Invalid minimum memory, reserve, or utilization cap')
rows = []
for gpu_id in ids.split(','):
    result = subprocess.run([
        'nvidia-smi', '-i', gpu_id,
        '--query-gpu=index,memory.total,memory.free',
        '--format=csv,noheader,nounits',
    ], check=True, text=True, capture_output=True)
    index, total, free = [int(value.strip()) for value in result.stdout.strip().split(',')]
    if free < minimum or free <= reserve:
        raise ValueError(f'GPU {index}: free={free} MiB; need at least {minimum} MiB and reserve={reserve} MiB')
    rows.append({'gpu_id': index, 'total_mib': total, 'free_mib': free})
safe = min(Decimal(row['free_mib'] - reserve) / Decimal(row['total_mib']) for row in rows)
if requested == 'auto':
    utilization = min(safe, cap).quantize(Decimal('0.001'), rounding=ROUND_DOWN)
else:
    utilization = Decimal(requested)
    if not utilization.is_finite() or not 0 < utilization < 1 or utilization > safe:
        raise ValueError(f'Requested utilization={requested} exceeds safe current budget={safe:.4f}')
if utilization <= 0:
    raise ValueError('No usable GPU memory budget')
for row in rows:
    row['vllm_budget_mib'] = int(utilization * row['total_mib'])
    row['remaining_headroom_mib'] = row['free_mib'] - row['vllm_budget_mib']
    print(f"GPU {row['gpu_id']}: free={row['free_mib']} MiB; vLLM budget={row['vllm_budget_mib']} MiB; headroom={row['remaining_headroom_mib']} MiB", file=sys.stderr)
plan = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'tensor_parallel_size': 4, 'reserve_mib': reserve,
        'gpu_memory_utilization': float(utilization), 'gpus': rows}
Path(plan_path).write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
print(utilization)
PY_GPU
)"

python - "${CONFIG_PATH}" "${MODEL_NAME}" "${CHECKPOINT_PATH}" "${DATASET_NAME}" \
    "${MAX_NEW_TOKENS}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${GPU_UTILIZATION}" "${GDN_PREFILL_BACKEND}" <<'PY'
import json
import sys
from pathlib import Path

config_path, model_name, checkpoint_path, dataset_name, max_new_tokens, max_model_len, max_num_seqs, gpu_utilization, gdn_prefill_backend = sys.argv[1:]
config = {
    "model": {
        model_name: {
            "class": "Qwen3VLChat",
            "model_path": checkpoint_path,
            "use_custom_prompt": False,
            "use_vllm": True,
            "enable_thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "max_new_tokens": int(max_new_tokens),
            "max_model_len": int(max_model_len),
            "max_num_seqs": int(max_num_seqs),
            "max_image_inputs": 2,
            "gpu_utils": float(gpu_utilization),
        }
    },
    "data": {dataset_name: {"class": "I2CHallu", "dataset": dataset_name}},
}
if gdn_prefill_backend:
    config['model'][model_name]['vllm_additional_config'] = {'gdn_prefill_backend': gdn_prefill_backend}
Path(config_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

echo "Dataset: ${DATASET_NAME}"
echo "TSV: ${DATA_TSV}"
echo "Prompt: ${I2C_HALLU_V8_DUAL_IMAGE_PROMPT_PATH}"
echo "Input order: GT image, candidate rendered image, ${PROMPT_VERSION^^} text with candidate code lines"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Work directory: ${WORK_DIR}"
echo "Target physical GPUs: ${GPU_IDS}; tensor parallel size: 4"
echo "Batch size / max sequences: ${LOCAL_BATCH_SIZE} / ${MAX_NUM_SEQS}"
echo "GPU memory utilization: ${GPU_UTILIZATION}; reserved headroom per GPU: ${GPU_MEMORY_RESERVE_MIB} MiB"
echo "GPU memory plan: ${MEMORY_PLAN_PATH}"


if [[ "${DRY_RUN:-0}" == "1" ]]; then
    python "${SCRIPT_DIR}/dual_infer_scripts/validate_i2c_hallu_update_1015_dual_image.py" "${PROMPT_VERSION}"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
unset MASTER_ADDR MASTER_PORT
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLMEVAL_LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}"
export PRED_FORMAT=xlsx
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
ulimit -n 65535 || true

python run_i2c_hallu_qwen35_compat.py \
    --config "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --mode infer \
    --reuse \
    --reuse-aux infer \
    --verbose

python - "${CONFIG_PATH}" "${WORK_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
from vlmeval.dataset.i2c_hallu import normalize_prediction_record_code_spans

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
work_dir = Path(sys.argv[2])
model_name = next(iter(config['model']))
dataset_name = next(iter(config['data']))
files = list((work_dir / model_name).glob(f'T*/{model_name}_{dataset_name}.xlsx'))
if not files:
    raise FileNotFoundError(f'No prediction XLSX found for {model_name} / {dataset_name}')
src = max(files, key=lambda path: path.stat().st_mtime)
dst = src.with_name(src.stem + '_gen_results.jsonl')
frame = pd.read_excel(src)
if len(frame) != 1040 or frame['index'].nunique() != 1040:
    raise ValueError(f'Incomplete predictions: rows={len(frame)}; expected 1040 unique records')
invalid = 0
with dst.open('w', encoding='utf-8') as output:
    for record in frame.to_dict(orient='records'):
        normalized = normalize_prediction_record_code_spans(record)
        invalid += normalized['prediction_conversion_status'] != 'ok'
        output.write(json.dumps(normalized, ensure_ascii=False) + '\n')
print(f'Exported evaluation JSONL: {dst}; invalid prediction JSON records={invalid}')
if invalid:
    raise ValueError(f'{invalid} records require prediction review; raw responses preserved in JSONL')
PY

echo "Inference and JSONL export completed successfully."
