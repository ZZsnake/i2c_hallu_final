#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROMPT_VERSION=v8
export I2C_HALLU_V8_DUAL_IMAGE_PROMPT_PATH="${KIT_ROOT:-${SCRIPT_DIR}}/../prompt_v8_refined_dual_image.py"
cd "${KIT_ROOT}"
eval "$(/root/miniforge3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-image2code}"
export PYTHONUNBUFFERED=1
python "${KIT_ROOT}/../build_benchmark_update_1015_dual_image_tsv.py"
WORK_DIR="${WORK_DIR:-${KIT_ROOT}/outputs/benchmark_update_1015_${PROMPT_VERSION}/gpt-6-sol_api_dual_image}"
LOG_FILE="${LOG_FILE:-${KIT_ROOT}/log/i2c_hallu_update_1015_${PROMPT_VERSION}_gpt6_sol_api_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1
ARGS=(
    --dataset "I2C_Hallu_Benchmark_Update_1015_${PROMPT_VERSION^^}_Dual_Image"
    --model gpt-6-sol
    --base-url "${API_BASE_URL:-https://www.micuapi.ai/v1}"
    --api-key-env MICU_API_KEY
    --workers "${API_WORKERS:-16}"
    --work-dir "${WORK_DIR}"
    --timeout 180
    --max-retries 2
    --retry-delay 3
    --max-tokens 16384
    --token-field max_completion_tokens
    --resume
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--validate-only)
fi
exec python "${SCRIPT_DIR}/run_i2c_hallu_dual_image_api.py" "${ARGS[@]}" "$@"
