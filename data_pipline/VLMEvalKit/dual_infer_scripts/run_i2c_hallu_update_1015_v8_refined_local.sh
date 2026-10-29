#!/usr/bin/env bash
# Same 1015 V8 checkpoint-1320 TP4 inference settings, using the refined V8 prompt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROMPT_PATH="${KIT_ROOT}/../prompt_v8_refined_dual_image.py"
[[ -f "${PROMPT_PATH}" ]] || { echo "Missing prompt: ${PROMPT_PATH}" >&2; exit 1; }
PROMPT_SHA256="$(sha256sum "${PROMPT_PATH}" | cut -d' ' -f1)"

export PROMPT_VERSION=v8
export I2C_HALLU_V8_DUAL_IMAGE_PROMPT_PATH="${PROMPT_PATH}"
OUTPUT_BASE="${PROMPT_OUTPUT_BASE:-${KIT_ROOT}/outputs}"
export WORK_DIR="${OUTPUT_BASE}/benchmark_update_1015_v8_prompt_${PROMPT_SHA256}/local"
export LOG_FILE="${KIT_ROOT}/log/i2c_hallu_update_1015_v8_prompt_${PROMPT_SHA256:0:12}_qwen35_27b_ckpt1320_tp4_$(date +%Y%m%d_%H%M%S).log"

echo "Prompt: ${PROMPT_PATH} (SHA256: ${PROMPT_SHA256})"
echo "Local output: ${WORK_DIR}"
exec bash "${SCRIPT_DIR}/run_i2c_hallu_update_1015_local_dual_image.sh"
