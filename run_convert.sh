#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-/root/miniforge3/envs/image2code/bin/python}" "${ROOT}/eval_pipeline/conversion/convert_i2c_hallu_flash_results.py" \
 --candidate-code-dir "${ROOT}/benchmark_update_1015/resources/candidate_code" --strict "$@"
