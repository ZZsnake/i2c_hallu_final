#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/data_pipline/VLMEvalKit/dual_infer_scripts/run_i2c_hallu_update_1015_v8_refined_local.sh" "$@"
