#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then echo "Usage: bash run_eval.sh PREDICTIONS.jsonl [evaluator options]" >&2; exit 2; fi
PRED="$1"
shift
export WORKERS="${EVAL_WORKERS:-${WORKERS:-16}}"
exec bash "${ROOT}/eval_pipeline/run_evaluation.sh" "${PRED}" \
 "${GT_FILE:-${ROOT}/benchmark_update_1015/benchmark.jsonl}" \
 "${ROOT}/benchmark_update_1015/resources/candidate_code" \
 "${EVAL_OUTPUT_DIR:-${ROOT}/eval_pipeline/outputs/code_span_v6_half_credit_20261026/$(basename "${PRED}" .jsonl)}" "$@"
