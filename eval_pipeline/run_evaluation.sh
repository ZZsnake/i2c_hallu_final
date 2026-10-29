#!/usr/bin/env bash
set -euo pipefail

# Usage: run_evaluation.sh PREDICTIONS.jsonl GT.jsonl CANDIDATE_CODE_DIR OUTPUT_DIR [extra evaluator flags]
if (( $# < 4 )); then
  echo "Usage: $0 PREDICTIONS.jsonl GT.jsonl CANDIDATE_CODE_DIR OUTPUT_DIR [extra flags]" >&2
  exit 2
fi
PRED_FILE=$1
GT_FILE=$2
CODE_DIR=$3
OUTPUT_DIR=$4
shift 4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"${PYTHON_BIN:-/root/miniforge3/envs/image2code/bin/python}" "${SCRIPT_DIR}/evaluate_i2c_hallu_localization_half_credit.py" \
  --pred-file "${PRED_FILE}" \
  --gt-file "${GT_FILE}" \
  --candidate-code-dir "${CODE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --restrict-to-gt \
  --run-batch-api \
  --resume \
  --base-url "${BASE_URL:-https://api.deepseek.com}" \
  --model "${MODEL:-deepseek-flash}" \
  --api-key-env DEEPSEEK_API_KEY \
  --disable-thinking \
  --response-format json_object \
  --timeout "${TIMEOUT:-600}" \
  --max-retries "${MAX_RETRIES:-6}" \
  --retry-delay "${RETRY_DELAY:-3}" \
  --schema-retries "${SCHEMA_RETRIES:-3}" \
  --schema-error-feedback \
  --workers "${WORKERS:-16}" \
  "$@"

# Dry-run deliberately creates no summary.
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && exit 0
done
"${PYTHON_BIN:-/root/miniforge3/envs/image2code/bin/python}" - "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding='utf-8'))
selection = summary['input_selection']
judge = summary['judge_status']
span = summary['code_span']
pending = {
    'duplicate_prediction_ids': selection['duplicate_selected_record_id_count'],
    'missing_gt_records': selection['missing_gt_record_count'],
    'missing_set_judgements': judge['set_match_missing_cache_count'],
    'invalid_set_judgements': judge['set_match_invalid_cache_count'],
    'missing_anchor_judgements': span['missing_cache_count'],
    'invalid_anchor_judgements': span['invalid_cache_count'],
}
print(json.dumps({'summary': str(path), 'pending': pending}, ensure_ascii=False))
if any(pending.values()):
    sys.exit('Evaluation is incomplete. Rerun the same command to resume.')
PY
