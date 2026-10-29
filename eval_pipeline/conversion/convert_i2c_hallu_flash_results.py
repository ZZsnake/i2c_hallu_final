#!/usr/bin/env python3
"""Convert Qwen Flash I2C-Hallu predictions into evaluator-ready JSONL."""

import argparse
import glob
import json
import math
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


from prediction_conversion_core import normalize_prediction_record_code_spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert raw I2C-Hallu model predictions to the '
            'sample_id/is_consistent/gold_errors schema used by the evaluator.'
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-file', type=Path, help='Input .xlsx, .xls, or .jsonl file.')
    source.add_argument(
        '--input-glob',
        help='Input glob; the most recently modified matching file is selected.',
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        help='Output JSONL. Defaults to <xlsx-stem>_gen_results.jsonl.',
    )
    parser.add_argument(
        '--candidate-code-dir',
        type=Path,
        help='Local candidate-code directory for prediction files with stale absolute paths.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit non-zero after writing if any record could not be converted.',
    )
    return parser.parse_args()


def resolve_input_file(args: argparse.Namespace) -> Path:
    if args.input_file is not None:
        input_file = args.input_file.expanduser()
    else:
        matches = [Path(path) for path in glob.glob(args.input_glob, recursive=True)]
        files = [path for path in matches if path.is_file()]
        if not files:
            raise FileNotFoundError(f'No input file matched: {args.input_glob}')
        input_file = max(files, key=lambda path: path.stat().st_mtime)
    input_file = input_file.resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    return input_file


def default_output_file(input_file: Path) -> Path:
    if input_file.suffix.lower() in {'.xlsx', '.xls'}:
        return input_file.with_name(input_file.stem + '_gen_results.jsonl')
    suffix = '_eval_ready.jsonl' if input_file.stem.endswith('_gen_results') else '_gen_results.jsonl'
    return input_file.with_name(input_file.stem + suffix)


def load_records(input_file: Path) -> List[Dict[str, Any]]:
    suffix = input_file.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        import pandas as pd

        records = pd.read_excel(input_file).to_dict(orient='records')
    elif suffix == '.jsonl':
        records = []
        with input_file.open('r', encoding='utf-8') as file_obj:
            for line_number, line in enumerate(file_obj, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSONL at {input_file}:{line_number}: {exc}'
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f'Expected a JSON object at {input_file}:{line_number}, '
                        f'got {type(record).__name__}'
                    )
                records.append(record)
    else:
        raise ValueError(f'Unsupported input format: {input_file.suffix}')
    if not records:
        raise ValueError(f'Input contains no records: {input_file}')
    return records


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars and non-finite values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, 'item'):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def convert_records(
    records: Iterable[Dict[str, Any]], candidate_code_dir: Path | None = None
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    converted_records = []
    summary = {
        'record_count': 0,
        'converted_record_count': 0,
        'failed_record_count': 0,
        'json_repaired_record_count': 0,
        'prediction_error_count': 0,
        'code_span_normalized_count': 0,
        'conversion_warning_count': 0,
        'code_span_warning_count': 0,
    }
    for raw_record in records:
        record = normalize_prediction_record_code_spans(
            dict(raw_record), candidate_code_dir=candidate_code_dir
        )
        converted_records.append(json_safe(record))
        summary['record_count'] += 1
        if record.get('prediction_conversion_status') == 'ok':
            summary['converted_record_count'] += 1
            summary['prediction_error_count'] += int(record.get('prediction_error_count') or 0)
        else:
            summary['failed_record_count'] += 1
        summary['json_repaired_record_count'] += int(bool(record.get('prediction_json_repaired')))
        summary['code_span_normalized_count'] += int(
            record.get('code_span_normalized_count') or 0
        )
        summary['conversion_warning_count'] += len(
            record.get('prediction_conversion_warnings') or []
        )
        summary['code_span_warning_count'] += len(
            record.get('code_span_normalization_warnings') or []
        )
    return converted_records, summary


def atomic_write_jsonl(output_file: Path, records: Iterable[Dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=output_file.parent,
            prefix=f'.{output_file.name}.',
            suffix='.tmp',
            delete=False,
        ) as file_obj:
            temp_path = Path(file_obj.name)
            for record in records:
                file_obj.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                    + '\n'
                )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, output_file)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def print_summary(input_file: Path, output_file: Path, summary: Dict[str, int]) -> None:
    print(f'Input file: {input_file}')
    print(f'Exported JSONL: {output_file}')
    print(f'Records: {summary["record_count"]}')
    print(f'Converted records: {summary["converted_record_count"]}')
    print(f'Conversion failures: {summary["failed_record_count"]}')
    print(f'JSON-repaired records: {summary["json_repaired_record_count"]}')
    print(f'Prediction errors total: {summary["prediction_error_count"]}')
    print(f'Code-span corrections total: {summary["code_span_normalized_count"]}')
    print(f'Conversion warnings: {summary["conversion_warning_count"]}')
    print(f'Code-span warnings: {summary["code_span_warning_count"]}')


def main() -> int:
    args = parse_args()
    input_file = resolve_input_file(args)
    output_file = (
        args.output_file.expanduser().resolve()
        if args.output_file is not None
        else default_output_file(input_file)
    )
    records = load_records(input_file)
    converted_records, summary = convert_records(
        records,
        candidate_code_dir=args.candidate_code_dir.expanduser().resolve()
        if args.candidate_code_dir is not None else None,
    )
    atomic_write_jsonl(output_file, converted_records)
    print_summary(input_file, output_file, summary)
    if args.strict and summary['failed_record_count']:
        print(
            'ERROR: one or more predictions could not be converted; inspect '
            'prediction_conversion_error in the output JSONL.',
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
