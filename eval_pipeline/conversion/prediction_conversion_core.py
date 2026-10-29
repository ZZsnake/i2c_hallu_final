"""Portable extraction of the final I2C-Hallu prediction conversion logic.

Source: data_pipline/VLMEvalKit/vlmeval/dataset/i2c_hallu.py (2026-10-29).
Only prediction parsing, code-span normalization, and local path resolution
are copied; inference framework dependencies are intentionally absent.
"""
import json
import re
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]

def resolve_candidate_code_path(value, candidate_code_dir=None):
    path = Path(str(value))
    resolved = path if path.is_absolute() else BENCHMARK_ROOT / path
    if resolved.is_file() or candidate_code_dir is None:
        return resolved
    # Prediction files may contain absolute paths from the inference machine.
    # The benchmark release keeps the same content-addressed basename locally.
    return Path(candidate_code_dir) / path.name



def _prediction_json_candidates(text):
    """Return de-duplicated strict/embedded JSON candidates in priority order."""
    text = text.lstrip('\ufeff').strip()
    candidates = [text]

    # Qwen-compatible endpoints may emit a closing thinking marker even when
    # thinking was disabled, or wrap the final answer in a Markdown fence.
    without_thinking = re.sub(r'^\s*</think>\s*', '', text, flags=re.IGNORECASE)
    without_thinking = re.sub(
        r'^\s*<think>.*?</think>\s*',
        '',
        without_thinking,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates.append(without_thinking)
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r'```(?:json)?\s*(.*?)```',
            without_thinking,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    # Include the widest object-shaped substring for responses containing
    # prose before or after the final JSON object.
    for candidate in list(candidates):
        start = candidate.find('{')
        end = candidate.rfind('}')
        if start >= 0 and end > start:
            candidates.append(candidate[start:end + 1])

    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _select_prediction_object(objects):
    """Prefer an object carrying the expected prediction error array."""
    for result in objects:
        if isinstance(result, dict) and (
            isinstance(result.get('errors'), list)
            or isinstance(result.get('gold_errors'), list)
        ):
            return result
    for result in objects:
        if isinstance(result, dict):
            return result
    return None


def parse_prediction_json_with_metadata(value):
    """Defensively parse a model prediction and report any lossy repair.

    Parsing is attempted in this order: strict JSON, embedded/fenced strict
    JSON, then ``json_repair`` over object-shaped candidates. The repair
    fallback is deliberately last so valid model output is never rewritten.
    """
    if isinstance(value, dict):
        return json.loads(json.dumps(value, ensure_ascii=False)), {
            'method': 'object',
            'repaired': False,
            'warnings': [],
        }
    if not isinstance(value, str) or not value.strip():
        raise ValueError('prediction is empty or is not a JSON string')

    candidates = _prediction_json_candidates(value)
    strict_objects = []
    strict_errors = []
    for candidate in candidates:
        try:
            strict_objects.append(json.loads(candidate))
        except (TypeError, ValueError) as exc:
            strict_errors.append(str(exc))
    result = _select_prediction_object(strict_objects)
    if result is not None:
        embedded = candidates.index(next(
            candidate for candidate in candidates
            if _json_candidate_equals(candidate, result)
        )) > 0
        warnings = ['prediction JSON was extracted from a wrapper'] if embedded else []
        return result, {
            'method': 'embedded_json' if embedded else 'strict_json',
            'repaired': False,
            'warnings': warnings,
        }

    try:
        from json_repair import repair_json
    except ImportError as exc:
        detail = strict_errors[0] if strict_errors else 'no JSON object candidate found'
        raise ValueError(
            f'prediction is not valid JSON ({detail}); install json-repair for defensive recovery'
        ) from exc

    repaired_objects = []
    for candidate in reversed(candidates):
        if '{' not in candidate:
            continue
        try:
            repaired = repair_json(candidate, return_objects=True)
        except Exception:
            continue
        if isinstance(repaired, dict):
            repaired_objects.append(repaired)
    result = _select_prediction_object(repaired_objects)
    if result is None:
        detail = strict_errors[0] if strict_errors else 'no JSON object candidate found'
        raise ValueError(f'unable to recover prediction JSON: {detail}')
    return result, {
        'method': 'json_repair',
        'repaired': True,
        'warnings': ['prediction JSON required json_repair recovery'],
    }


def _json_candidate_equals(candidate, expected):
    """Return whether a candidate strictly decodes to the selected object."""
    try:
        return json.loads(candidate) == expected
    except (TypeError, ValueError):
        return False


def parse_prediction_json(value):
    """Backward-compatible parser returning only the prediction object."""
    prediction, _metadata = parse_prediction_json_with_metadata(value)
    return prediction


def externalize_prediction_fields(prediction, sample_id):
    """Add deterministic record fields that the model does not need to generate."""
    if 'errors' in prediction:
        errors = prediction['errors']
    elif 'gold_errors' in prediction:
        # Keep old/checkpointed V6 responses readable during migration.
        errors = prediction['gold_errors']
    else:
        raise ValueError('prediction.errors is missing')
    if not isinstance(errors, list):
        raise ValueError('prediction.errors is not a JSON array')

    gold_errors = []
    for error_index, raw_error in enumerate(errors, 1):
        if not isinstance(raw_error, dict):
            raise ValueError(f'prediction.errors[{error_index - 1}] is not a JSON object')
        error = dict(raw_error)
        error.pop('error_id', None)
        gold_errors.append({'error_id': f'E{error_index}', **error})

    return {
        'sample_id': str(sample_id),
        'is_consistent': not gold_errors,
        'gold_errors': gold_errors,
    }


def anchor_line_ranges(candidate_code, anchor):
    """Return all exact 1-based inclusive source ranges occupied by anchor."""
    ranges = []
    position = candidate_code.find(anchor)
    while position >= 0:
        start_line = candidate_code.count('\n', 0, position) + 1
        end_line = start_line + anchor.count('\n')
        ranges.append((start_line, end_line))
        position = candidate_code.find(anchor, position + 1)
    return sorted(set(ranges))


def normalize_prediction_code_spans(prediction, candidate_code):
    """Keep valid model line ranges; repair only unambiguous invalid ranges.

    An anchor may be a short substring of a larger causal code block. Its
    occurrence therefore validates a declared range; it does not define the
    entire range to score.
    """
    normalized_count = 0
    warnings = []
    line_count = len(candidate_code.splitlines())
    errors = prediction.get('gold_errors')
    if not isinstance(errors, list):
        return prediction, normalized_count, ['prediction.gold_errors is not a JSON array']

    for error_index, error in enumerate(errors):
        if not isinstance(error, dict):
            warnings.append(f'gold_errors[{error_index}] is not a JSON object')
            continue
        error_id = str(error.get('error_id') or f'index_{error_index}')
        code_span = error.get('code_span')
        if code_span is None:
            continue
        if not isinstance(code_span, list):
            warnings.append(f'{error_id}.code_span is not a JSON array')
            continue

        for marker_index, marker in enumerate(code_span):
            prefix = f'{error_id}.code_span[{marker_index}]'
            if not isinstance(marker, dict):
                warnings.append(f'{prefix} is not a JSON object')
                continue
            anchor = marker.get('anchor')
            if not isinstance(anchor, str) or not anchor:
                warnings.append(f'{prefix}.anchor is empty or invalid')
                continue
            ranges = anchor_line_ranges(candidate_code, anchor)
            if not ranges:
                warnings.append(f'{prefix}.anchor is not a verbatim substring of candidate code')
                continue

            declared = (marker.get('start_line'), marker.get('end_line'))
            valid_declared = (
                all(type(line) is int for line in declared)
                and 1 <= declared[0] <= declared[1] <= line_count
            )
            if valid_declared and any(
                declared[0] <= start_line and end_line <= declared[1]
                for start_line, end_line in ranges
            ):
                continue
            if len(ranges) == 1:
                marker['start_line'], marker['end_line'] = ranges[0]
                normalized_count += 1
                continue
            warnings.append(
                f'{prefix}.declared range does not contain anchor: declared={declared}, '
                f'exact_ranges={ranges}; multiple repairs are possible, so no repair was applied'
            )

    return prediction, normalized_count, warnings


def normalize_prediction_record_code_spans(record, candidate_code_dir=None):
    """Combine request ID with model errors, then normalize one output record."""
    record.setdefault('execution_success', None)
    sample_id = record.get('sample_id')
    if sample_id is None:
        sample_id = record.get('index')
    if sample_id is not None:
        record['sample_id'] = str(sample_id)
    # Reuse the original model response when converting an already-normalized
    # JSONL file, making the operation idempotent and preserving repair counts.
    raw_prediction = record.get('prediction_raw')
    if raw_prediction is None:
        raw_prediction = record.get('prediction')
    record['prediction_raw'] = raw_prediction
    try:
        if sample_id is None:
            raise ValueError('record has neither sample_id nor index')
        if 'candidate_code_path' not in record:
            raise ValueError('record is missing candidate_code_path')
        code_path = resolve_candidate_code_path(
            record['candidate_code_path'], candidate_code_dir
        )
        if not code_path.is_file():
            raise FileNotFoundError(f'candidate code not found: {code_path}')
        candidate_code = code_path.read_text(encoding='utf-8', errors='replace')
        prediction, parse_metadata = parse_prediction_json_with_metadata(raw_prediction)
        prediction = externalize_prediction_fields(prediction, sample_id)
        prediction, normalized_count, warnings = normalize_prediction_code_spans(
            prediction, candidate_code
        )
        record['prediction'] = json.dumps(prediction, ensure_ascii=False, separators=(',', ':'))
        record['prediction_conversion_status'] = 'ok'
        record['prediction_conversion_error'] = None
        record['prediction_conversion_method'] = parse_metadata['method']
        record['prediction_json_repaired'] = parse_metadata['repaired']
        record['prediction_conversion_warnings'] = parse_metadata['warnings']
        record['prediction_error_count'] = len(prediction['gold_errors'])
        record['code_span_normalized_count'] = normalized_count
        record['code_span_normalization_warnings'] = warnings
    except Exception as exc:
        record['prediction_conversion_status'] = 'error'
        record['prediction_conversion_error'] = f'{type(exc).__name__}: {exc}'
        record['prediction_conversion_method'] = None
        record['prediction_json_repaired'] = False
        record['prediction_conversion_warnings'] = []
        record['prediction_error_count'] = None
        record['code_span_normalized_count'] = 0
        record['code_span_normalization_warnings'] = []
    return record


