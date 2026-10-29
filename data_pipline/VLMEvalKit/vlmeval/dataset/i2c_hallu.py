import json
import os
import os.path as osp
import re
import runpy
from functools import lru_cache
from pathlib import Path

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset


PIPELINE_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parents[4]
PROMPT_V8_DUAL_IMAGE_PATH = PIPELINE_ROOT / 'prompt_v8_refined_dual_image.py'
INPUT_BLOCK_MARKER = '[[INPUT_BLOCK]]'
DUAL_IMAGE_DATASETS = {'I2C_Hallu_Benchmark_Update_1015_V8_Dual_Image'}


BENCHMARK_UPDATE_1015_DUAL_IMAGE_DATA_PATH = str(
    PIPELINE_ROOT / 'i2c_hallu_benchmark_update_1015_dual_image.tsv'
)


def source_lines(candidate_code):
    """Use the same source-line definition as the downstream evaluator."""
    lines = candidate_code.splitlines()
    return lines if lines else ['']


def candidate_code_lines(candidate_code):
    return [
        {'l_n': line_number, 't': text}
        for line_number, text in enumerate(source_lines(candidate_code), 1)
    ]






@lru_cache(maxsize=1)
def load_v8_dual_image_prompt_template():
    prompt_path = Path(os.environ.get('I2C_HALLU_V8_DUAL_IMAGE_PROMPT_PATH', str(PROMPT_V8_DUAL_IMAGE_PATH)))
    namespace = runpy.run_path(str(prompt_path))
    prompt_template = namespace.get('prompt')
    if not isinstance(prompt_template, str) or INPUT_BLOCK_MARKER not in prompt_template:
        raise ValueError(f'Valid V8 dual-image prompt with {INPUT_BLOCK_MARKER} not found: {prompt_path}')
    return prompt_template


def resolve_candidate_code_path(value):
    path = Path(str(value))
    if path.is_absolute():
        return path

    # Historical I2C-Hallu data stores paths relative to the repository root,
    # while the 2026-09-13 benchmark update stores its resources next to the
    # TSV under ``data_pipline/``. Prefer an existing repository-root path to
    # preserve the existing datasets' behavior, then fall back to the pipeline
    # root for the update bundle.
    for root in (BENCHMARK_ROOT, PIPELINE_ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate
    return BENCHMARK_ROOT / path


def localize_resource_paths(data, columns):
    """Resolve local resource columns so the dataset is independent of CWD."""
    data = data.copy()
    for column in columns:
        if column not in data:
            continue
        data[column] = [
            str(resolve_candidate_code_path(value))
            if isinstance(value, str) and value.strip()
            else value
            for value in data[column]
        ]
    return data




def build_dual_image_question(line, version='v8'):
    code_path = resolve_candidate_code_path(line['candidate_code_path'])
    if not code_path.is_file():
        raise FileNotFoundError(f'Candidate code not found: {code_path}')
    candidate_code = code_path.read_text(encoding='utf-8', errors='replace')
    actual_input = {
        'figure_type': str(line.get('figure_type') or line.get('image_type') or 'Unknown'),
        'code_type': str(line.get('code_type') or 'Unknown'),
        'candidate_code_lines': candidate_code_lines(candidate_code),
    }
    if version != 'v8':
        raise ValueError('Only V8 refined is supported')
    template = load_v8_dual_image_prompt_template()
    return template.replace(
        INPUT_BLOCK_MARKER,
        json.dumps(actual_input, ensure_ascii=False, separators=(',', ':')),
        1,
    )




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


def _final_embedded_prediction(value):
    """Accept a complete JSON answer at the end of explanatory text.

    Some local models explain their decision before a final JSON object, or
    return the error array itself in a JSON fence. Never use an intermediate
    object from a truncated response as the final prediction.
    """
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r'\{|\[', value):
        try:
            result, consumed = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        trailing = value[match.start() + consumed:].strip()
        if trailing not in ('', '```'):
            continue
        if isinstance(result, dict) and (
            isinstance(result.get('errors'), list)
            or isinstance(result.get('gold_errors'), list)
        ):
            found = result
        elif isinstance(result, list) and all(
            isinstance(item, dict) and ('category' in item or 'description' in item)
            for item in result
        ):
            found = {'errors': result}
    return found


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

    result = _final_embedded_prediction(value)
    if result is not None:
        return result, {
            'method': 'final_embedded_json',
            'repaired': False,
            'warnings': ['final prediction JSON was extracted from explanatory text'],
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
    """Validate line-range/anchor pairs and repair only deterministic mismatches."""
    normalized_count = 0
    warnings = []
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
            if declared in ranges:
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


def normalize_prediction_record_code_spans(record):
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
        code_path = resolve_candidate_code_path(record['candidate_code_path'])
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


class I2CHallu(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {'I2C_Hallu_Benchmark_Update_1015_V8_Dual_Image': BENCHMARK_UPDATE_1015_DUAL_IMAGE_DATA_PATH}

    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')

    @staticmethod
    def _first_existing_column(data, names):
        lower_to_col = {str(col).lower(): col for col in data.columns}
        for name in names:
            col = lower_to_col.get(name.lower())
            if col is not None:
                return col
        return None

    @staticmethod
    def _is_missing_value(value):
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        return False

    @classmethod
    def _is_image_path_value(cls, value):
        if isinstance(value, list):
            values = value
        else:
            try:
                values = toliststr(value)
            except Exception:
                values = [value]

        for item in values:
            if cls._is_missing_value(item):
                continue
            text = str(item).strip()
            path = text.split('?', 1)[0].lower()
            if path.endswith(cls.IMAGE_EXTENSIONS) or osp.exists(text):
                return True
        return False

    @classmethod
    def _image_column_is_path(cls, data):
        if 'image' not in data:
            return False
        values = [x for x in data['image'] if not cls._is_missing_value(x)]
        values = values[:20]
        return bool(values) and all(cls._is_image_path_value(x) for x in values)

    @classmethod
    def _normalize_data(cls, data):
        if isinstance(data, list):
            data = pd.DataFrame(data)
        data = data.copy()

        if 'index' not in data:
            index_col = cls._first_existing_column(data, ['sample_id', 'id', 'uid', 'question_id'])
            data['index'] = data[index_col] if index_col is not None else range(len(data))

        if 'question' not in data:
            question_col = cls._first_existing_column(data, ['prompt', 'instruction', 'query', 'text'])
            if question_col is None:
                raise ValueError('I2CHallu data must contain a question, prompt, instruction, query, or text column.')
            data['question'] = data[question_col]

        if 'image' not in data:
            image_col = cls._first_existing_column(data, ['image_base64', 'image_b64', 'base64'])
            if image_col is not None:
                data['image'] = data[image_col]

        if 'image_path' not in data:
            image_path_col = cls._first_existing_column(
                data,
                ['image_path', 'img_path', 'image_file', 'image_file_path', 'gt_image_path', 'gt_image_src'],
            )
            if image_path_col is not None:
                data['image_path'] = data[image_path_col]

        if cls._image_column_is_path(data):
            if 'image_path' not in data:
                data['image_path'] = data['image']
            data = data.drop(columns=['image'])

        if 'image' not in data and 'image_path' not in data:
            raise ValueError('I2CHallu data must contain either an image or image_path column.')

        return data

    def load_data(self, dataset):
        source = self.DATASET_URL.get(dataset, None)
        if isinstance(source, str) and source and osp.exists(source):
            data = self._normalize_data(load(source))
        else:
            data = self._normalize_data(super().load_data(dataset))

        if dataset == 'I2C_Hallu_Benchmark_Update_9_13':
            data = localize_resource_paths(
                data,
                ('image_path', 'candidate_code_path', 'candidate_image_path', 'gt_image_path'),
            )

        if dataset in DUAL_IMAGE_DATASETS:
            for record in data.to_dict(orient='records'):
                image_paths = json.loads(record['image_path'])
                expected_paths = [record['gt_image_path'], record['candidate_image_path']]
                if not isinstance(image_paths, list) or len(image_paths) != 2 or image_paths != expected_paths:
                    raise ValueError(f'Invalid dual-image path order for {record["record_id"]}')
            data = localize_resource_paths(
                data,
                ('candidate_code_path', 'candidate_image_path', 'gt_image_path'),
            )
            data['image_path'] = [
                str([record['gt_image_path'], record['candidate_image_path']])
                for record in data.to_dict(orient='records')
            ]

        if dataset in DUAL_IMAGE_DATASETS:
            data = data.copy()
            version = 'v8'
            data['question'] = [
                build_dual_image_question(record, version)
                for record in data.to_dict(orient='records')
            ]
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            image_paths = toliststr(line['image_path'])
        else:
            image_paths = self.dump_image(line)

        msgs = [dict(type='image', value=image_path) for image_path in image_paths]
        msgs.append(dict(type='text', value=str(line['question'])))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        records = data.to_dict(orient='records') if hasattr(data, 'to_dict') else data

        records = [normalize_prediction_record_code_spans(item) for item in records]

        result_file = get_intermediate_file_path(eval_file, '_gen_results', 'jsonl')
        dump(records, result_file)
        return {
            'total_count': len(records),
            'result_file': result_file,
        }
