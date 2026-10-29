#!/usr/bin/env python3
"""Evaluate I2C hallucination predictions with set-level semantic alignment.

Stages:
1. align original predicted/GT JSON error lists using their available semantic evidence;
2. compute error-discovery precision, recall, and F1 from one-to-one ID pairs;
3. compute line-range F1 and bidirectional anchor F1 for code_span on matched pairs;
4. report both components and their geometric-mean combined score.

The original label-bucket evaluator is intentionally left unchanged. This file
reuses its input parsing and OpenAI-compatible client helpers.
"""

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import current_thread
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import evaluate_i2c_hallu as base
from prompt.set_semantic_match_judge_prompt_v3 import SET_SEMANTIC_MATCH_JUDGE_PROMPT_V3


KIND_SET_MATCH = "set_match"
KIND_CODE_SPAN = "code_span"
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATE_CODE_DIR = BENCHMARK_ROOT / "data" / "candidate_code"
CODE_EXTENSIONS = {
    "python": ".py",
    "svg": ".svg",
    "latex-tikz": ".tex",
    "html-css": ".html",
}


def combined_span_score(
    line_f1: Optional[float],
    anchor_f1: Optional[float],
) -> Optional[float]:
    """Combine line and anchor quality using their geometric mean."""
    if line_f1 is None or anchor_f1 is None:
        return None
    return math.sqrt(float(line_f1) * float(anchor_f1))


def resolve_candidate_code_path(
    candidate_code_dir: Path,
    sample_id: str,
    code_type: str,
) -> Path:
    extension = CODE_EXTENSIONS.get(code_type)
    if extension:
        expected = candidate_code_dir / f"{sample_id}{extension}"
        if expected.is_file():
            return expected
    matches = sorted(
        path for path in candidate_code_dir.glob(f"{sample_id}.*") if path.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Candidate code not found for sample_id={sample_id}")
    raise ValueError(
        f"Multiple candidate code files found for sample_id={sample_id}: {matches}"
    )


def resolve_record_candidate_code_path(
    record: Dict[str, Any],
    gt_record: Dict[str, Any],
    candidate_code_dir: Path,
    sample_id: str,
    code_type: str,
) -> Path:
    """Resolve formal benchmark asset paths, with the legacy layout as fallback.

    Formal benchmark JSONL rows carry a content-addressed
    ``candidate_code_path``.  When a prediction was produced on a different
    machine, that absolute path can be stale; in that case its content-addressed
    filename is relocated into the configured ``candidate_code_dir``.  MiniBench
    rows do not carry paths, so their historical
    ``<candidate_code_dir>/<sample_id>.<extension>`` convention remains fully
    supported.
    """
    for source_name, source_record in (("prediction", record), ("GT", gt_record)):
        raw_path = source_record.get("candidate_code_path")
        if raw_path is None or not str(raw_path).strip():
            continue
        candidate_path = Path(str(raw_path)).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = BENCHMARK_ROOT / candidate_path
        candidate_path = candidate_path.resolve()
        if candidate_path.is_file():
            return candidate_path
        relocated_path = candidate_code_dir / candidate_path.name
        if relocated_path.is_file():
            return relocated_path
        raise FileNotFoundError(
            f"{source_name} candidate_code_path does not exist for "
            f"sample_id={sample_id}: {candidate_path}; also not found at "
            f"configured candidate-code directory: {relocated_path}"
        )
    return resolve_candidate_code_path(candidate_code_dir, sample_id, code_type)


def token_count(usage: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def normalize_token_usage(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not usage:
        return None
    prompt_tokens = token_count(
        usage, ("prompt_tokens", "input_tokens", "promptTokens", "promptTokenCount")
    )
    completion_tokens = token_count(
        usage,
        ("completion_tokens", "output_tokens", "completionTokens", "candidatesTokenCount"),
    )
    total_tokens = token_count(usage, ("total_tokens", "totalTokens", "totalTokenCount"))
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return {key: value for key, value in normalized.items() if value is not None} or None


def aggregate_token_usage(
    usages: Sequence[Optional[Dict[str, Any]]],
) -> Optional[Dict[str, int]]:
    """Sum API-reported usage across response-validation retry attempts."""
    totals: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    has_usage = False
    for usage in usages:
        normalized = normalize_token_usage(usage)
        if not normalized:
            continue
        has_usage = True
        for key in totals:
            totals[key] += int(normalized.get(key, 0))
    return totals if has_usage else None


def empty_token_totals() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "records_with_usage": 0,
        "records_without_usage": 0,
    }


def add_token_usage(totals: Dict[str, int], row: Dict[str, Any]) -> None:
    usage = row.get("token_usage")
    if not isinstance(usage, dict) or not usage:
        totals["records_without_usage"] += 1
        return
    totals["records_with_usage"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None and not isinstance(value, bool):
            totals[key] += int(value)


def merge_token_totals(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key in target:
        target[key] += int(source.get(key, 0))


def original_error_payload(error: base.ErrorItem) -> Dict[str, Any]:
    payload = dict(error.raw)
    if not base.normalize_text(payload.get("error_id")):
        payload["error_id"] = error.error_id
    return payload


def duplicate_ids(errors: Sequence[base.ErrorItem]) -> List[str]:
    counts = Counter(error.error_id for error in errors)
    return sorted(error_id for error_id, count in counts.items() if count > 1)


def set_match_request(
    sample_id: str,
    gt_errors: Sequence[base.ErrorItem],
    pred_errors: Sequence[base.ErrorItem],
) -> Dict[str, Any]:
    gt_payload = [original_error_payload(error) for error in gt_errors]
    pred_payload = [original_error_payload(error) for error in pred_errors]
    prompt_inputs = {
        "ground_truth_errors_json": json.dumps(gt_payload, ensure_ascii=False, indent=2),
        "predicted_errors_json": json.dumps(pred_payload, ensure_ascii=False, indent=2),
    }
    content = json.dumps(
        {"version": "set_match_v3", "sample_id": sample_id, "gt": gt_payload, "pred": pred_payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    request_id = f"set_match_v3:{sample_id}:{base.stable_hash(content)}"
    return {
        "kind": KIND_SET_MATCH,
        "pair_id": request_id,
        "sample_id": sample_id,
        "gt_error_ids": [error.error_id for error in gt_errors],
        "predicted_error_ids": [error.error_id for error in pred_errors],
        "prompt_inputs": prompt_inputs,
        "judge_prompt": base.render_prompt(SET_SEMANTIC_MATCH_JUDGE_PROMPT_V3, prompt_inputs),
    }


def code_span_pair_id(
    sample_id: str,
    gt_error: base.ErrorItem,
    pred_error: base.ErrorItem,
    candidate_code: str,
) -> str:
    fields = [
        "code_span_v5_anchor_match_recall",
        sample_id,
        base.stable_hash(candidate_code, length=40),
        gt_error.error_id,
        base.canonical_code_span(gt_error.code_span),
        pred_error.error_id,
        base.canonical_code_span(pred_error.code_span),
    ]
    return f"{KIND_CODE_SPAN}:{sample_id}:{gt_error.error_id}:{pred_error.error_id}:{base.stable_hash(chr(10).join(fields))}"


def code_span_request(
    sample_id: str,
    gt_error: base.ErrorItem,
    pred_error: base.ErrorItem,
    candidate_code: str,
) -> Dict[str, Any]:
    prompt_inputs = base.span_prompt_inputs(gt_error, pred_error, candidate_code)
    return {
        "kind": KIND_CODE_SPAN,
        "pair_id": code_span_pair_id(sample_id, gt_error, pred_error, candidate_code),
        "sample_id": sample_id,
        "gt_error_id": gt_error.error_id,
        "predicted_error_id": pred_error.error_id,
        "gt_anchor_count": len(gt_error.code_span or []),
        "predicted_anchor_count": len(pred_error.code_span or []),
        "prompt_inputs": prompt_inputs,
        "judge_prompt": base.render_prompt(base.CODE_SPAN_JUDGE_PROMPT, prompt_inputs),
        "expected_output_schema": {
            "matched_anchor_count": "integer",
            "anchor_matches": [
                {
                    "gt_anchor_index": "integer",
                    "predicted_anchor_indices": ["integer"],
                    "reason": "string",
                }
            ],
            "reason": "string",
        },
    }


def string_id_list(value: Any, field: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field} must contain non-empty string IDs")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate IDs")
    return result


def validate_set_match_result(result: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    pred_ids = string_id_list(request.get("predicted_error_ids"), "request.predicted_error_ids")
    gt_ids = string_id_list(request.get("gt_error_ids"), "request.gt_error_ids")
    pred_id_set = set(pred_ids)
    gt_id_set = set(gt_ids)
    raw_matches = result.get("matches")
    if not isinstance(raw_matches, list):
        raise ValueError("matches must be a JSON array")

    used_pred: Set[str] = set()
    used_gt: Set[str] = set()
    matches: List[Dict[str, str]] = []
    for index, raw_match in enumerate(raw_matches):
        if not isinstance(raw_match, dict):
            raise ValueError(f"matches[{index}] must be a JSON object")
        pred_id = raw_match.get("predicted_error_id")
        gt_id = raw_match.get("gt_error_id")
        confidence = raw_match.get("confidence")
        reason = raw_match.get("reason")
        if pred_id not in pred_id_set:
            raise ValueError(f"matches[{index}] has unknown predicted_error_id: {pred_id}")
        if gt_id not in gt_id_set:
            raise ValueError(f"matches[{index}] has unknown gt_error_id: {gt_id}")
        if pred_id in used_pred:
            raise ValueError(f"predicted_error_id is matched more than once: {pred_id}")
        if gt_id in used_gt:
            raise ValueError(f"gt_error_id is matched more than once: {gt_id}")
        if confidence not in ALLOWED_CONFIDENCE:
            raise ValueError(f"matches[{index}].confidence must be high, medium, or low")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"matches[{index}].reason must be a non-empty string")
        used_pred.add(str(pred_id))
        used_gt.add(str(gt_id))
        matches.append(
            {
                "predicted_error_id": str(pred_id),
                "gt_error_id": str(gt_id),
                "confidence": str(confidence),
                "reason": reason.strip(),
            }
        )

    unmatched_pred = string_id_list(
        result.get("unmatched_predicted_error_ids"), "unmatched_predicted_error_ids"
    )
    unmatched_gt = string_id_list(result.get("unmatched_gt_error_ids"), "unmatched_gt_error_ids")
    expected_unmatched_pred = pred_id_set - used_pred
    expected_unmatched_gt = gt_id_set - used_gt
    if set(unmatched_pred) != expected_unmatched_pred:
        raise ValueError("unmatched_predicted_error_ids is not the exact complement of matches")
    if set(unmatched_gt) != expected_unmatched_gt:
        raise ValueError("unmatched_gt_error_ids is not the exact complement of matches")
    return {
        "matches": matches,
        "unmatched_predicted_error_ids": [item for item in pred_ids if item in expected_unmatched_pred],
        "unmatched_gt_error_ids": [item for item in gt_ids if item in expected_unmatched_gt],
    }


def validate_judge_result(result: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    kind = request.get("kind")
    if kind == KIND_SET_MATCH:
        return validate_set_match_result(result, request)
    if kind == KIND_CODE_SPAN:
        return base.validate_anchor_match_result(
            result,
            int(request.get("gt_anchor_count", -1)),
            int(request.get("predicted_anchor_count", -1)),
        )
    raise ValueError(f"unsupported judge kind: {kind}")


def judge_output_record(
    request: Dict[str, Any],
    status: str,
    model_result: Optional[Dict[str, Any]] = None,
    raw_response: Optional[str] = None,
    parse_error: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    token_usage: Optional[Dict[str, int]] = None,
    schema_retry_count: int = 0,
    schema_retry_errors: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "kind": request.get("kind"),
        "pair_id": request.get("pair_id"),
        "sample_id": request.get("sample_id"),
        "gt_error_id": request.get("gt_error_id"),
        "predicted_error_id": request.get("predicted_error_id"),
        "gt_anchor_count": request.get("gt_anchor_count"),
        "predicted_anchor_count": request.get("predicted_anchor_count"),
        "status": status,
        "model_result": model_result,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "usage": usage,
        "token_usage": token_usage if token_usage is not None else normalize_token_usage(usage),
        "schema_retry_count": schema_retry_count,
        "schema_retry_errors": list(schema_retry_errors or []),
        "error": error,
    }


def process_judge_request(
    openai_cls: Any,
    args: argparse.Namespace,
    request_base: Dict[str, Any],
    request: Dict[str, Any],
    logger: base.StatusLogger,
) -> Tuple[str, Dict[str, Any]]:
    pair_id = str(request.get("pair_id") or "")
    kind = str(request.get("kind") or "")
    try:
        prompt = request.get("judge_prompt")
        if not pair_id:
            raise ValueError("judge request is missing pair_id")
        if kind not in {KIND_SET_MATCH, KIND_CODE_SPAN}:
            raise ValueError(f"unsupported judge request kind: {kind}")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("judge request is missing judge_prompt")
        logger.emit(
            "judge_request_start",
            pair_id,
            kind=kind,
            sample_id=request.get("sample_id"),
            worker=current_thread().name,
            prompt_chars=len(prompt),
        )
        request_args = {
            **request_base,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = base.get_thread_client(openai_cls, args.base_url, args.api_key_value, args.timeout)
        schema_retry_errors: List[str] = []
        attempt_usages: List[Optional[Dict[str, Any]]] = []
        for schema_attempt in range(args.schema_retries + 1):
            raw_response, usage = base.call_with_retries(
                client=client,
                request_args=request_args,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                logger=logger,
                pair_id=pair_id,
                stream=False,
            )
            attempt_usages.append(usage)
            parsed, parse_error = base.parse_json_like(raw_response, prefer_object=True)
            if not isinstance(parsed, dict):
                response_error = parse_error or "response is not a JSON object"
                schema_retry_errors.append(f"response parse failed: {response_error}")
                if schema_attempt < args.schema_retries:
                    if getattr(args, "schema_error_feedback", False):
                        request_args["messages"] = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": raw_response},
                            {"role": "user", "content": (
                                "Your previous response could not be parsed as a JSON object: "
                                f"{response_error}. Return a corrected JSON object only."
                            )},
                        ]
                    delay_seconds = args.retry_delay * (2 ** schema_attempt)
                    logger.emit(
                        "judge_parse_retry",
                        pair_id,
                        kind=kind,
                        schema_attempt=schema_attempt + 1,
                        max_schema_retries=args.schema_retries,
                        error=response_error,
                        delay_seconds=delay_seconds,
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                logger.emit(
                    "judge_parse_error",
                    pair_id,
                    kind=kind,
                    schema_attempts=schema_attempt + 1,
                    parse_error=response_error,
                )
                return pair_id, judge_output_record(
                    request,
                    status="parse_error",
                    raw_response=raw_response,
                    parse_error=response_error,
                    usage=usage,
                    token_usage=aggregate_token_usage(attempt_usages),
                    schema_retry_count=schema_attempt,
                    schema_retry_errors=schema_retry_errors,
                )
            try:
                model_result = validate_judge_result(parsed, request)
            except Exception as exc:
                schema_error = str(exc)
                schema_retry_errors.append(schema_error)
                if schema_attempt < args.schema_retries:
                    if getattr(args, "schema_error_feedback", False):
                        if kind == KIND_CODE_SPAN:
                            gt_count = int(request.get("gt_anchor_count", 0))
                            pred_count = int(request.get("predicted_anchor_count", 0))
                            correction_hint = (
                                f"GT anchor indices must be zero-based 0..{gt_count - 1}; "
                                f"predicted anchor indices must be zero-based "
                                f"0..{pred_count - 1}. Each matched GT anchor must "
                                "have coverage as the number 0.5 or 1. "
                                "matched_anchor_count must equal the number of matches. "
                                "Use exactly the three top-level fields "
                                "matched_anchor_count, anchor_matches, reason; "
                                "omit any type or format field."
                            )
                        else:
                            correction_hint = (
                                "Correct the JSON so each predicted error ID and each "
                                "ground-truth error ID appears exactly once, either "
                                "in one match or in its unmatched list."
                            )
                        request_args["messages"] = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": raw_response},
                            {"role": "user", "content": (
                                "Your previous JSON failed validation: "
                                f"{schema_error}. {correction_hint} "
                                "Return the corrected JSON object only."
                            )},
                        ]
                    delay_seconds = args.retry_delay * (2 ** schema_attempt)
                    logger.emit(
                        "judge_schema_retry",
                        pair_id,
                        kind=kind,
                        schema_attempt=schema_attempt + 1,
                        max_schema_retries=args.schema_retries,
                        error=schema_error,
                        delay_seconds=delay_seconds,
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                logger.emit(
                    "judge_schema_error",
                    pair_id,
                    kind=kind,
                    schema_attempts=schema_attempt + 1,
                    error=schema_error,
                )
                return pair_id, judge_output_record(
                    request,
                    status="schema_error",
                    raw_response=raw_response,
                    parse_error=schema_error,
                    usage=usage,
                    token_usage=aggregate_token_usage(attempt_usages),
                    schema_retry_count=schema_attempt,
                    schema_retry_errors=schema_retry_errors,
                )
            logger.emit(
                "judge_request_done",
                pair_id,
                kind=kind,
                schema_retry_count=schema_attempt,
            )
            return pair_id, judge_output_record(
                request,
                status="ok",
                model_result=model_result,
                raw_response=raw_response,
                usage=usage,
                token_usage=aggregate_token_usage(attempt_usages),
                schema_retry_count=schema_attempt,
                schema_retry_errors=schema_retry_errors,
            )
        raise AssertionError("schema retry loop exited without returning")
    except Exception as exc:
        logger.emit(
            "judge_request_failed",
            pair_id or None,
            kind=kind,
            error=f"{type(exc).__name__}: {exc}",
        )
        return pair_id, judge_output_record(
            request,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def successful_cache_row(row: Dict[str, Any], kind: Optional[str] = None) -> bool:
    return (
        row.get("status") == "ok"
        and isinstance(row.get("model_result"), dict)
        and bool(row.get("pair_id"))
        and (kind is None or row.get("kind") == kind)
    )


def load_completed_ids(path: Path, kind: str) -> Set[str]:
    if not path.exists():
        return set()
    completed: Set[str] = set()
    for row in base.iter_jsonl(path):
        if successful_cache_row(row, kind):
            completed.add(str(row["pair_id"]))
    return completed


def load_cache(path: Path, kind: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for row in base.iter_jsonl(path):
        if successful_cache_row(row, kind):
            cache[str(row["pair_id"])] = row
    return cache


def unique_requests(requests: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    result: List[Dict[str, Any]] = []
    for request in requests:
        pair_id = str(request.get("pair_id") or "")
        if pair_id and pair_id not in seen:
            seen.add(pair_id)
            result.append(request)
    return result


def build_request_base(args: argparse.Namespace) -> Dict[str, Any]:
    request: Dict[str, Any] = {"model": args.model}
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.top_p is not None:
        request["top_p"] = args.top_p
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.response_format == "json_object":
        request["response_format"] = {"type": "json_object"}
    extra_params = base.parse_extra_params(args.extra_param)
    if args.disable_thinking:
        # DeepSeek's OpenAI-compatible endpoint requires provider-specific
        # parameters to be placed in the SDK's ``extra_body`` argument.
        extra_body = extra_params.get("extra_body", {})
        if not isinstance(extra_body, dict):
            raise ValueError("--disable-thinking requires extra_body to be a JSON object")
        existing_thinking = extra_body.get("thinking")
        if existing_thinking not in (None, {"type": "disabled"}):
            raise ValueError(
                "--disable-thinking conflicts with --extra-param extra_body.thinking"
            )
        extra_params["extra_body"] = {
            **extra_body,
            "thinking": {"type": "disabled"},
        }
    request.update(extra_params)
    return request


def run_judge_batch(
    requests: Sequence[Dict[str, Any]],
    output_file: Path,
    args: argparse.Namespace,
    logger: base.StatusLogger,
    stage: str,
) -> Dict[str, int]:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    selected = unique_requests(requests)[args.start:]
    if args.limit is not None:
        selected = selected[:args.limit]
    completed = load_completed_ids(output_file, stage) if args.resume else set()
    pending = [request for request in selected if str(request["pair_id"]) not in completed]
    logger.emit(
        "batch_stage_start",
        stage=stage,
        output=str(output_file),
        selected=len(selected),
        skipped=len(selected) - len(pending),
        pending=len(pending),
        model=args.model,
        workers=args.workers,
        resume=args.resume,
    )

    if args.dry_run:
        print(json.dumps({
            "stage": stage,
            "selected_requests": len(selected),
            "pending_requests": len(pending),
            "first_request": pending[0] if pending else None,
        }, ensure_ascii=False, indent=2))
        return empty_token_totals()

    try:
        from openai import OpenAI
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("Install API dependencies first: pip install openai tqdm") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    request_base = build_request_base(args)
    token_totals = empty_token_totals()
    with output_file.open("a", encoding="utf-8") as file_obj:
        if args.workers == 1:
            progress = tqdm(pending, total=len(pending), desc=f"{stage} judge")
            for request in progress:
                pair_id, row = process_judge_request(OpenAI, args, request_base, request, logger)
                progress.set_postfix_str(pair_id[:32])
                base.append_jsonl_line(file_obj, row)
                add_token_usage(token_totals, row)
                logger.emit(
                    "judge_output_written",
                    pair_id,
                    stage=stage,
                    status=row.get("status"),
                    token_usage=row.get("token_usage"),
                    token_totals=token_totals,
                )
        else:
            logger.emit("workers_start", stage=stage, workers=args.workers, submitted=len(pending))
            with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix=f"{stage}-worker") as executor:
                future_to_id = {
                    executor.submit(
                        process_judge_request, OpenAI, args, request_base, request, logger
                    ): str(request["pair_id"])
                    for request in pending
                }
                progress = tqdm(as_completed(future_to_id), total=len(future_to_id), desc=f"{stage} judge")
                for future in progress:
                    pair_id_hint = future_to_id[future]
                    try:
                        pair_id, row = future.result()
                    except Exception as exc:
                        pair_id = pair_id_hint
                        row = {
                            "kind": stage,
                            "pair_id": pair_id,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "token_usage": None,
                        }
                    progress.set_postfix_str(pair_id[:32])
                    base.append_jsonl_line(file_obj, row)
                    add_token_usage(token_totals, row)
                    logger.emit(
                        "judge_output_written",
                        pair_id,
                        stage=stage,
                        status=row.get("status"),
                        token_usage=row.get("token_usage"),
                        token_totals=token_totals,
                    )
    logger.emit(
        "batch_stage_done",
        stage=stage,
        output=str(output_file),
        written=len(pending),
        token_totals=token_totals,
    )
    return token_totals


EvalItem = Tuple[Dict[str, Any], List[base.ErrorItem], base.ParsedPrediction, str]


def load_eval_inputs(args: argparse.Namespace) -> List[EvalItem]:
    gt_map = base.load_gt_map(args.gt_file)
    candidate_code_dir = args.candidate_code_dir.expanduser().resolve()
    if not candidate_code_dir.is_dir():
        raise FileNotFoundError(f"Candidate code directory not found: {candidate_code_dir}")
    items: List[EvalItem] = []
    selection = {
        "restrict_to_gt": bool(args.restrict_to_gt),
        "gt_record_count": len(gt_map),
        "prediction_rows_read": 0,
        "selected_prediction_rows": 0,
        "skipped_prediction_rows_not_in_gt": 0,
    }
    selected_ids: List[str] = []
    for record in base.iter_jsonl(args.pred_file):
        selection["prediction_rows_read"] += 1
        sample_id = base.get_record_sample_id(record)
        if args.restrict_to_gt and sample_id not in gt_map:
            selection["skipped_prediction_rows_not_in_gt"] += 1
            continue
        if not sample_id:
            raise ValueError("Every prediction record must have a stable sample_id or index")
        if gt_map:
            if sample_id not in gt_map:
                raise KeyError(f"Sample {sample_id} from predictions is missing in --gt-file")
            gt_errors, gt_record = gt_map[sample_id]
        else:
            gt_errors, _warnings = base.extract_gt_errors(record, sample_id)
            gt_record = record
        code_type = base.normalize_text(
            gt_record.get("code_type") or record.get("code_type")
        )
        candidate_code_path = resolve_record_candidate_code_path(
            record, gt_record, candidate_code_dir, sample_id, code_type
        )
        candidate_code = candidate_code_path.read_text(
            encoding="utf-8", errors="replace"
        )
        items.append(
            (record, gt_errors, base.parse_prediction(record, sample_id), candidate_code)
        )
        selection["selected_prediction_rows"] += 1
        selected_ids.append(sample_id)
    selection["selected_unique_record_ids"] = len(set(selected_ids))
    selection["duplicate_selected_record_id_count"] = (
        len(selected_ids) - len(set(selected_ids))
    )
    selection["missing_gt_record_count"] = len(set(gt_map) - set(selected_ids))
    # Store this on args instead of expanding EvalItem so output reporting stays
    # independent from the evaluator's internal tuple representation.
    args.input_selection = selection
    return items


def collect_set_requests(items: Sequence[EvalItem]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for record, gt_errors, pred, _candidate_code in items:
        if not gt_errors or not pred.errors:
            continue
        if duplicate_ids(gt_errors) or duplicate_ids(pred.errors):
            continue
        requests.append(set_match_request(base.get_record_sample_id(record), gt_errors, pred.errors))
    return requests


def resolve_set_result(
    sample_id: str,
    gt_errors: Sequence[base.ErrorItem],
    pred_errors: Sequence[base.ErrorItem],
    cache: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    gt_ids = [error.error_id for error in gt_errors]
    pred_ids = [error.error_id for error in pred_errors]
    duplicate_gt = duplicate_ids(gt_errors)
    duplicate_pred = duplicate_ids(pred_errors)
    if duplicate_gt or duplicate_pred:
        return {
            "matches": [],
            "unmatched_predicted_error_ids": pred_ids,
            "unmatched_gt_error_ids": gt_ids,
        }, "invalid_duplicate_error_ids", None
    if not gt_errors or not pred_errors:
        return {
            "matches": [],
            "unmatched_predicted_error_ids": pred_ids,
            "unmatched_gt_error_ids": gt_ids,
        }, "deterministic_empty_side", None

    request = set_match_request(sample_id, gt_errors, pred_errors)
    pair_id = str(request["pair_id"])
    row = cache.get(pair_id)
    if row is None:
        return {
            "matches": [],
            "unmatched_predicted_error_ids": pred_ids,
            "unmatched_gt_error_ids": gt_ids,
        }, "missing_cache", pair_id
    try:
        result = validate_set_match_result(row["model_result"], request)
    except Exception as exc:
        return {
            "matches": [],
            "unmatched_predicted_error_ids": pred_ids,
            "unmatched_gt_error_ids": gt_ids,
            "cache_validation_error": str(exc),
        }, "invalid_cache", pair_id
    return result, "ok", pair_id


def collect_downstream_requests(
    items: Sequence[EvalItem],
    match_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    span_requests: List[Dict[str, Any]] = []
    for record, gt_errors, pred, candidate_code in items:
        sample_id = base.get_record_sample_id(record)
        result, status, _pair_id = resolve_set_result(sample_id, gt_errors, pred.errors, match_cache)
        if status not in {"ok", "deterministic_empty_side"}:
            continue
        gt_by_id = {error.error_id: error for error in gt_errors}
        pred_by_id = {error.error_id: error for error in pred.errors}
        for match in result["matches"]:
            gt_error = gt_by_id[match["gt_error_id"]]
            pred_error = pred_by_id[match["predicted_error_id"]]
            if gt_error.code_span and pred_error.code_span:
                span_requests.append(
                    code_span_request(sample_id, gt_error, pred_error, candidate_code)
                )
    return unique_requests(span_requests)


def cached_anchor_judgement(
    request: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    pair_id = str(request["pair_id"])
    gt_anchor_count = int(request["gt_anchor_count"])
    predicted_anchor_count = int(request["predicted_anchor_count"])
    row = cache.get(pair_id)
    if row is None:
        return {
            "pair_id": pair_id,
            "status": "missing_cache",
            "matched_anchor_count": None,
            "matched_gt_anchor_count": None,
            "supported_predicted_anchor_count": None,
            "supported_predicted_anchor_indices": [],
            "gt_anchor_count": gt_anchor_count,
            "predicted_anchor_count": predicted_anchor_count,
            "anchor_matches": [],
            "anchor_precision": None,
            "anchor_recall": None,
            "anchor_f1": None,
            "anchor_score": None,
            "reason": "",
        }
    try:
        result = base.validate_anchor_match_result(
            row["model_result"],
            gt_anchor_count,
            predicted_anchor_count,
        )
    except Exception as exc:
        return {
            "pair_id": pair_id,
            "status": "invalid_cache",
            "matched_anchor_count": None,
            "matched_gt_anchor_count": None,
            "supported_predicted_anchor_count": None,
            "supported_predicted_anchor_indices": [],
            "gt_anchor_count": gt_anchor_count,
            "predicted_anchor_count": predicted_anchor_count,
            "anchor_matches": [],
            "anchor_precision": None,
            "anchor_recall": None,
            "anchor_f1": None,
            "anchor_score": None,
            "reason": str(exc),
        }
    matched_gt_count = int(result["matched_anchor_count"])
    supported_pred_indices = sorted(
        {
            int(pred_index)
            for anchor_match in result["anchor_matches"]
            for pred_index in anchor_match["predicted_anchor_indices"]
        }
    )
    supported_pred_count = len(supported_pred_indices)
    anchor_precision = base.safe_div(supported_pred_count, predicted_anchor_count)
    anchor_recall = base.safe_div(matched_gt_count, gt_anchor_count)
    anchor_f1 = base.f1_score(anchor_precision, anchor_recall)
    return {
        "pair_id": pair_id,
        "status": "ok",
        **result,
        "matched_gt_anchor_count": matched_gt_count,
        "supported_predicted_anchor_count": supported_pred_count,
        "supported_predicted_anchor_indices": supported_pred_indices,
        "gt_anchor_count": gt_anchor_count,
        "predicted_anchor_count": predicted_anchor_count,
        "anchor_precision": anchor_precision,
        "anchor_recall": anchor_recall,
        "anchor_f1": anchor_f1,
        "anchor_score": anchor_f1,
    }


def evaluate_sample(
    item: EvalItem,
    match_cache: Dict[str, Dict[str, Any]],
    span_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    record, gt_errors, pred, candidate_code = item
    sample_id = base.get_record_sample_id(record)
    match_result, match_status, match_pair_id = resolve_set_result(
        sample_id, gt_errors, pred.errors, match_cache
    )
    gt_by_id = {error.error_id: error for error in gt_errors}
    pred_by_id = {error.error_id: error for error in pred.errors}
    matched: List[Dict[str, Any]] = []
    combined_scores: List[float] = []
    line_f1_scores: List[float] = []
    anchor_precision_scores: List[float] = []
    anchor_recall_scores: List[float] = []
    anchor_f1_scores: List[float] = []

    for match in match_result["matches"]:
        gt_error = gt_by_id[match["gt_error_id"]]
        pred_error = pred_by_id[match["predicted_error_id"]]
        line_metrics = base.code_span_line_metrics(gt_error.code_span, pred_error.code_span)
        if not line_metrics["evaluable"]:
            span = {
                "pair_id": None,
                "status": "unscorable_gt",
                "matched_gt_anchor_count": None,
                "supported_predicted_anchor_count": None,
                "supported_predicted_anchor_indices": [],
                "gt_anchor_count": 0,
                "predicted_anchor_count": len(pred_error.code_span or []),
                "anchor_matches": [],
                "anchor_precision": None,
                "anchor_recall": None,
                "anchor_f1": None,
                "anchor_score": None,
                "reason": "GT code_span is absent; no code location is required or available",
            }
            combined_score = None
            anchor_precision = None
            anchor_recall = None
            anchor_f1 = None
        elif not pred_error.code_span:
            span = {
                "pair_id": None,
                "status": "missing_prediction",
                "matched_gt_anchor_count": 0,
                "supported_predicted_anchor_count": 0,
                "supported_predicted_anchor_indices": [],
                "gt_anchor_count": len(gt_error.code_span or []),
                "predicted_anchor_count": 0,
                "anchor_matches": [],
                "anchor_precision": 0.0,
                "anchor_recall": 0.0,
                "anchor_f1": 0.0,
                "anchor_score": 0.0,
                "reason": "Prediction omitted code_span while GT provides an evaluable location",
            }
            anchor_precision = 0.0
            anchor_recall = 0.0
            anchor_f1 = 0.0
            combined_score = 0.0
        else:
            span = cached_anchor_judgement(
                code_span_request(sample_id, gt_error, pred_error, candidate_code),
                span_cache,
            )
            anchor_precision = span["anchor_precision"]
            anchor_recall = span["anchor_recall"]
            anchor_f1 = span["anchor_f1"]
            combined_score = combined_span_score(
                line_metrics["line_f1"],
                anchor_f1,
            )
        if line_metrics["line_f1"] is not None:
            line_f1_scores.append(float(line_metrics["line_f1"]))
        if anchor_precision is not None:
            anchor_precision_scores.append(float(anchor_precision))
        if anchor_recall is not None:
            anchor_recall_scores.append(float(anchor_recall))
        if anchor_f1 is not None:
            anchor_f1_scores.append(float(anchor_f1))
        if combined_score is not None:
            combined_scores.append(float(combined_score))
        matched.append(
            {
                "predicted_error_id": pred_error.error_id,
                "gt_error_id": gt_error.error_id,
                "confidence": match["confidence"],
                "reason": match["reason"],
                "predicted_error": pred_error.brief(),
                "gt_error": gt_error.brief(),
                "code_span_judgement": {
                    **span,
                    "score": combined_score,
                    "line_precision": line_metrics["line_precision"],
                    "line_recall": line_metrics["line_recall"],
                    "line_f1": line_metrics["line_f1"],
                    "overlap_line_count": line_metrics["overlap_line_count"],
                    "gt_line_count": line_metrics["gt_line_count"],
                    "predicted_line_count": line_metrics["predicted_line_count"],
                    "anchor_precision": anchor_precision,
                    "anchor_recall": anchor_recall,
                    "anchor_f1": anchor_f1,
                    "anchor_score": anchor_f1,
                    "combined_score": combined_score,
                },
            }
        )

    matched_pred_ids = {match["predicted_error_id"] for match in match_result["matches"]}
    matched_gt_ids = {match["gt_error_id"] for match in match_result["matches"]}
    unmatched_pred = [error for error in pred.errors if error.error_id not in matched_pred_ids]
    unmatched_gt = [error for error in gt_errors if error.error_id not in matched_gt_ids]
    tp = len(matched)
    fp = len(pred.errors) - tp
    fn = len(gt_errors) - tp
    gt_is_consistent = len(gt_errors) == 0
    consistency_correct = pred.is_consistent == gt_is_consistent
    span_evaluable_gt = sum(1 for error in gt_errors if bool(error.code_span))

    return {
        "sample_id": sample_id,
        "figure_type": record.get("figure_type") or record.get("image_type"),
        "code_type": record.get("code_type"),
        "gt_is_consistent": gt_is_consistent,
        "pred_is_consistent": pred.is_consistent,
        "consistency_correct": bool(consistency_correct),
        "prediction_parse_error": pred.parse_error,
        "prediction_schema_warnings": pred.schema_warnings,
        "duplicate_gt_error_ids": duplicate_ids(gt_errors),
        "duplicate_predicted_error_ids": duplicate_ids(pred.errors),
        "gt_error_count": len(gt_errors),
        "predicted_error_count": len(pred.errors),
        "invalid_predicted_taxonomy_count": sum(1 for error in pred.errors if not error.valid_taxonomy),
        "match_judge": {
            "pair_id": match_pair_id,
            "status": match_status,
            "raw_alignment": match_result,
        },
        "error_discovery": {"tp": tp, "fp": fp, "fn": fn, **base.prf(tp, fp, fn)},
        "matched_errors": matched,
        "unmatched_predicted_errors": [error.brief() for error in unmatched_pred],
        "unmatched_gt_errors": [error.brief() for error in unmatched_gt],
        "code_span": {
            "matched_score_sum": sum(combined_scores),
            "matched_scored_count": len(combined_scores),
            "matched_average_score": (
                base.safe_div(sum(combined_scores), len(combined_scores))
                if combined_scores
                else None
            ),
            "line_f1_sum": sum(line_f1_scores),
            "line_f1_count": len(line_f1_scores),
            "line_f1_average": (
                base.safe_div(sum(line_f1_scores), len(line_f1_scores))
                if line_f1_scores
                else None
            ),
            "anchor_precision_sum": sum(anchor_precision_scores),
            "anchor_precision_count": len(anchor_precision_scores),
            "anchor_precision_average": (
                base.safe_div(
                    sum(anchor_precision_scores),
                    len(anchor_precision_scores),
                )
                if anchor_precision_scores
                else None
            ),
            "anchor_recall_sum": sum(anchor_recall_scores),
            "anchor_recall_count": len(anchor_recall_scores),
            "anchor_recall_average": (
                base.safe_div(sum(anchor_recall_scores), len(anchor_recall_scores))
                if anchor_recall_scores
                else None
            ),
            "anchor_f1_sum": sum(anchor_f1_scores),
            "anchor_f1_count": len(anchor_f1_scores),
            "anchor_f1_average": (
                base.safe_div(sum(anchor_f1_scores), len(anchor_f1_scores))
                if anchor_f1_scores
                else None
            ),
            "anchor_score_sum": sum(anchor_f1_scores),
            "anchor_score_count": len(anchor_f1_scores),
            "anchor_score_average": (
                base.safe_div(sum(anchor_f1_scores), len(anchor_f1_scores))
                if anchor_f1_scores
                else None
            ),
            "all_gt_count": len(gt_errors),
            "evaluable_gt_count": span_evaluable_gt,
        },
    }


def downstream_summary(
    rows: Sequence[Dict[str, Any]],
    field: str,
    judgement_field: str,
) -> Dict[str, Any]:
    score_sum = sum(float(row[field]["matched_score_sum"]) for row in rows)
    scored_count = sum(int(row[field]["matched_scored_count"]) for row in rows)
    all_gt_count = sum(int(row[field]["all_gt_count"]) for row in rows)
    evaluable_gt_count = sum(int(row[field]["evaluable_gt_count"]) for row in rows)
    line_f1_sum = sum(float(row[field]["line_f1_sum"]) for row in rows)
    line_f1_count = sum(int(row[field]["line_f1_count"]) for row in rows)
    anchor_precision_sum = sum(
        float(row[field]["anchor_precision_sum"]) for row in rows
    )
    anchor_precision_count = sum(
        int(row[field]["anchor_precision_count"]) for row in rows
    )
    anchor_recall_sum = sum(float(row[field]["anchor_recall_sum"]) for row in rows)
    anchor_recall_count = sum(int(row[field]["anchor_recall_count"]) for row in rows)
    anchor_f1_sum = sum(float(row[field]["anchor_f1_sum"]) for row in rows)
    anchor_f1_count = sum(int(row[field]["anchor_f1_count"]) for row in rows)
    missing_cache = 0
    invalid_cache = 0
    for row in rows:
        for match in row["matched_errors"]:
            status = match[judgement_field]["status"]
            missing_cache += int(status == "missing_cache")
            invalid_cache += int(status == "invalid_cache")
    return {
        "matched_only": {
            "score_sum": score_sum,
            "scored_count": scored_count,
            "average_score": base.safe_div(score_sum, scored_count) if scored_count else None,
            "score_definition": "sqrt(line_f1 * anchor_f1)",
        },
        "line_localization": {
            "f1_sum": line_f1_sum,
            "scored_count": line_f1_count,
            "average_f1": base.safe_div(line_f1_sum, line_f1_count) if line_f1_count else None,
        },
        "anchor_judge": {
            "precision_sum": anchor_precision_sum,
            "precision_scored_count": anchor_precision_count,
            "average_precision": (
                base.safe_div(anchor_precision_sum, anchor_precision_count)
                if anchor_precision_count
                else None
            ),
            "recall_sum": anchor_recall_sum,
            "recall_scored_count": anchor_recall_count,
            "average_recall": (
                base.safe_div(anchor_recall_sum, anchor_recall_count)
                if anchor_recall_count
                else None
            ),
            "f1_sum": anchor_f1_sum,
            "f1_scored_count": anchor_f1_count,
            "average_f1": (
                base.safe_div(anchor_f1_sum, anchor_f1_count)
                if anchor_f1_count
                else None
            ),
            "score_sum": anchor_f1_sum,
            "scored_count": anchor_f1_count,
            "average_score": (
                base.safe_div(anchor_f1_sum, anchor_f1_count)
                if anchor_f1_count
                else None
            ),
            "precision_definition": (
                "supported_predicted_anchor_count / predicted_anchor_count"
            ),
            "recall_definition": (
                "matched_gt_anchor_count / gt_anchor_count"
            ),
            "f1_definition": (
                "2 * anchor_precision * anchor_recall / "
                "(anchor_precision + anchor_recall)"
            ),
            "score_definition": "anchor_f1",
        },
        "end_to_end_evaluable_gt": {
            "score_sum": score_sum,
            "evaluable_gt_error_count": evaluable_gt_count,
            "average_score": (
                base.safe_div(score_sum, evaluable_gt_count)
                if evaluable_gt_count
                else None
            ),
            "undetected_evaluable_gt_contribute_zero": True,
            "gt_without_code_span_excluded": True,
        },
        "gt_code_span_coverage": {
            "evaluable_gt_error_count": evaluable_gt_count,
            "all_gt_error_count": all_gt_count,
            "coverage": base.safe_div(evaluable_gt_count, all_gt_count),
        },
        "missing_cache_count": missing_cache,
        "invalid_cache_count": invalid_cache,
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    tp = sum(row["error_discovery"]["tp"] for row in rows)
    fp = sum(row["error_discovery"]["fp"] for row in rows)
    fn = sum(row["error_discovery"]["fn"] for row in rows)
    gt_count = sum(row["gt_error_count"] for row in rows)
    pred_count = sum(row["predicted_error_count"] for row in rows)
    match_status = Counter(row["match_judge"]["status"] for row in rows)
    by_gt_category: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for match in row["matched_errors"]:
            by_gt_category[match["gt_error"]["category"]]["tp"] += 1
        for error in row["unmatched_gt_errors"]:
            by_gt_category[error["category"]]["fn"] += 1

    return {
        "evaluation_mode": {
            "matching_granularity": "set_level_per_sample",
            "description_and_visual_evidence_are_primary": True,
            "input_schema_is_format_agnostic": True,
            "matching_semantic_fields": ["description", "visual_evidence_when_present"],
            "missing_optional_fields_are_penalized": False,
            "category_is_hard_filter": False,
            "subtype_is_hard_filter": False,
            "one_to_one_alignment": True,
            "downstream_scoring": ["code_span_line_f1", "code_span_anchor_judge"],
            "code_span_combination": (
                "sqrt(line_f1 * anchor_f1)"
            ),
            "code_span_line_f1_definition": (
                "inclusive line-set overlap from every marker's start_line/end_line"
            ),
            "code_span_anchor_judge_context": [
                "candidate_code_lines",
                "ground_truth_start_line_end_line_anchor",
                "predicted_start_line_end_line_anchor",
            ],
            "code_span_anchor_occurrence_checked_upstream": True,
            "code_span_anchor_judge_checks_declared_range_occurrence": False,
            "code_span_anchor_judge_model_outputs_score": False,
            "code_span_anchor_judge_match_identifiers": [
                "zero_based_gt_anchor_index",
                "zero_based_predicted_anchor_index",
            ],
        },
        "sample_count": total,
        "sample_level_consistency": {
            "correct": sum(1 for row in rows if row["consistency_correct"]),
            "total": total,
            "accuracy": base.safe_div(sum(1 for row in rows if row["consistency_correct"]), total),
        },
        "error_discovery": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_error_count": pred_count,
            "gt_error_count": gt_count,
            **base.prf(tp, fp, fn),
        },
        "code_span": downstream_summary(
            rows, "code_span", "code_span_judgement"
        ),
        "error_discovery_by_gt_category": {
            key: {
                "tp": counts["tp"],
                "fn": counts["fn"],
                "recall": base.safe_div(counts["tp"], counts["tp"] + counts["fn"]),
            }
            for key, counts in sorted(by_gt_category.items())
        },
        "format_validity": {
            "prediction_parse_failed_samples": sum(1 for row in rows if row["prediction_parse_error"]),
            "prediction_schema_warning_samples": sum(
                1 for row in rows if row["prediction_schema_warnings"]
            ),
            "invalid_predicted_taxonomy_count": sum(
                row["invalid_predicted_taxonomy_count"] for row in rows
            ),
            "duplicate_error_id_samples": sum(
                1
                for row in rows
                if row["duplicate_gt_error_ids"] or row["duplicate_predicted_error_ids"]
            ),
        },
        "judge_status": {
            "set_match_status_counts": dict(sorted(match_status.items())),
            "set_match_missing_cache_count": match_status["missing_cache"],
            "set_match_invalid_cache_count": match_status["invalid_cache"],
        },
    }


def default_output_dir(pred_file: Path) -> Path:
    stem = pred_file.with_suffix("")
    return stem.with_name(stem.name + "_i2c_eval_set_match_v3")


def resolve_paths(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    judge_dir = args.judge_output_dir or output_dir / "judge_cache"
    output_file = args.output_file
    if output_file is not None and not output_file.is_absolute():
        output_file = output_dir / output_file
    match_cache = args.match_cache or output_file or judge_dir / "set_match_cache.jsonl"
    span_cache = args.span_cache or judge_dir / "code_span_cache.jsonl"
    log_file = args.log_file or judge_dir / "batch_api.log"
    status_file = args.status_file or judge_dir / "batch_api.status.jsonl"
    return match_cache, span_cache, log_file, status_file


def run_batch_pipeline(
    args: argparse.Namespace,
    items: Sequence[EvalItem],
    paths: Tuple[Path, Path, Path, Path],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    match_path, span_path, log_path, status_path = paths
    logger = base.StatusLogger(log_path, status_path, quiet=args.quiet)
    if not args.dry_run:
        api_key = args.api_key or os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"Missing API key. Pass --api-key or set {args.api_key_env}."
            )
        args.api_key_value = api_key
    else:
        args.api_key_value = "dry-run"

    logger.emit(
        "run_start",
        model=args.model,
        base_url=args.base_url,
        prediction_file=str(args.pred_file),
        gt_file=str(args.gt_file) if args.gt_file else None,
        sample_count=len(items),
        workers=args.workers,
        resume=args.resume,
    )
    run_token_totals = empty_token_totals()
    set_requests = collect_set_requests(items)
    if args.dump_match_requests:
        base.dump_jsonl(args.dump_match_requests, set_requests)
    set_token_totals = run_judge_batch(set_requests, match_path, args, logger, KIND_SET_MATCH)
    merge_token_totals(run_token_totals, set_token_totals)
    if args.dry_run:
        raise SystemExit(0)

    match_cache = load_cache(match_path, KIND_SET_MATCH)
    span_requests = collect_downstream_requests(items, match_cache)
    if args.dump_span_requests:
        base.dump_jsonl(args.dump_span_requests, span_requests)
    span_token_totals = run_judge_batch(span_requests, span_path, args, logger, KIND_CODE_SPAN)
    merge_token_totals(run_token_totals, span_token_totals)
    logger.emit("run_done", token_totals=run_token_totals)
    return (
        load_cache(match_path, KIND_SET_MATCH),
        load_cache(span_path, KIND_CODE_SPAN),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate I2C hallucination predictions with one set-level semantic "
            "alignment per sample using original JSON error lists, followed "
            "by code-span scoring."
        )
    )
    parser.add_argument("--pred-file", required=True, type=Path, help="Prediction JSONL file.")
    parser.add_argument(
        "--gt-file",
        type=Path,
        default=None,
        help="Optional GT JSONL; otherwise GT is read from each prediction record.",
    )
    parser.add_argument(
        "--candidate-code-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_CODE_DIR,
        help="Directory containing candidate code files keyed by sample_id.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-batch-api", action="store_true")
    parser.add_argument("--judge-output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Alias for --match-cache, provided for batch-runner compatibility.",
    )
    parser.add_argument("--match-cache", type=Path, default=None)
    parser.add_argument("--span-cache", type=Path, default=None)
    parser.add_argument("--base-url", default=base.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="Prefer --api-key-env for security.")
    parser.add_argument("--api-key-env", default=base.DEFAULT_API_KEY_ENV)
    parser.add_argument("--model", default=base.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument(
        "--schema-retries",
        type=int,
        default=3,
        help=(
            "Maximum additional API calls after an otherwise successful response "
            "fails JSON parsing or result-schema validation."
        ),
    )
    parser.add_argument(
        "--schema-error-feedback",
        action="store_true",
        help="Include the previous invalid response and validation error in schema retries.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--response-format", choices=("none", "json_object"), default="json_object"
    )
    parser.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restrict-to-gt",
        action="store_true",
        help=(
            "Skip prediction rows whose record_id/sample_id is absent from --gt-file. "
            "Use this to score an exact benchmark subset from a larger prediction JSONL."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Send DeepSeek-compatible extra_body={thinking:{type:disabled}} with "
            "every judge request."
        ),
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--dump-match-requests", type=Path, default=None)
    parser.add_argument("--dump-span-requests", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.output_file is not None and args.match_cache is not None:
        parser.error("--output-file and --match-cache are aliases; use only one")
    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.schema_retries < 0:
        parser.error("--schema-retries must be >= 0")
    if args.restrict_to_gt and args.gt_file is None:
        parser.error("--restrict-to-gt requires --gt-file")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.pred_file)
    paths = resolve_paths(args, output_dir)
    items = load_eval_inputs(args)
    if args.run_batch_api:
        match_cache, span_cache = run_batch_pipeline(args, items, paths)
    else:
        match_cache = load_cache(paths[0], KIND_SET_MATCH)
        span_cache = load_cache(paths[1], KIND_CODE_SPAN)

    rows = [
        evaluate_sample(item, match_cache, span_cache)
        for item in items
    ]
    summary = summarize(rows)
    summary["input_selection"] = args.input_selection
    base.dump_json(output_dir / "summary.json", summary)
    base.dump_jsonl(output_dir / "per_sample.jsonl", rows)
    if not args.quiet:
        print(f"Wrote {output_dir / 'summary.json'}")
        print(f"Wrote {output_dir / 'per_sample.jsonl'}")
        compact = {
            "sample_accuracy": summary["sample_level_consistency"]["accuracy"],
            "error_discovery_precision": summary["error_discovery"]["precision"],
            "error_discovery_recall": summary["error_discovery"]["recall"],
            "error_discovery_f1": summary["error_discovery"]["f1"],
            "code_span_line_f1": summary["code_span"]["line_localization"]["average_f1"],
            "code_span_anchor_precision": summary["code_span"]["anchor_judge"]["average_precision"],
            "code_span_anchor_recall": summary["code_span"]["anchor_judge"]["average_recall"],
            "code_span_anchor_f1": summary["code_span"]["anchor_judge"]["average_f1"],
            "code_span_matched_only": summary["code_span"]["matched_only"]["average_score"],
            "code_span_end_to_end": summary["code_span"]["end_to_end_evaluable_gt"]["average_score"],
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
