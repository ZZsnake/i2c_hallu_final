#!/usr/bin/env python3
"""Evaluation pipeline for I2C hallucination auditing results.

The pipeline follows three ordered stages:

1. sample-level consistency accuracy
2. strict error matching within (category, subtype) buckets
3. code_span line localization and GT-anchor match recall on strict matched errors

The semantic and span judges are intentionally pluggable. Benchmark reporting
should use prompt-rendered LLM-as-judge decisions loaded from cache. Lexical
judges remain available only as deterministic smoke-test fallbacks.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, current_thread, local
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from prompt.code_span_judge_prompt import CODE_SPAN_JUDGE_PROMPT
from prompt.semantic_match_judge_prompt import SEMANTIC_MATCH_JUDGE_PROMPT

DEFAULT_BASE_URL = "https://www.micuapi.ai/v1"
DEFAULT_JUDGE_MODEL = "gpt-5.4"
DEFAULT_API_KEY_ENV = "MICU_API_KEY"
THREAD_LOCAL = local()


TAXONOMY = {
    "entity_error": {
        "missing_entity",
        "extra_duplicate_entity",
        "missing_line_edge_arrow",
        "extra_line_edge_arrow",
        "missing_label_text_entity",
        "extra_label_text_entity",
    },
    "attribute_content_error": {
        "color_mismatch",
        "shape_type_mismatch",
        "size_scale_mismatch",
        "text_symbol_mismatch",
        "numeric_value_error",
        "axis_scale_error",
        "legend_mapping_error",
        "component_attribute_error",
    },
    "layout_relation_error": {
        "position_shift",
        "ordering_error",
        "rotation_orientation_error",
        "occlusion_error",
        "projection_perspective_error",
        "connectivity_error",
        "arrow_direction_error",
        "hierarchy_grouping_error",
        "alignment_error",
    },
    "style_rendering_error": {
        "border_stroke_style_error",
        "line_style_error",
        "marker_style_error",
        "font_style_error",
        "gridline_style_error",
        "decorative_effect_error",
        "rendering_artifact_error",
    },
}

TAXONOMY_ALIASES = {
    ("entity_error", "extra_entity"): ("entity_error", "extra_duplicate_entity"),
    ("entity_error", "duplicate_entity"): ("entity_error", "extra_duplicate_entity"),
}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_code_span(
    value: Any,
    field: str,
    schema_warnings: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """Validate the new localization-array schema without changing anchor text."""
    if value is None:
        return None
    if not isinstance(value, list):
        schema_warnings.append(f"{field} must be a JSON array")
        return None
    if not value:
        schema_warnings.append(f"{field} must not be empty when present")
        return None

    markers: List[Dict[str, Any]] = []
    expected_keys = {"start_line", "end_line", "anchor"}
    for index, marker in enumerate(value):
        marker_field = f"{field}[{index}]"
        if not isinstance(marker, dict):
            schema_warnings.append(f"{marker_field} must be a JSON object")
            return None
        if set(marker) != expected_keys:
            schema_warnings.append(
                f"{marker_field} must contain exactly start_line, end_line, and anchor"
            )
            return None
        start_line = marker.get("start_line")
        end_line = marker.get("end_line")
        anchor = marker.get("anchor")
        if (
            isinstance(start_line, bool)
            or not isinstance(start_line, int)
            or isinstance(end_line, bool)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
        ):
            schema_warnings.append(
                f"{marker_field} line numbers must be positive inclusive integers with start_line <= end_line"
            )
            return None
        if not isinstance(anchor, str) or not anchor.strip():
            schema_warnings.append(f"{marker_field}.anchor must be a non-empty string")
            return None
        markers.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "anchor": anchor,
            }
        )
    return markers


def canonical_code_span(code_span: Optional[Sequence[Dict[str, Any]]]) -> str:
    return json.dumps(code_span or [], ensure_ascii=False, separators=(",", ":"))


def code_span_line_metrics(
    gt_code_span: Optional[Sequence[Dict[str, Any]]],
    pred_code_span: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute line-level PRF over the union of inclusive marker intervals."""

    def merge_intervals(
        code_span: Optional[Sequence[Dict[str, Any]]],
    ) -> List[Tuple[int, int]]:
        intervals = sorted(
            (int(marker["start_line"]), int(marker["end_line"]))
            for marker in (code_span or [])
        )
        merged: List[Tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def line_count(intervals: Sequence[Tuple[int, int]]) -> int:
        return sum(end - start + 1 for start, end in intervals)

    gt_intervals = merge_intervals(gt_code_span)
    pred_intervals = merge_intervals(pred_code_span)
    if not gt_intervals:
        return {
            "evaluable": False,
            "line_precision": None,
            "line_recall": None,
            "line_f1": None,
            "overlap_line_count": 0,
            "gt_line_count": 0,
            "predicted_line_count": line_count(pred_intervals),
        }

    overlap = 0
    gt_index = 0
    pred_index = 0
    while gt_index < len(gt_intervals) and pred_index < len(pred_intervals):
        gt_start, gt_end = gt_intervals[gt_index]
        pred_start, pred_end = pred_intervals[pred_index]
        overlap += max(0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1)
        if gt_end < pred_end:
            gt_index += 1
        else:
            pred_index += 1

    gt_lines = line_count(gt_intervals)
    pred_lines = line_count(pred_intervals)
    precision = safe_div(overlap, pred_lines)
    recall = safe_div(overlap, gt_lines)
    return {
        "evaluable": True,
        "line_precision": precision,
        "line_recall": recall,
        "line_f1": f1_score(precision, recall),
        "overlap_line_count": overlap,
        "gt_line_count": gt_lines,
        "predicted_line_count": pred_lines,
    }


def render_prompt(template: str, values: Dict[str, Any]) -> str:
    """Render brace placeholders without interpreting literal braces in JSON/code."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
    return rendered


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


def sequence_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def token_jaccard(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_like(value: Any, prefer_object: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """Parse JSON strings, including common fenced or prefixed model outputs."""
    if isinstance(value, (dict, list)):
        return value, None
    if value is None:
        return None, "missing"
    if not isinstance(value, str):
        return None, "not a JSON string/object"

    text = strip_markdown_fence(value)
    try:
        return json.loads(text), None
    except Exception as exc:
        first_err = str(exc)

    candidates: List[str] = []
    if prefer_object:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    if not prefer_object:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except Exception:
            pass
    return None, first_err


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj).__name__}")
            yield obj


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_line(file_obj: Any, row: Dict[str, Any]) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_obj.flush()


class StatusLogger:
    def __init__(self, log_file: Path, status_file: Path, quiet: bool = False) -> None:
        self.log_file = log_file
        self.status_file = status_file
        self.quiet = quiet
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: str, pair_id: Optional[str] = None, **fields: Any) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        item = {
            "time": now,
            "event": event,
            "pair_id": pair_id,
            **fields,
        }
        clean_item = {key: value for key, value in item.items() if value is not None}
        message_fields = " ".join(
            f"{key}={value}" for key, value in clean_item.items() if key not in {"time", "event"}
        )
        text_line = f"[{now}] {event}"
        if message_fields:
            text_line += f" {message_fields}"
        with self._lock:
            if not self.quiet:
                print(text_line, flush=True)
            with self.log_file.open("a", encoding="utf-8") as file_obj:
                file_obj.write(text_line + "\n")
                file_obj.flush()
            with self.status_file.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                file_obj.flush()


def parse_extra_params(values: Sequence[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--extra-param must be KEY=JSON_VALUE, got: {value}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--extra-param key is empty in: {value}")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def completion_content(completion: Any) -> str:
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def content_part_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def serialize_usage(completion: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return dict(usage)


def get_thread_client(openai_cls: Any, base_url: str, api_key: str, timeout: float) -> Any:
    state_key = (base_url, api_key, timeout)
    cached_key = getattr(THREAD_LOCAL, "client_key", None)
    cached_client = getattr(THREAD_LOCAL, "client", None)
    if cached_key != state_key or cached_client is None:
        cached_client = openai_cls(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        THREAD_LOCAL.client_key = state_key
        THREAD_LOCAL.client = cached_client
    return cached_client


def stream_completion_content(stream: Any, logger: StatusLogger, pair_id: str, attempt: int) -> str:
    parts: List[str] = []
    chunk_count = 0
    last_log_time = time.monotonic()
    for chunk in stream:
        chunk_count += 1
        choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
        delta = getattr(choice, "delta", None) if choice is not None else None
        text = content_part_text(getattr(delta, "content", None))
        if text:
            parts.append(text)
        now = time.monotonic()
        if chunk_count == 1:
            logger.emit("api_stream_first_chunk", pair_id, attempt=attempt)
        elif now - last_log_time >= 10:
            logger.emit(
                "api_stream_receiving",
                pair_id,
                attempt=attempt,
                chunks=chunk_count,
                response_chars=sum(len(part) for part in parts),
            )
            last_log_time = now
    content = "".join(parts)
    logger.emit(
        "api_stream_finished",
        pair_id,
        attempt=attempt,
        chunks=chunk_count,
        response_chars=len(content),
    )
    return content


def call_with_retries(
    client: Any,
    request_args: Dict[str, Any],
    max_retries: int,
    retry_delay: float,
    logger: StatusLogger,
    pair_id: str,
    stream: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    for attempt in range(max_retries + 1):
        try:
            logger.emit("api_request_start", pair_id, attempt=attempt + 1, stream=stream)
            start_time = time.monotonic()
            if stream:
                response_stream = client.chat.completions.create(**request_args, stream=True)
                raw_response = stream_completion_content(response_stream, logger, pair_id, attempt + 1)
                elapsed = round(time.monotonic() - start_time, 3)
                logger.emit(
                    "api_request_done",
                    pair_id,
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed,
                    response_chars=len(raw_response),
                )
                return raw_response, None
            completion = client.chat.completions.create(**request_args)
            raw_response = completion_content(completion)
            elapsed = round(time.monotonic() - start_time, 3)
            usage = serialize_usage(completion)
            logger.emit(
                "api_request_done",
                pair_id,
                attempt=attempt + 1,
                elapsed_seconds=elapsed,
                response_chars=len(raw_response),
                usage=usage,
            )
            return raw_response, usage
        except Exception as exc:
            logger.emit(
                "api_request_error",
                pair_id,
                attempt=attempt + 1,
                error=f"{type(exc).__name__}: {exc}",
            )
            # Payment is required before another request can succeed. Retrying
            # the same request only wastes time and obscures the real failure.
            if getattr(exc, "status_code", None) == 402:
                raise
            if attempt >= max_retries:
                raise
            sleep_seconds = retry_delay * (2**attempt)
            logger.emit("api_retry_sleep", pair_id, attempt=attempt + 1, sleep_seconds=sleep_seconds)
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable retry state")


@dataclass
class ErrorItem:
    source: str
    sample_id: str
    index: int
    error_id: str
    category: str
    subtype: str
    description: str
    visual_evidence: str
    code_span: Optional[List[Dict[str, Any]]]
    responsible_code_span: Optional[str]
    raw: Dict[str, Any]
    valid_taxonomy: bool
    invalid_reason: Optional[str] = None

    @property
    def label_key(self) -> Tuple[str, str]:
        return (self.category, self.subtype)

    def brief(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "index": self.index,
            "error_id": self.error_id,
            "category": self.category,
            "subtype": self.subtype,
            "description": self.description,
            "visual_evidence": self.visual_evidence,
            "code_span": self.code_span,
            "responsible_code_span": self.responsible_code_span,
            "valid_taxonomy": self.valid_taxonomy,
            "invalid_reason": self.invalid_reason,
        }


@dataclass
class ParsedPrediction:
    raw: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    is_consistent: Optional[bool]
    errors: List[ErrorItem]
    schema_warnings: List[str]


@dataclass
class SemanticJudgeResult:
    description_match: bool
    visual_evidence_match: bool
    same_error: bool
    score: float
    confidence: str
    reason: str
    pair_id: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description_match": self.description_match,
            "visual_evidence_match": self.visual_evidence_match,
            "same_error": self.same_error,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "pair_id": self.pair_id,
            "source": self.source,
        }


@dataclass
class SpanJudgeResult:
    score: Optional[float]
    reason: str
    pair_id: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "reason": self.reason,
            "pair_id": self.pair_id,
            "source": self.source,
        }


class SemanticJudge:
    def compare(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SemanticJudgeResult:
        raise NotImplementedError


class LabelOnlySemanticJudge(SemanticJudge):
    """Development fallback: every same-label pair is treated as semantically equal."""

    def compare(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SemanticJudgeResult:
        return SemanticJudgeResult(
            description_match=True,
            visual_evidence_match=True,
            same_error=True,
            score=1.0,
            confidence="debug",
            reason="label_only judge treats same category/subtype as a match",
            pair_id=pair_id,
            source="label_only",
        )


class LexicalSemanticJudge(SemanticJudge):
    """Deterministic fallback for smoke tests; not a replacement for LLM-as-judge."""

    def __init__(self, threshold: float = 0.42):
        self.threshold = threshold

    def _field_match(self, gt_text: str, pred_text: str) -> Tuple[bool, float]:
        gt_text = normalize_text(gt_text).lower()
        pred_text = normalize_text(pred_text).lower()
        seq = sequence_ratio(gt_text, pred_text)
        jac = token_jaccard(gt_text, pred_text)
        score = max(seq, jac)
        return score >= self.threshold, score

    def compare(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SemanticJudgeResult:
        desc_match, desc_score = self._field_match(gt_error.description, pred_error.description)
        evidence_match, evidence_score = self._field_match(
            gt_error.visual_evidence, pred_error.visual_evidence
        )
        same_error = desc_match and evidence_match
        score = (desc_score + evidence_score) / 2
        return SemanticJudgeResult(
            description_match=desc_match,
            visual_evidence_match=evidence_match,
            same_error=same_error,
            score=score,
            confidence="low",
            reason=(
                "lexical fallback: "
                f"description_score={desc_score:.3f}, evidence_score={evidence_score:.3f}"
            ),
            pair_id=pair_id,
            source="lexical",
        )


class CachedSemanticJudge(SemanticJudge):
    def __init__(self, cache: Dict[str, Dict[str, Any]], fallback: Optional[SemanticJudge] = None):
        self.cache = cache
        self.fallback = fallback
        self.missing = 0

    def compare(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SemanticJudgeResult:
        cached = self.cache.get(pair_id)
        if cached is not None:
            payload = judge_payload(cached)
            desc_match = coerce_bool(payload.get("description_match"), False)
            evidence_match = coerce_bool(payload.get("visual_evidence_match"), False)
            same_error = desc_match and evidence_match
            score = 1.0 if same_error else 0.0
            return SemanticJudgeResult(
                description_match=desc_match,
                visual_evidence_match=evidence_match,
                same_error=same_error,
                score=score,
                confidence=str(payload.get("confidence", "cached")),
                reason=str(payload.get("reason", "cached LLM-as-judge decision")),
                pair_id=pair_id,
                source="cache",
            )
        self.missing += 1
        if self.fallback is not None:
            return self.fallback.compare(gt_error, pred_error, pair_id)
        return SemanticJudgeResult(
            description_match=False,
            visual_evidence_match=False,
            same_error=False,
            score=0.0,
            confidence="missing",
            reason="missing semantic judgement in cache",
            pair_id=pair_id,
            source="cache_missing",
        )


class SpanJudge:
    def score(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SpanJudgeResult:
        raise NotImplementedError


class LexicalSpanJudge(SpanJudge):
    def score(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SpanJudgeResult:
        gt_span = gt_error.code_span
        pred_span = pred_error.code_span
        line_metrics = code_span_line_metrics(gt_span, pred_span)
        if not line_metrics["evaluable"]:
            return SpanJudgeResult(
                score=None,
                reason="GT code_span is unavailable; span quality is not scorable",
                pair_id=pair_id,
                source="lexical",
            )
        if not pred_span:
            return SpanJudgeResult(
                score=0.0,
                reason="prediction code_span is empty while GT code_span is available",
                pair_id=pair_id,
                source="lexical",
            )
        pred_anchors = [normalize_text(marker["anchor"]) for marker in pred_span]
        matched_anchor_count = sum(
            any(
                normalize_text(gt_marker["anchor"]) == pred_anchor
                for pred_anchor in pred_anchors
            )
            for gt_marker in gt_span
        )
        anchor_score = safe_div(matched_anchor_count, len(gt_span))
        score = line_metrics["line_f1"] * anchor_score
        return SpanJudgeResult(
            score=score,
            reason=(
                f"exact lexical anchor matches={matched_anchor_count}/{len(gt_span)}; "
                f"line_f1={line_metrics['line_f1']:.3f}"
            ),
            pair_id=pair_id,
            source="lexical",
        )


class CachedSpanJudge(SpanJudge):
    def __init__(self, cache: Dict[str, Dict[str, Any]], fallback: Optional[SpanJudge] = None):
        self.cache = cache
        self.fallback = fallback
        self.missing = 0

    def score(self, gt_error: ErrorItem, pred_error: ErrorItem, pair_id: str) -> SpanJudgeResult:
        cached = self.cache.get(pair_id)
        if cached is not None:
            payload = judge_payload(cached)
            anchor_result = validate_anchor_match_result(
                payload,
                len(gt_error.code_span or []),
                len(pred_error.code_span or []),
            )
            matched_anchor_count = int(anchor_result["matched_anchor_count"])
            gt_anchor_count = len(gt_error.code_span or [])
            anchor_score = (
                safe_div(matched_anchor_count, gt_anchor_count)
                if gt_anchor_count
                else None
            )
            line_metrics = code_span_line_metrics(gt_error.code_span, pred_error.code_span)
            score = (
                None
                if anchor_score is None or line_metrics["line_f1"] is None
                else line_metrics["line_f1"] * anchor_score
            )
            return SpanJudgeResult(
                score=score,
                reason=(
                    f"matched_anchors={matched_anchor_count}/{gt_anchor_count}; "
                    f"anchor_score={anchor_score}; line_f1={line_metrics['line_f1']}; "
                    f"{anchor_result['reason']}"
                ),
                pair_id=pair_id,
                source="cache",
            )
        self.missing += 1
        if self.fallback is not None:
            return self.fallback.score(gt_error, pred_error, pair_id)
        return SpanJudgeResult(
            score=0.0,
            reason="missing span judgement in cache",
            pair_id=pair_id,
            source="cache_missing",
        )


def normalize_taxonomy_label(category: str, subtype: str) -> Tuple[str, str]:
    return TAXONOMY_ALIASES.get((category, subtype), (category, subtype))


def validate_taxonomy(category: str, subtype: str) -> Tuple[bool, Optional[str]]:
    if category not in TAXONOMY:
        return False, f"invalid category: {category}"
    if subtype not in TAXONOMY[category]:
        return False, f"invalid subtype for {category}: {subtype}"
    return True, None


def make_error_items(
    errors: Any,
    sample_id: str,
    source: str,
    schema_warnings: Optional[List[str]] = None,
) -> List[ErrorItem]:
    if schema_warnings is None:
        schema_warnings = []
    if errors is None:
        return []
    if not isinstance(errors, list):
        schema_warnings.append(f"{source}.gold_errors is not a list")
        return []

    items: List[ErrorItem] = []
    for idx, raw in enumerate(errors):
        if not isinstance(raw, dict):
            schema_warnings.append(f"{source}.gold_errors[{idx}] is not an object")
            continue
        # Some predictions emit a single marker as an object. Normalize only
        # predictions, including the payload used to key set-match requests,
        # so these inputs behave like the equivalent one-element array.
        if source == "pred" and isinstance(raw.get("code_span"), dict):
            raw = {**raw, "code_span": [raw["code_span"]]}
        raw_category = normalize_text(raw.get("category"))
        raw_subtype = normalize_text(raw.get("subtype"))
        category, subtype = normalize_taxonomy_label(raw_category, raw_subtype)
        valid, reason = validate_taxonomy(category, subtype)
        code_span = parse_code_span(
            raw.get("code_span"),
            f"{source}.gold_errors[{idx}].code_span",
            schema_warnings,
        )
        legacy_span = raw.get("responsible_code_span")
        items.append(
            ErrorItem(
                source=source,
                sample_id=sample_id,
                index=idx,
                error_id=normalize_text(raw.get("error_id")) or f"{idx + 1}",
                category=category,
                subtype=subtype,
                description=normalize_text(raw.get("description")),
                visual_evidence=normalize_text(raw.get("visual_evidence")),
                code_span=code_span,
                responsible_code_span=None if legacy_span is None else str(legacy_span),
                raw=raw,
                valid_taxonomy=valid,
                invalid_reason=reason,
            )
        )
    return items


def get_record_sample_id(record: Dict[str, Any]) -> str:
    # Formal benchmark rows retain their source ``sample_id`` for provenance,
    # but use ``record_id`` as the unique evaluation key (positive/negative
    # variants can share the same source sample).  Prefer it whenever present;
    # the legacy MiniBench records simply fall through to sample_id/index.
    for key in ("record_id", "sample_id", "index"):
        value = record.get(key)
        if value is not None:
            return str(value)
    pred, _ = parse_json_like(record.get("prediction"), prefer_object=True)
    if isinstance(pred, dict) and pred.get("sample_id") is not None:
        return str(pred["sample_id"])
    answer, _ = parse_json_like(record.get("answer"), prefer_object=True)
    if isinstance(answer, dict) and answer.get("sample_id") is not None:
        return str(answer["sample_id"])
    return ""


def extract_gt_errors(record: Dict[str, Any], sample_id: str) -> Tuple[List[ErrorItem], List[str]]:
    warnings: List[str] = []
    if isinstance(record.get("reviewed_model_result"), dict):
        errors = record["reviewed_model_result"].get("gold_errors", [])
        return make_error_items(errors, sample_id, "gt", warnings), warnings

    # The formal I2C-Hallu benchmark stores the reviewed reference annotations
    # directly in model_result.errors.  Keep this after reviewed_model_result
    # so legacy, explicitly reviewed annotations retain precedence.
    if isinstance(record.get("model_result"), dict):
        model_result = record["model_result"]
        errors = model_result.get("gold_errors", model_result.get("errors", []))
        return make_error_items(errors, sample_id, "gt", warnings), warnings

    if "answer" in record:
        answer, err = parse_json_like(record.get("answer"), prefer_object=False)
        if err:
            warnings.append(f"answer parse error: {err}")
            return [], warnings
        if isinstance(answer, list):
            return make_error_items(answer, sample_id, "gt", warnings), warnings
        if isinstance(answer, dict):
            return make_error_items(answer.get("gold_errors", []), sample_id, "gt", warnings), warnings
        warnings.append(f"answer has unsupported type: {type(answer).__name__}")
        return [], warnings

    if "gold_errors" in record:
        return make_error_items(record.get("gold_errors", []), sample_id, "gt", warnings), warnings

    warnings.append("no GT errors found in record")
    return [], warnings


def parse_prediction(record: Dict[str, Any], sample_id: str) -> ParsedPrediction:
    warnings: List[str] = []
    pred, err = parse_json_like(record.get("prediction"), prefer_object=True)
    if err:
        return ParsedPrediction(
            raw=None,
            parse_error=err,
            is_consistent=None,
            errors=[],
            schema_warnings=warnings,
        )
    if isinstance(pred, list):
        errors = make_error_items(pred, sample_id, "pred", warnings)
        return ParsedPrediction(
            raw={"is_consistent": len(errors) == 0, "gold_errors": pred},
            parse_error=None,
            is_consistent=len(errors) == 0,
            errors=errors,
            schema_warnings=warnings,
        )
    if not isinstance(pred, dict):
        return ParsedPrediction(
            raw=None,
            parse_error=f"prediction has unsupported type: {type(pred).__name__}",
            is_consistent=None,
            errors=[],
            schema_warnings=warnings,
        )

    is_consistent = pred.get("is_consistent")
    if not isinstance(is_consistent, bool):
        warnings.append("prediction.is_consistent is missing or not boolean")
        is_consistent = None
    errors = make_error_items(pred.get("gold_errors", []), sample_id, "pred", warnings)
    if isinstance(is_consistent, bool) and is_consistent != (len(errors) == 0):
        warnings.append("prediction.is_consistent is not self-consistent with gold_errors length")
    return ParsedPrediction(
        raw=pred,
        parse_error=None,
        is_consistent=is_consistent,
        errors=errors,
        schema_warnings=warnings,
    )


def load_gt_map(path: Optional[Path]) -> Dict[str, Tuple[List[ErrorItem], Dict[str, Any]]]:
    if path is None:
        return {}
    gt_map: Dict[str, Tuple[List[ErrorItem], Dict[str, Any]]] = {}
    for record in iter_jsonl(path):
        sample_id = get_record_sample_id(record)
        errors, _ = extract_gt_errors(record, sample_id)
        gt_map[sample_id] = (errors, record)
    return gt_map


def load_judge_cache(path: Optional[Path], kind: str) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("kind") not in (None, kind):
            continue
        pair_id = row.get("pair_id")
        if not pair_id:
            continue
        cache[str(pair_id)] = row
    return cache


def judge_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the JSON object returned by the judge from common cache wrappers."""
    direct_keys = {
        "description_match",
        "visual_evidence_match",
        "score",
        "confidence",
        "matched_anchor_count",
        "anchor_matches",
        "reason",
    }
    if direct_keys & row.keys():
        return row

    for key in ("model_result", "judgement", "judge_result", "result"):
        value = row.get(key)
        if isinstance(value, dict):
            return value

    for key in ("raw_response", "response", "content", "answer", "output"):
        parsed, _err = parse_json_like(row.get(key), prefer_object=True)
        if isinstance(parsed, dict):
            return parsed
    return {}


def validate_anchor_match_result(
    result: Dict[str, Any],
    gt_anchor_count: int,
    predicted_anchor_count: int,
) -> Dict[str, Any]:
    """Validate anchor identifiers returned by the judge without accepting a score."""
    expected_keys = {"matched_anchor_count", "anchor_matches", "reason"}
    if set(result) != expected_keys:
        raise ValueError(
            "anchor result must contain exactly matched_anchor_count, anchor_matches, and reason"
        )
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("anchor result reason must be a non-empty string")
    matched_anchor_count = result.get("matched_anchor_count")
    if isinstance(matched_anchor_count, bool) or not isinstance(matched_anchor_count, int):
        raise ValueError("matched_anchor_count must be an integer")
    if not 0 <= matched_anchor_count <= gt_anchor_count:
        raise ValueError(
            f"matched_anchor_count must be between 0 and {gt_anchor_count}"
        )
    raw_matches = result.get("anchor_matches")
    if not isinstance(raw_matches, list):
        raise ValueError("anchor_matches must be a JSON array")
    if matched_anchor_count != len(raw_matches):
        raise ValueError("matched_anchor_count must equal the number of anchor_matches")

    used_gt_indices: Set[int] = set()
    anchor_matches: List[Dict[str, Any]] = []
    match_keys = {"gt_anchor_index", "predicted_anchor_indices", "reason"}
    for match_index, raw_match in enumerate(raw_matches):
        if not isinstance(raw_match, dict) or set(raw_match) != match_keys:
            raise ValueError(
                f"anchor_matches[{match_index}] must contain exactly "
                "gt_anchor_index, predicted_anchor_indices, and reason"
            )
        gt_anchor_index = raw_match.get("gt_anchor_index")
        if isinstance(gt_anchor_index, bool) or not isinstance(gt_anchor_index, int):
            raise ValueError(f"anchor_matches[{match_index}].gt_anchor_index must be an integer")
        if not 0 <= gt_anchor_index < gt_anchor_count:
            raise ValueError(
                f"anchor_matches[{match_index}].gt_anchor_index is out of range"
            )
        if gt_anchor_index in used_gt_indices:
            raise ValueError(f"ground-truth anchor {gt_anchor_index} is matched more than once")

        raw_pred_indices = raw_match.get("predicted_anchor_indices")
        if not isinstance(raw_pred_indices, list) or not raw_pred_indices:
            raise ValueError(
                f"anchor_matches[{match_index}].predicted_anchor_indices "
                "must be a non-empty JSON array"
            )
        predicted_anchor_indices: List[int] = []
        for pred_index in raw_pred_indices:
            if isinstance(pred_index, bool) or not isinstance(pred_index, int):
                raise ValueError(
                    f"anchor_matches[{match_index}].predicted_anchor_indices "
                    "must contain integers"
                )
            if not 0 <= pred_index < predicted_anchor_count:
                raise ValueError(
                    f"anchor_matches[{match_index}] predicted anchor {pred_index} is out of range"
                )
            predicted_anchor_indices.append(pred_index)
        if len(predicted_anchor_indices) != len(set(predicted_anchor_indices)):
            raise ValueError(
                f"anchor_matches[{match_index}].predicted_anchor_indices contains duplicates"
            )
        match_reason = raw_match.get("reason")
        if not isinstance(match_reason, str) or not match_reason.strip():
            raise ValueError(f"anchor_matches[{match_index}].reason must be non-empty")

        used_gt_indices.add(gt_anchor_index)
        anchor_matches.append(
            {
                "gt_anchor_index": gt_anchor_index,
                "predicted_anchor_indices": predicted_anchor_indices,
                "reason": match_reason.strip(),
            }
        )

    return {
        "matched_anchor_count": matched_anchor_count,
        "anchor_matches": sorted(anchor_matches, key=lambda item: item["gt_anchor_index"]),
        "reason": reason.strip(),
    }


def validate_judge_result(
    kind: str,
    result: Dict[str, Any],
    request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if kind == "semantic_match":
        return {
            "description_match": coerce_bool(result.get("description_match"), False),
            "visual_evidence_match": coerce_bool(result.get("visual_evidence_match"), False),
            "confidence": str(result.get("confidence", "")),
            "reason": str(result.get("reason", "")),
        }
    if kind == "code_span":
        request = request or {}
        return validate_anchor_match_result(
            result,
            int(request.get("gt_anchor_count", -1)),
            int(request.get("predicted_anchor_count", -1)),
        )
    raise ValueError(f"unsupported judge kind: {kind}")


def is_successful_judge_output(item: Dict[str, Any]) -> bool:
    if item.get("status") != "ok":
        return False
    kind = item.get("kind")
    result = item.get("model_result")
    if kind not in {"semantic_match", "code_span"} or not isinstance(result, dict):
        return False
    try:
        validate_judge_result(str(kind), result, item)
    except Exception:
        return False
    return bool(item.get("pair_id"))


def load_completed_pair_ids(output_file: Path) -> Set[str]:
    completed: Set[str] = set()
    if not output_file.exists():
        return completed
    with output_file.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_successful_judge_output(item):
                completed.add(str(item["pair_id"]))
    return completed


def judge_output_record(
    request: Dict[str, Any],
    status: str,
    model_result: Optional[Dict[str, Any]] = None,
    raw_response: Optional[str] = None,
    parse_error: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "kind": request.get("kind"),
        "pair_id": request.get("pair_id"),
        "sample_id": request.get("sample_id"),
        "gt_error_index": request.get("gt_error_index"),
        "pred_error_index": request.get("pred_error_index"),
        "gt_anchor_count": request.get("gt_anchor_count"),
        "predicted_anchor_count": request.get("predicted_anchor_count"),
        "status": status,
        "model_result": model_result,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "usage": usage,
        "error": error,
    }


def process_judge_request(
    openai_cls: Any,
    base_url: str,
    api_key: str,
    timeout: float,
    request_base: Dict[str, Any],
    request: Dict[str, Any],
    max_retries: int,
    retry_delay: float,
    stream: bool,
    logger: StatusLogger,
) -> Tuple[str, Dict[str, Any]]:
    pair_id = str(request.get("pair_id") or "")
    kind = str(request.get("kind") or "")
    try:
        prompt = str(request.get("judge_prompt") or "")
        if not pair_id:
            raise ValueError("judge request is missing pair_id")
        if kind not in {"semantic_match", "code_span"}:
            raise ValueError(f"unsupported judge request kind: {kind}")
        if not prompt:
            raise ValueError("judge request is missing judge_prompt")

        logger.emit(
            "judge_request_start",
            pair_id,
            kind=kind,
            worker=current_thread().name,
            prompt_chars=len(prompt),
        )
        request_args = {
            **request_base,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = get_thread_client(openai_cls, base_url, api_key, timeout)
        raw_response, usage = call_with_retries(
            client=client,
            request_args=request_args,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            pair_id=pair_id,
            stream=stream,
        )
        parsed, parse_error = parse_json_like(raw_response, prefer_object=True)
        if not isinstance(parsed, dict):
            logger.emit("judge_parse_error", pair_id, kind=kind, parse_error=parse_error)
            return pair_id, judge_output_record(
                request,
                status="parse_error",
                raw_response=raw_response,
                parse_error=parse_error or "response is not a JSON object",
                usage=usage,
            )
        try:
            model_result = validate_judge_result(kind, parsed, request)
        except Exception as exc:
            logger.emit("judge_schema_error", pair_id, kind=kind, error=str(exc))
            return pair_id, judge_output_record(
                request,
                status="schema_error",
                raw_response=raw_response,
                parse_error=str(exc),
                usage=usage,
            )
        logger.emit("judge_request_done", pair_id, kind=kind)
        return pair_id, judge_output_record(
            request,
            status="ok",
            model_result=model_result,
            raw_response=raw_response,
            usage=usage,
        )
    except Exception as exc:
        logger.emit("judge_request_failed", pair_id or None, kind=kind, error=f"{type(exc).__name__}: {exc}")
        return pair_id, judge_output_record(
            request,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def semantic_pair_id(sample_id: str, gt_error: ErrorItem, pred_error: ErrorItem) -> str:
    content = "\n".join(
        [
            sample_id,
            gt_error.category,
            gt_error.subtype,
            str(gt_error.index),
            gt_error.description,
            gt_error.visual_evidence,
            str(pred_error.index),
            pred_error.description,
            pred_error.visual_evidence,
        ]
    )
    return f"semantic:{sample_id}:{gt_error.index}:{pred_error.index}:{stable_hash(content)}"


def span_pair_id(sample_id: str, gt_error: ErrorItem, pred_error: ErrorItem) -> str:
    content = "\n".join(
        [
            "code_span_v5_anchor_match_recall",
            sample_id,
            str(gt_error.index),
            canonical_code_span(gt_error.code_span),
            str(pred_error.index),
            canonical_code_span(pred_error.code_span),
        ]
    )
    return f"span:{sample_id}:{gt_error.index}:{pred_error.index}:{stable_hash(content)}"


def semantic_judge_request(
    sample_id: str,
    gt_error: ErrorItem,
    pred_error: ErrorItem,
    pair_id: str,
) -> Dict[str, Any]:
    prompt_inputs = {
        "category": gt_error.category,
        "subtype": gt_error.subtype,
        "gt_description": gt_error.description,
        "gt_visual_evidence": gt_error.visual_evidence,
        "pred_description": pred_error.description,
        "pred_visual_evidence": pred_error.visual_evidence,
    }
    prompt = render_prompt(SEMANTIC_MATCH_JUDGE_PROMPT, prompt_inputs)
    return {
        "kind": "semantic_match",
        "pair_id": pair_id,
        "sample_id": sample_id,
        "gt_error_index": gt_error.index,
        "pred_error_index": pred_error.index,
        "prompt_inputs": prompt_inputs,
        "judge_prompt": prompt,
        "expected_output_schema": {
            "description_match": "boolean",
            "visual_evidence_match": "boolean",
            "confidence": ["high", "medium", "low"],
            "reason": "string",
        },
    }


def numbered_candidate_code(candidate_code: str) -> str:
    lines = candidate_code.splitlines()
    if not lines:
        lines = [""]
    return json.dumps(
        [
            {"line_number": line_number, "text": line}
            for line_number, line in enumerate(lines, 1)
        ],
        ensure_ascii=False,
        indent=2,
    )


def span_prompt_inputs(
    gt_error: ErrorItem,
    pred_error: ErrorItem,
    candidate_code: str = "",
) -> Dict[str, Any]:
    return {
        "description": gt_error.description,
        "visual_evidence": gt_error.visual_evidence,
        "candidate_code_lines": numbered_candidate_code(candidate_code),
        "gt_code_span": json.dumps(
            gt_error.code_span or [],
            ensure_ascii=False,
            indent=2,
        ),
        "pred_code_span": json.dumps(
            pred_error.code_span or [],
            ensure_ascii=False,
            indent=2,
        ),
    }


def span_judge_request(
    sample_id: str,
    gt_error: ErrorItem,
    pred_error: ErrorItem,
    pair_id: str,
) -> Dict[str, Any]:
    prompt_inputs = span_prompt_inputs(gt_error, pred_error)
    prompt = render_prompt(CODE_SPAN_JUDGE_PROMPT, prompt_inputs)
    return {
        "kind": "code_span",
        "pair_id": pair_id,
        "sample_id": sample_id,
        "gt_error_index": gt_error.index,
        "pred_error_index": pred_error.index,
        "gt_anchor_count": len(gt_error.code_span or []),
        "predicted_anchor_count": len(pred_error.code_span or []),
        "prompt_inputs": prompt_inputs,
        "judge_prompt": prompt,
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


def group_valid_errors(errors: Sequence[ErrorItem]) -> Dict[Tuple[str, str], List[ErrorItem]]:
    grouped: Dict[Tuple[str, str], List[ErrorItem]] = defaultdict(list)
    for error in errors:
        if error.valid_taxonomy:
            grouped[error.label_key].append(error)
    return grouped


def maximum_cardinality_matching(
    left: Sequence[ErrorItem],
    right: Sequence[ErrorItem],
    edge_scores: Dict[Tuple[int, int], float],
) -> List[Tuple[ErrorItem, ErrorItem]]:
    """Return left-right pairs maximizing match count.

    Edges are keyed by local indexes within left/right. Scores only order DFS
    choices as a deterministic tie-breaker.
    """
    if not left or not right:
        return []

    right_order: Dict[int, List[int]] = defaultdict(list)
    for (li, ri), _score in edge_scores.items():
        right_order[li].append(ri)
    for li, ris in right_order.items():
        ris.sort(key=lambda ri: (-edge_scores[(li, ri)], ri))

    match_right_to_left: Dict[int, int] = {}

    def dfs(li: int, seen: Set[int]) -> bool:
        for ri in right_order.get(li, []):
            if ri in seen:
                continue
            seen.add(ri)
            if ri not in match_right_to_left or dfs(match_right_to_left[ri], seen):
                match_right_to_left[ri] = li
                return True
        return False

    left_order = sorted(range(len(left)), key=lambda i: (-len(right_order.get(i, [])), i))
    for li in left_order:
        dfs(li, set())

    pairs = [(left[li], right[ri]) for ri, li in match_right_to_left.items()]
    pairs.sort(key=lambda pair: (pair[0].index, pair[1].index))
    return pairs


def label_only_match_count(
    gt_errors: Sequence[ErrorItem],
    pred_errors: Sequence[ErrorItem],
    key_fn,
) -> Tuple[int, int, int]:
    gt_counter = Counter(key_fn(e) for e in gt_errors if e.valid_taxonomy)
    pred_counter = Counter(key_fn(e) for e in pred_errors if e.valid_taxonomy)
    tp = sum(min(gt_counter[k], pred_counter[k]) for k in gt_counter.keys() | pred_counter.keys())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gt_counter.values()) - tp
    fp += sum(1 for e in pred_errors if not e.valid_taxonomy)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def evaluate_sample(
    record: Dict[str, Any],
    gt_errors: List[ErrorItem],
    pred: ParsedPrediction,
    semantic_judge: SemanticJudge,
    span_judge: SpanJudge,
    semantic_requests: List[Dict[str, Any]],
    span_requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sample_id = get_record_sample_id(record)
    gt_is_consistent = len(gt_errors) == 0
    pred_is_consistent = pred.is_consistent
    consistency_correct = pred_is_consistent == gt_is_consistent

    valid_gt = [e for e in gt_errors if e.valid_taxonomy]
    valid_pred = [e for e in pred.errors if e.valid_taxonomy]
    invalid_pred = [e for e in pred.errors if not e.valid_taxonomy]

    gt_by_label = group_valid_errors(valid_gt)
    pred_by_label = group_valid_errors(valid_pred)

    matched_pairs: List[Tuple[ErrorItem, ErrorItem, SemanticJudgeResult]] = []
    semantic_edges: List[Dict[str, Any]] = []

    for label in sorted(gt_by_label.keys() & pred_by_label.keys()):
        gt_bucket = gt_by_label[label]
        pred_bucket = pred_by_label[label]
        edge_scores: Dict[Tuple[int, int], float] = {}
        judge_by_local_pair: Dict[Tuple[int, int], SemanticJudgeResult] = {}

        for gi, gt_error in enumerate(gt_bucket):
            for pi, pred_error in enumerate(pred_bucket):
                pair_id = semantic_pair_id(sample_id, gt_error, pred_error)
                request = semantic_judge_request(sample_id, gt_error, pred_error, pair_id)
                semantic_requests.append(request)
                judgement = semantic_judge.compare(gt_error, pred_error, pair_id)
                judge_by_local_pair[(gi, pi)] = judgement
                semantic_edges.append(
                    {
                        "pair_id": pair_id,
                        "gt_error_index": gt_error.index,
                        "pred_error_index": pred_error.index,
                        "category": label[0],
                        "subtype": label[1],
                        "judgement": judgement.to_dict(),
                    }
                )
                if judgement.same_error:
                    edge_scores[(gi, pi)] = judgement.score

        bucket_pairs = maximum_cardinality_matching(gt_bucket, pred_bucket, edge_scores)
        for gt_error, pred_error in bucket_pairs:
            gi = gt_bucket.index(gt_error)
            pi = pred_bucket.index(pred_error)
            matched_pairs.append((gt_error, pred_error, judge_by_local_pair[(gi, pi)]))

    matched_gt_indexes = {gt_error.index for gt_error, _pred_error, _judge in matched_pairs}
    matched_pred_indexes = {pred_error.index for _gt_error, pred_error, _judge in matched_pairs}
    unmatched_gt = [e for e in gt_errors if e.index not in matched_gt_indexes]
    unmatched_pred = [e for e in pred.errors if e.index not in matched_pred_indexes]

    span_results: List[Dict[str, Any]] = []
    span_scores: List[float] = []
    for gt_error, pred_error, semantic_result in matched_pairs:
        pair_id = span_pair_id(sample_id, gt_error, pred_error)
        span_requests.append(span_judge_request(sample_id, gt_error, pred_error, pair_id))
        result = span_judge.score(gt_error, pred_error, pair_id)
        if result.score is not None:
            span_scores.append(result.score)
        span_results.append(
            {
                "pair_id": pair_id,
                "semantic_pair_id": semantic_result.pair_id,
                "gt_error_index": gt_error.index,
                "pred_error_index": pred_error.index,
                "category": gt_error.category,
                "subtype": gt_error.subtype,
                "judgement": result.to_dict(),
            }
        )

    cat_tp, cat_fp, cat_fn = label_only_match_count(
        gt_errors, pred.errors, key_fn=lambda e: e.category
    )
    subtype_tp, subtype_fp, subtype_fn = label_only_match_count(
        gt_errors, pred.errors, key_fn=lambda e: (e.category, e.subtype)
    )

    return {
        "sample_id": sample_id,
        "figure_type": record.get("figure_type") or record.get("image_type"),
        "code_type": record.get("code_type"),
        "gt_is_consistent": gt_is_consistent,
        "pred_is_consistent": pred_is_consistent,
        "consistency_correct": bool(consistency_correct),
        "prediction_parse_error": pred.parse_error,
        "prediction_schema_warnings": pred.schema_warnings,
        "gt_error_count": len(gt_errors),
        "pred_error_count": len(pred.errors),
        "valid_gt_error_count": len(valid_gt),
        "valid_pred_error_count": len(valid_pred),
        "invalid_pred_error_count": len(invalid_pred),
        "strict_tp": len(matched_pairs),
        "strict_fp": len(unmatched_pred),
        "strict_fn": len(unmatched_gt),
        "category_only": {"tp": cat_tp, "fp": cat_fp, "fn": cat_fn, **prf(cat_tp, cat_fp, cat_fn)},
        "category_subtype_only": {
            "tp": subtype_tp,
            "fp": subtype_fp,
            "fn": subtype_fn,
            **prf(subtype_tp, subtype_fp, subtype_fn),
        },
        "matched_errors": [
            {
                "gt_error": gt_error.brief(),
                "pred_error": pred_error.brief(),
                "semantic_judgement": semantic_result.to_dict(),
            }
            for gt_error, pred_error, semantic_result in matched_pairs
        ],
        "unmatched_gt_errors": [e.brief() for e in unmatched_gt],
        "unmatched_pred_errors": [e.brief() for e in unmatched_pred],
        "semantic_edges": semantic_edges,
        "span_results": span_results,
        "span_score_sum": sum(span_scores),
        "span_score_count": len(span_scores),
        "span_strong_count": sum(1 for score in span_scores if score >= 0.75),
        "span_perfect_count": sum(1 for score in span_scores if score >= 1.0),
    }


def build_semantic_judge(args: argparse.Namespace) -> SemanticJudge:
    fallback: Optional[SemanticJudge]
    if args.semantic_judge == "label_only":
        fallback = LabelOnlySemanticJudge()
    elif args.semantic_judge == "lexical":
        fallback = LexicalSemanticJudge(threshold=args.lexical_threshold)
    elif args.semantic_judge == "cache":
        fallback = None
    else:
        raise ValueError(f"Unsupported semantic judge: {args.semantic_judge}")

    cache = load_judge_cache(args.semantic_cache, "semantic_match")
    if cache or args.semantic_judge == "cache":
        return CachedSemanticJudge(cache, fallback=fallback)
    if fallback is None:
        raise ValueError("No semantic judge configured")
    return fallback


def build_span_judge(args: argparse.Namespace) -> SpanJudge:
    fallback: Optional[SpanJudge]
    if args.span_judge == "lexical":
        fallback = LexicalSpanJudge()
    elif args.span_judge == "cache":
        fallback = None
    else:
        raise ValueError(f"Unsupported span judge: {args.span_judge}")

    cache = load_judge_cache(args.span_cache, "code_span")
    if cache or args.span_judge == "cache":
        return CachedSpanJudge(cache, fallback=fallback)
    if fallback is None:
        raise ValueError("No span judge configured")
    return fallback


def summarize(sample_rows: Sequence[Dict[str, Any]], semantic_judge: SemanticJudge,
              span_judge: SpanJudge) -> Dict[str, Any]:
    total = len(sample_rows)
    consistency_correct = sum(1 for row in sample_rows if row["consistency_correct"])
    strict_tp = sum(row["strict_tp"] for row in sample_rows)
    strict_fp = sum(row["strict_fp"] for row in sample_rows)
    strict_fn = sum(row["strict_fn"] for row in sample_rows)

    category_tp = sum(row["category_only"]["tp"] for row in sample_rows)
    category_fp = sum(row["category_only"]["fp"] for row in sample_rows)
    category_fn = sum(row["category_only"]["fn"] for row in sample_rows)
    subtype_tp = sum(row["category_subtype_only"]["tp"] for row in sample_rows)
    subtype_fp = sum(row["category_subtype_only"]["fp"] for row in sample_rows)
    subtype_fn = sum(row["category_subtype_only"]["fn"] for row in sample_rows)

    span_score_sum = sum(row["span_score_sum"] for row in sample_rows)
    span_score_count = sum(row["span_score_count"] for row in sample_rows)
    span_strong_count = sum(row["span_strong_count"] for row in sample_rows)
    span_perfect_count = sum(row["span_perfect_count"] for row in sample_rows)

    parse_failed = sum(1 for row in sample_rows if row["prediction_parse_error"])
    schema_warning_samples = sum(1 for row in sample_rows if row["prediction_schema_warnings"])
    invalid_pred_errors = sum(row["invalid_pred_error_count"] for row in sample_rows)
    pred_errors = sum(row["pred_error_count"] for row in sample_rows)
    valid_pred_errors = sum(row["valid_pred_error_count"] for row in sample_rows)
    gt_errors = sum(row["gt_error_count"] for row in sample_rows)
    consistent_gt = sum(1 for row in sample_rows if row["gt_is_consistent"])
    inconsistent_gt = total - consistent_gt

    by_label: Dict[str, Counter] = defaultdict(Counter)
    for row in sample_rows:
        for match in row["matched_errors"]:
            key = f"{match['gt_error']['category']}::{match['gt_error']['subtype']}"
            by_label[key]["tp"] += 1
        for err in row["unmatched_gt_errors"]:
            if err["valid_taxonomy"]:
                key = f"{err['category']}::{err['subtype']}"
            else:
                key = "__invalid_gt_taxonomy__"
            by_label[key]["fn"] += 1
        for err in row["unmatched_pred_errors"]:
            if err["valid_taxonomy"]:
                key = f"{err['category']}::{err['subtype']}"
            else:
                key = "__invalid_pred_taxonomy__"
            by_label[key]["fp"] += 1

    strict = {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn, **prf(strict_tp, strict_fp, strict_fn)}
    category = {
        "tp": category_tp,
        "fp": category_fp,
        "fn": category_fn,
        **prf(category_tp, category_fp, category_fn),
    }
    subtype = {
        "tp": subtype_tp,
        "fp": subtype_fp,
        "fn": subtype_fn,
        **prf(subtype_tp, subtype_fp, subtype_fn),
    }

    return {
        "sample_count": total,
        "gt_distribution": {
            "consistent_samples": consistent_gt,
            "inconsistent_samples": inconsistent_gt,
            "gt_error_count": gt_errors,
        },
        "sample_level_consistency": {
            "correct": consistency_correct,
            "total": total,
            "accuracy": safe_div(consistency_correct, total),
        },
        "strict_error_matching": strict,
        "diagnostic_category_only": category,
        "diagnostic_category_subtype_only": subtype,
        "code_span": {
            "score_sum": span_score_sum,
            "scored_count": span_score_count,
            "average_score": safe_div(span_score_sum, span_score_count),
            "strong_rate_score_ge_0_75": safe_div(span_strong_count, span_score_count),
            "perfect_rate_score_eq_1": safe_div(span_perfect_count, span_score_count),
        },
        "format_validity": {
            "prediction_parse_failed_samples": parse_failed,
            "prediction_parse_rate": safe_div(total - parse_failed, total),
            "schema_warning_samples": schema_warning_samples,
            "pred_error_count": pred_errors,
            "valid_pred_error_count": valid_pred_errors,
            "invalid_pred_error_count": invalid_pred_errors,
            "taxonomy_valid_rate": 1.0 if pred_errors == 0 else safe_div(valid_pred_errors, pred_errors),
        },
        "strict_by_category_subtype": {
            key: {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], **prf(c["tp"], c["fp"], c["fn"])}
            for key, c in sorted(by_label.items())
        },
        "judge_status": {
            "semantic_missing_cache_count": getattr(semantic_judge, "missing", 0),
            "span_missing_cache_count": getattr(span_judge, "missing", 0),
        },
    }


def default_output_dir(pred_file: Path) -> Path:
    return pred_file.with_suffix("").with_name(pred_file.with_suffix("").name + "_i2c_eval")


def load_eval_inputs(args: argparse.Namespace) -> List[Tuple[Dict[str, Any], List[ErrorItem], ParsedPrediction]]:
    gt_map = load_gt_map(args.gt_file)
    items: List[Tuple[Dict[str, Any], List[ErrorItem], ParsedPrediction]] = []
    for record in iter_jsonl(args.pred_file):
        sample_id = get_record_sample_id(record)
        if gt_map:
            if sample_id not in gt_map:
                raise KeyError(f"Sample {sample_id} from predictions is missing in --gt-file")
            gt_errors = gt_map[sample_id][0]
        else:
            gt_errors, _warnings = extract_gt_errors(record, sample_id)
        pred = parse_prediction(record, sample_id)
        items.append((record, gt_errors, pred))
    return items


def evaluate_items(
    items: Sequence[Tuple[Dict[str, Any], List[ErrorItem], ParsedPrediction]],
    semantic_judge: SemanticJudge,
    span_judge: SpanJudge,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sample_rows: List[Dict[str, Any]] = []
    semantic_requests: List[Dict[str, Any]] = []
    span_requests: List[Dict[str, Any]] = []
    for record, gt_errors, pred in items:
        sample_rows.append(
            evaluate_sample(
                record=record,
                gt_errors=gt_errors,
                pred=pred,
                semantic_judge=semantic_judge,
                span_judge=span_judge,
                semantic_requests=semantic_requests,
                span_requests=span_requests,
            )
        )
    return sample_rows, semantic_requests, span_requests


def unique_requests(requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    for request in requests:
        pair_id = str(request.get("pair_id") or "")
        if not pair_id or pair_id in seen:
            continue
        seen.add(pair_id)
        unique.append(request)
    return unique


def batch_output_paths(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path, Path, Path]:
    judge_output_dir = args.judge_output_dir or output_dir / "judge_cache"
    judge_output_dir.mkdir(parents=True, exist_ok=True)

    semantic_cache = args.semantic_cache or judge_output_dir / "semantic_cache.jsonl"
    span_cache = args.span_cache or judge_output_dir / "span_cache.jsonl"
    log_file = args.log_file or judge_output_dir / "batch_api.log"
    status_file = args.status_file or judge_output_dir / "batch_api.status.jsonl"
    return semantic_cache, span_cache, log_file, status_file


def build_request_base(args: argparse.Namespace) -> Dict[str, Any]:
    request_base: Dict[str, Any] = {"model": args.model}
    if args.temperature is not None:
        request_base["temperature"] = args.temperature
    if args.top_p is not None:
        request_base["top_p"] = args.top_p
    if args.max_tokens is not None:
        request_base["max_tokens"] = args.max_tokens
    if args.response_format == "json_object":
        request_base["response_format"] = {"type": "json_object"}
    request_base.update(parse_extra_params(args.extra_param))
    return request_base


def selected_request_slice(
    requests: Sequence[Dict[str, Any]],
    start: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected = list(requests[start:])
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_judge_batch(
    requests: Sequence[Dict[str, Any]],
    output_file: Path,
    args: argparse.Namespace,
    logger: StatusLogger,
    stage: str,
) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    try:
        from openai import OpenAI
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("Please install dependencies first: pip install openai tqdm") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    request_base = build_request_base(args)
    selected = selected_request_slice(unique_requests(requests), args.start, args.limit)
    completed_ids = load_completed_pair_ids(output_file) if args.resume else set()
    pending = [
        request for request in selected
        if str(request.get("pair_id") or "") not in completed_ids
    ]

    logger.emit(
        "batch_stage_start",
        stage=stage,
        output=str(output_file),
        selected=len(selected),
        completed=len(completed_ids),
        pending=len(pending),
        model=args.model,
        base_url=args.base_url,
        workers=args.workers,
        resume=args.resume,
    )

    if args.dry_run:
        print(f"stage: {stage}")
        print(f"selected_requests: {len(selected)}")
        print(f"pending_requests: {len(pending)}")
        if pending:
            first = pending[0]
            print("first_request:")
            print(json.dumps({
                "kind": first.get("kind"),
                "pair_id": first.get("pair_id"),
                "sample_id": first.get("sample_id"),
                "gt_error_index": first.get("gt_error_index"),
                "pred_error_index": first.get("pred_error_index"),
                "prompt_inputs": first.get("prompt_inputs"),
                "judge_prompt": first.get("judge_prompt"),
            }, ensure_ascii=False, indent=2))
        return

    api_key = args.api_key or os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"Missing API key. Pass --api-key, set {args.api_key_env}, or set OPENAI_API_KEY."
        )

    with output_file.open("a", encoding="utf-8") as file_obj:
        if args.workers == 1:
            progress = tqdm(pending, total=len(pending), desc=f"{stage} judge")
            for request in progress:
                pair_id = str(request.get("pair_id") or "")
                progress.set_postfix_str(pair_id[:32])
                _pair_id, item = process_judge_request(
                    openai_cls=OpenAI,
                    base_url=args.base_url,
                    api_key=api_key,
                    timeout=args.timeout,
                    request_base=request_base,
                    request=request,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    stream=args.stream,
                    logger=logger,
                )
                append_jsonl_line(file_obj, item)
                logger.emit("judge_output_written", pair_id or None, stage=stage, status=item.get("status"))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        process_judge_request,
                        OpenAI,
                        args.base_url,
                        api_key,
                        args.timeout,
                        request_base,
                        request,
                        args.max_retries,
                        args.retry_delay,
                        args.stream,
                        logger,
                    )
                    for request in pending
                ]
                progress = tqdm(as_completed(futures), total=len(futures), desc=f"{stage} judge")
                for future in progress:
                    pair_id, item = future.result()
                    progress.set_postfix_str(pair_id[:32])
                    append_jsonl_line(file_obj, item)
                    logger.emit(
                        "judge_output_written",
                        pair_id or None,
                        stage=stage,
                        status=item.get("status"),
                    )

    logger.emit("batch_stage_done", stage=stage, output=str(output_file), written=len(pending))


def run_batch_api_pipeline(args: argparse.Namespace, output_dir: Path) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SemanticJudge, SpanJudge
]:
    items = load_eval_inputs(args)
    semantic_cache_path, span_cache_path, log_file, status_file = batch_output_paths(args, output_dir)
    logger = StatusLogger(log_file=log_file, status_file=status_file, quiet=args.quiet)

    empty_semantic = CachedSemanticJudge({})
    empty_span = CachedSpanJudge({})
    _rows, semantic_requests, _span_requests = evaluate_items(items, empty_semantic, empty_span)
    if args.dump_semantic_requests:
        dump_jsonl(args.dump_semantic_requests, unique_requests(semantic_requests))
    run_judge_batch(semantic_requests, semantic_cache_path, args, logger, stage="semantic")
    if args.dry_run:
        raise SystemExit(0)

    semantic_judge_for_span = CachedSemanticJudge(
        load_judge_cache(semantic_cache_path, "semantic_match")
    )
    span_placeholder = CachedSpanJudge({})
    _rows, _semantic_requests, span_requests = evaluate_items(
        items,
        semantic_judge_for_span,
        span_placeholder,
    )
    if args.dump_span_requests:
        dump_jsonl(args.dump_span_requests, unique_requests(span_requests))
    run_judge_batch(span_requests, span_cache_path, args, logger, stage="span")

    final_semantic_judge = CachedSemanticJudge(load_judge_cache(semantic_cache_path, "semantic_match"))
    final_span_judge = CachedSpanJudge(load_judge_cache(span_cache_path, "code_span"))
    sample_rows, semantic_requests, span_requests = evaluate_items(
        items,
        final_semantic_judge,
        final_span_judge,
    )
    return sample_rows, semantic_requests, span_requests, final_semantic_judge, final_span_judge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate I2C hallucination auditing JSONL predictions."
    )
    parser.add_argument("--pred-file", required=True, type=Path, help="Model result JSONL file.")
    parser.add_argument(
        "--gt-file",
        default=None,
        type=Path,
        help="Optional GT JSONL. If omitted, each result row's answer field is used as GT.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory for summary.json and per_sample.jsonl.",
    )
    parser.add_argument(
        "--run-batch-api",
        action="store_true",
        help=(
            "Run prompt-based semantic and code-span judges with an OpenAI-compatible "
            "batch API before writing final metrics."
        ),
    )
    parser.add_argument(
        "--judge-output-dir",
        default=None,
        type=Path,
        help="Directory for resumable judge cache/log/status files.",
    )
    parser.add_argument(
        "--semantic-judge",
        choices=("lexical", "label_only", "cache"),
        default="cache",
        help=(
            "Semantic judge for same-label error pairs. cache expects prompt-based "
            "LLM-as-judge decisions in --semantic-cache; lexical/label_only are "
            "development fallbacks."
        ),
    )
    parser.add_argument(
        "--semantic-cache",
        default=None,
        type=Path,
        help=(
            "JSONL cache with semantic_match decisions keyed by pair_id. In "
            "--run-batch-api mode this is also the append-only semantic output path."
        ),
    )
    parser.add_argument(
        "--span-judge",
        choices=("lexical", "cache"),
        default="cache",
        help=(
            "Code-span judge. cache expects prompt-based LLM-as-judge decisions in "
            "--span-cache; lexical is a development fallback."
        ),
    )
    parser.add_argument(
        "--span-cache",
        default=None,
        type=Path,
        help=(
            "JSONL cache with code_span decisions keyed by pair_id. In --run-batch-api "
            "mode this is also the append-only span output path."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer --api-key-env.")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment variable to read the API key from if --api-key is omitted.",
    )
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model name.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional generation temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top_p value.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max completion tokens.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per judge request.")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Initial retry delay in seconds.")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent API workers.")
    parser.add_argument("--stream", action="store_true", help="Use streaming chat completions.")
    parser.add_argument(
        "--log-file",
        default=None,
        type=Path,
        help="Batch API text log path. Defaults to judge-output-dir/batch_api.log.",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        type=Path,
        help="Batch API JSONL status path. Defaults to judge-output-dir/batch_api.status.jsonl.",
    )
    parser.add_argument(
        "--response-format",
        choices=("none", "json_object"),
        default="json_object",
        help="Optional response_format passed to the API.",
    )
    parser.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Additional chat.completions.create parameter. Can be repeated.",
    )
    parser.add_argument("--start", type=int, default=0, help="Number of initial judge requests to skip.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum judge requests per stage.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip judge pair_ids already completed in the append-only cache outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build judge prompts and print the first pending request without calling the API.",
    )
    parser.add_argument(
        "--lexical-threshold",
        default=0.42,
        type=float,
        help="Threshold for the lexical semantic fallback.",
    )
    parser.add_argument(
        "--dump-semantic-requests",
        default=None,
        type=Path,
        help="Optional JSONL path for rendered semantic judge prompts.",
    )
    parser.add_argument(
        "--dump-span-requests",
        default=None,
        type=Path,
        help="Optional JSONL path for rendered span judge prompts on strict matched errors.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the compact metric summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.pred_file)
    if args.run_batch_api:
        (
            sample_rows,
            semantic_requests,
            span_requests,
            semantic_judge,
            span_judge,
        ) = run_batch_api_pipeline(args, output_dir)
    else:
        semantic_judge = build_semantic_judge(args)
        span_judge = build_span_judge(args)
        items = load_eval_inputs(args)
        sample_rows, semantic_requests, span_requests = evaluate_items(
            items,
            semantic_judge,
            span_judge,
        )

    summary = summarize(sample_rows, semantic_judge, span_judge)
    dump_json(output_dir / "summary.json", summary)
    dump_jsonl(output_dir / "per_sample.jsonl", sample_rows)

    if args.dump_semantic_requests:
        dump_jsonl(args.dump_semantic_requests, unique_requests(semantic_requests))
    if args.dump_span_requests:
        dump_jsonl(args.dump_span_requests, unique_requests(span_requests))

    if not args.quiet:
        print(f"Wrote {output_dir / 'summary.json'}")
        print(f"Wrote {output_dir / 'per_sample.jsonl'}")
        compact = {
            "sample_accuracy": summary["sample_level_consistency"]["accuracy"],
            "strict_error_f1": summary["strict_error_matching"]["f1"],
            "strict_error_precision": summary["strict_error_matching"]["precision"],
            "strict_error_recall": summary["strict_error_matching"]["recall"],
            "code_span_average_score": summary["code_span"]["average_score"],
            "prediction_parse_rate": summary["format_validity"]["prediction_parse_rate"],
            "taxonomy_valid_rate": summary["format_validity"]["taxonomy_valid_rate"],
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
