"""Helpers adapted from /mnt/disk1/data_group/zhoujiawei/skills/scripts/batch_api_runner_template.py.

Kept locally so the inference program does not depend on an external template path.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock, local
from typing import Any
THREAD_LOCAL = local()

class StatusLogger:
    def __init__(self, log_file: Path, status_file: Path) -> None:
        self.log_file = log_file
        self.status_file = status_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: str, sample_id_value: str | None = None, **fields: Any) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        item = {
            "time": now,
            "event": event,
            "sample_id": sample_id_value,
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
            if getattr(self, '_progress_active', False):
                # Keep full event logs on disk; only show actionable events above the bar.
                if event in {'api_request_error', 'api_retry_sleep', 'sample_error',
                             'sample_future_error', 'run_stop_requested'}:
                    from tqdm import tqdm
                    tqdm.write(text_line, file=sys.stderr)
            else:
                print(text_line, file=sys.stderr, flush=True)
            with self.log_file.open("a", encoding="utf-8") as file_obj:
                file_obj.write(text_line + "\n")
                file_obj.flush()
            with self.status_file.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                file_obj.flush()


def parse_extra_params(values: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
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


def content_part_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def serialize_usage_value(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    try:
        return dict(usage)
    except (TypeError, ValueError):
        keys = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        )
        return {
            key: getattr(usage, key)
            for key in keys
            if getattr(usage, key, None) is not None
        } or None


def token_count_from_usage(usage: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
    return None


def normalize_token_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not usage:
        return None

    prompt_tokens = token_count_from_usage(
        usage,
        ("prompt_tokens", "input_tokens", "promptTokens", "promptTokenCount"),
    )
    completion_tokens = token_count_from_usage(
        usage,
        (
            "completion_tokens",
            "output_tokens",
            "completionTokens",
            "candidatesTokenCount",
        ),
    )
    total_tokens = token_count_from_usage(
        usage,
        ("total_tokens", "totalTokens", "totalTokenCount"),
    )
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return {key: value for key, value in token_usage.items() if value is not None} or None


def empty_token_totals() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "records_with_usage": 0,
        "records_without_usage": 0,
    }


def add_item_token_usage(totals: dict[str, int], item: dict[str, Any]) -> None:
    token_usage = item.get("token_usage")
    if not isinstance(token_usage, dict) or not token_usage:
        totals["records_without_usage"] += 1
        return

    totals["records_with_usage"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = token_usage.get(key)
        if isinstance(value, bool) or value is None:
            continue
        totals[key] += int(value)


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


