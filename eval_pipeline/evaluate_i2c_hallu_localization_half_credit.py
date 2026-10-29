#!/usr/bin/env python3
"""Versioned 0/0.5/1 anchor scorer using the existing set-match evaluator.

The semantic error matching and deterministic line F1 are unchanged.  This
module replaces only the code-span judge request, validation, and aggregation.
Use a separate output directory: old binary anchor cache rows are incompatible.
"""

from typing import Any, Dict

import evaluate_i2c_hallu as base
import evaluate_i2c_hallu_set_match as evaluator


RUBRIC_VERSION = "code_span_v6_half_credit_20261026"
LEGACY_CODE_SPAN_REQUEST = evaluator.code_span_request
LEGACY_CACHED_ANCHOR_JUDGEMENT = evaluator.cached_anchor_judgement

PROMPT = """You judge localization anchors for an already matched Image-to-Code error.

Judge what causal code each prediction identifies in the *candidate code*, not
what corrected code should be. The GT anchor is a full target worth 1 point.
For every GT anchor, compare all predicted anchors together and assign:
  1: the predicted anchor(s) identify the complete causal property, value,
     assignment, command, operation, or statement represented by this GT anchor;
  0.5: they identify the same causal code and at least one relevant causal
     property or operation, but omit another necessary part of this GT anchor;
  0: they give only an element/name/container, generic nearby code, an unrelated
     property, or a contradictory causal property/value.

Examples: GT `fill=yellow!30, draw=black`, prediction `draw=black` => 0.5;
prediction only a node name => 0. GT `visualMap` with causal orientation or
placement settings, prediction only `visualMap: {{` => 0. A verbatim copy of
the GT causal anchor in the candidate code is complete (1), even when that
candidate value differs from the desired image; the task is to locate the bug.

Use all predicted anchors jointly if multiple fragments are needed for a full
GT anchor. List only the predicted anchor indices that genuinely contribute to
each nonzero match. One predicted anchor may support multiple GT anchors when
it truly contains their causal content. Do not use line overlap or anchor
occurrence to decide coverage: these are checked or scored separately.

Matched error description: {description}
Visual evidence: {visual_evidence}

Candidate code with line numbers:
{candidate_code_lines}

GT code_span markers (zero-based array indices):
{gt_code_span}

Predicted code_span markers (zero-based array indices):
{pred_code_span}

Return JSON only, exactly these top-level fields:
{{
  "matched_anchor_count": 1,
  "anchor_matches": [
    {{"gt_anchor_index": 0, "predicted_anchor_indices": [0],
      "coverage": 0.5, "reason": "One causal property is located; another is absent."}}
  ],
  "reason": "Concise overall explanation."
}}
Include one entry for each GT anchor with coverage 0.5 or 1; omit zero matches.
matched_anchor_count must equal the number of entries. Each GT index is unique.
coverage must be numeric 0.5 or 1, never a string. Do not output an overall score.
"""


def code_span_request(sample_id, gt_error, pred_error, candidate_code):
    request = LEGACY_CODE_SPAN_REQUEST(sample_id, gt_error, pred_error, candidate_code)
    request["pair_id"] = request["pair_id"].replace(
        "code_span:", "code_span_half_credit:", 1
    ) + ":" + base.stable_hash(RUBRIC_VERSION + request["pair_id"])
    request["judge_prompt"] = base.render_prompt(PROMPT, request["prompt_inputs"])
    request["expected_output_schema"]["anchor_matches"][0]["coverage"] = "0.5 or 1"
    return request


def validate_anchor_result(result: Dict[str, Any], gt_count: int, pred_count: int):
    # Some JSON-mode endpoints prepend their response-format marker even when
    # the requested fields are otherwise valid. It is metadata, not a score.
    if (isinstance(result, dict) and result.get("type") == "json_object" and
            set(result) == {"type", "matched_anchor_count", "anchor_matches", "reason"}):
        result = {key: value for key, value in result.items() if key != "type"}
    if not isinstance(result, dict) or set(result) != {
        "matched_anchor_count", "anchor_matches", "reason"
    }:
        raise ValueError("anchor result must have matched_anchor_count, anchor_matches, reason")
    reason = result["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty")
    count = result["matched_anchor_count"]
    matches = result["anchor_matches"]
    if isinstance(count, bool) or not isinstance(count, int) or not isinstance(matches, list):
        raise ValueError("matched_anchor_count must be integer and anchor_matches an array")
    if count != len(matches) or count > gt_count:
        raise ValueError("matched_anchor_count must equal valid anchor_matches length")
    seen = set()
    validated = []
    for index, match in enumerate(matches):
        if not isinstance(match, dict) or set(match) != {
            "gt_anchor_index", "predicted_anchor_indices", "coverage", "reason"
        }:
            raise ValueError(f"anchor_matches[{index}] has invalid fields")
        gt_index = match["gt_anchor_index"]
        pred_indices = match["predicted_anchor_indices"]
        coverage = match["coverage"]
        if isinstance(gt_index, bool) or not isinstance(gt_index, int) or not 0 <= gt_index < gt_count or gt_index in seen:
            raise ValueError(f"anchor_matches[{index}] has invalid or repeated GT index")
        if not isinstance(pred_indices, list) or not pred_indices or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < pred_count
            for item in pred_indices
        ) or len(pred_indices) != len(set(pred_indices)):
            raise ValueError(f"anchor_matches[{index}] has invalid predicted indices")
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or coverage not in (0.5, 1):
            raise ValueError(f"anchor_matches[{index}].coverage must be 0.5 or 1")
        if not isinstance(match["reason"], str) or not match["reason"].strip():
            raise ValueError(f"anchor_matches[{index}].reason must be non-empty")
        seen.add(gt_index)
        validated.append({**match, "coverage": float(coverage), "reason": match["reason"].strip()})
    return {"matched_anchor_count": count,
            "anchor_matches": sorted(validated, key=lambda item: item["gt_anchor_index"]),
            "reason": reason.strip()}


def validate_judge_result(result, request):
    if request.get("kind") == evaluator.KIND_CODE_SPAN:
        return validate_anchor_result(
            result, int(request["gt_anchor_count"]), int(request["predicted_anchor_count"])
        )
    return evaluator.validate_set_match_result(result, request)


def cached_anchor_judgement(request, cache):
    pair_id = request["pair_id"]
    gt_count = request["gt_anchor_count"]
    pred_count = request["predicted_anchor_count"]
    row = cache.get(pair_id)
    if row is None:
        status, result, reason = "missing_cache", None, ""
    else:
        try:
            result = validate_anchor_result(row["model_result"], gt_count, pred_count)
            status, reason = "ok", result["reason"]
        except Exception as exc:
            status, result, reason = "invalid_cache", None, str(exc)
    if result is None:
        return {"pair_id": pair_id, "status": status, "matched_anchor_count": None,
                "matched_gt_anchor_count": None, "supported_predicted_anchor_count": None,
                "supported_predicted_anchor_indices": [], "gt_anchor_count": gt_count,
                "predicted_anchor_count": pred_count, "anchor_matches": [],
                "anchor_precision": None, "anchor_recall": None, "anchor_f1": None,
                "anchor_score": None, "reason": reason}

    pred_credit = [0.0] * pred_count
    gt_credit = 0.0
    for match in result["anchor_matches"]:
        coverage = match["coverage"]
        gt_credit += coverage
        for index in match["predicted_anchor_indices"]:
            # Max prevents one predicted marker from earning duplicate precision
            # credit when it supports multiple GT markers.
            pred_credit[index] = max(pred_credit[index], coverage)
    precision = base.safe_div(sum(pred_credit), pred_count)
    recall = base.safe_div(gt_credit, gt_count)
    f1 = base.f1_score(precision, recall)
    supported = [index for index, credit in enumerate(pred_credit) if credit]
    return {"pair_id": pair_id, "status": status, **result,
            "matched_gt_anchor_count": result["matched_anchor_count"],
            "supported_predicted_anchor_count": len(supported),
            "supported_predicted_anchor_indices": supported,
            "gt_anchor_count": gt_count, "predicted_anchor_count": pred_count,
            "gt_coverage_sum": gt_credit, "predicted_credit_sum": sum(pred_credit),
            "predicted_anchor_credits": pred_credit,
            "anchor_precision": precision, "anchor_recall": recall,
            "anchor_f1": f1, "anchor_score": f1}


def install():
    """Reuse the existing evaluator while replacing only anchor-specific hooks."""
    evaluator.code_span_request = code_span_request
    evaluator.validate_judge_result = validate_judge_result
    evaluator.cached_anchor_judgement = cached_anchor_judgement


def main():
    install()
    evaluator.main()


if __name__ == "__main__":
    main()
