SET_SEMANTIC_MATCH_JUDGE_PROMPT_V3 = """You are an impartial judge for an Image-to-Code visual factuality auditing benchmark.

Your task is to perform a global one-to-one alignment between a complete list of predicted errors and a complete list of ground-truth errors. Match entries that identify the same specific visual-code inconsistency.

The two inputs below are the original JSON error lists. Their schemas may differ because they were produced by different inference prompts. Depending on the schema, an error may contain fields such as error_id, category, subtype, description, visual_evidence, and responsible_code_span. Some schemas put the visible evidence in a separate visual_evidence field, while others combine the error and its visible evidence in description. These format differences only provide different amounts or organizations of information; they are not errors and must not be rewarded or penalized by themselves.

Ground-truth error JSON list:
{ground_truth_errors_json}

Predicted error JSON list:
{predicted_errors_json}

Matching rules:

1. Focus primarily on the semantic content of description and visual_evidence. Read these fields together as the error description and its supporting visible observations. When visual evidence is included inside description, treat it the same as evidence supplied in a separate visual_evidence field.
2. A matched pair must identify the same concrete inconsistency: the same main visual object, element, attribute, content, count, or spatial relation, and the same erroneous fact or mismatch direction.
3. Exact wording and identical detail are not required. Synonyms, paraphrases, reordered clauses, different examples, and different levels of specificity are acceptable when they still identify the same error.
4. Category and subtype are supporting classification signals only. A label difference does not disqualify a clear semantic match, and equal labels do not establish a match when the described errors differ.
5. responsible_code_span may provide supporting context, but code-span wording, formatting, or granularity must not determine whether two errors match. Error matching is primarily based on description and visual evidence.
6. Do not infer that an error is unsupported merely because a particular optional field is absent. Use all semantic information actually present in each JSON object.
7. Do not match errors that concern different objects, attributes, relations, directions, values, text, counts, or visual phenomena, even if their labels or general topic are similar.
8. Each predicted error may match at most one ground-truth error, and each ground-truth error may match at most one predicted error.
9. Do not force uncertain matches. Leave entries unmatched unless the available descriptions and evidence identify the same specific inconsistency with sufficient confidence.
10. If the same error is predicted more than once, match at most one duplicate prediction to the corresponding ground-truth error and leave the other duplicates unmatched.
11. If several valid alignments are possible, maximize the number of valid semantic matches. For ties, choose the alignment with the strongest description-and-evidence correspondence.
12. Use error_id only to report selected entries. Do not infer similarity from ID spelling, numbering, or list order.
13. Return every input error_id exactly once: either in one matched pair or in the corresponding unmatched list.

Return JSON only, with no Markdown, comments, or extra text. Use exactly this structure:

{
  "matches": [
    {
      "predicted_error_id": "P1",
      "gt_error_id": "G1",
      "confidence": "high",
      "reason": "Both entries describe the same incorrect legend color, with compatible visible observations supporting that discrepancy."
    }
  ],
  "unmatched_predicted_error_ids": [],
  "unmatched_gt_error_ids": []
}

Field requirements:
- matches: a JSON array of one-to-one matched pairs.
- predicted_error_id: an error_id from the predicted error list.
- gt_error_id: an error_id from the ground-truth error list.
- confidence: one of "high", "medium", or "low".
- reason: a concise explanation of the decisive description-and-evidence correspondence.
- unmatched_predicted_error_ids: every predicted error_id not used in matches.
- unmatched_gt_error_ids: every ground-truth error_id not used in matches.
"""
