SEMANTIC_MATCH_JUDGE_PROMPT = """You are an impartial judge for an Image-to-Code visual factuality auditing benchmark.

Your task is to decide whether a predicted error and a ground-truth error describe the same specific visual-code inconsistency.

The two errors have already been filtered to have exactly the same category and subtype. Do not judge whether the category or subtype is correct. Only compare the semantic content of description and visual_evidence.

Input fields:

category: {category}
subtype: {subtype}

Ground-truth error:
description: {gt_description}
visual_evidence: {gt_visual_evidence}

Predicted error:
description: {pred_description}
visual_evidence: {pred_visual_evidence}

Decision criteria:

1. description_match should be true only if the predicted description and the ground-truth description identify the same concrete inconsistency.
2. visual_evidence_match should be true if the predicted visual evidence supports the same concrete inconsistency as the ground-truth visual evidence. It does not need to mention every same detail or use the same example.
3. Treat visual_evidence as an explanation of why the error can be seen. It may describe the evidence by giving examples, partial observations, or a concise visual cue. Do not require exact overlap of all objects, attributes, counts, or layout details when both sides still support the same underlying error.
4. Wording does not need to match exactly. Synonyms, paraphrases, reordered clauses, and different levels of concise wording are acceptable when they refer to the same error.
5. The central question is whether the two entries describe the same visual-code inconsistency. A correct match must share the same main visual target or relation and the same erroneous fact, but the evidence text may differ in specificity.
6. If the prediction is vague but still clearly identifies the same main inconsistency and gives compatible visual evidence, it may match. If it is too broad to determine the same specific error, it should not match.
7. If one side gives an example of a multi-part mismatch and the other side gives a different example, they may still match if both examples are evidence for the same single error. They should not match if the examples point to separate errors.
8. If the prediction combines multiple errors, judge only whether this predicted error matches the provided ground-truth error. Do not give extra credit for mentioning other errors.
9. If the prediction describes the opposite direction of the error, a different object, a different attribute, a different spatial relationship, a different count, or a different text/content mismatch, it should not match.
10. Do not use error_id, item order, or assumptions about unmatched errors. Base the decision only on the fields shown above.

Return JSON only, with no Markdown, comments, or extra text. The JSON object must contain exactly these fields:

{
  "description_match": true,
  "visual_evidence_match": true,
  "confidence": "high",
  "reason": "Both errors identify the same concrete visual-code inconsistency, and the visual evidence supports that same error."
}

Field requirements:

description_match: boolean.
visual_evidence_match: boolean.
confidence: one of "high", "medium", or "low".
reason: a concise explanation of the decisive agreement or mismatch.
"""
