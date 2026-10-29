"""Refined V8 dual-image prompt for reliable visual-error detection.

Attach the ground-truth image first and the rendered candidate image second.
Replace [[INPUT_BLOCK]] with JSON containing figure_type, code_type, and
candidate_code_lines. Each code line has an authoritative 1-based l_n and
exact source text t.

This version keeps V8's optional causal code_span and JSON output format.
Its visual-evidence and duplicate-error rules are tightened using V7's
guidance, without making code localization a condition for reporting an error.
"""

prompt = """
You audit a candidate rendering against a ground-truth image. The images are
provided in that order. The input includes `figure_type`, `code_type`, and
`candidate_code_lines`, where each line has a 1-based `l_n` and exact text `t`.

Compare the images before inspecting the code. Report each distinct visual
discrepancy only when it is clear and salient at normal viewing size. A clear,
salient difference can qualify even when it does not change meaning, data, or
readability: for example, a conspicuous color change, substantially different
stroke width, or noticeable shift in position. Do not report slight shade or
font variations, tiny offsets, anti-aliasing, pixel noise, or rendering
artifacts. If the visual evidence is uncertain, omit the difference.

The ground-truth image defines what should appear. For every reported error,
describe the concrete contrast between the ground-truth and candidate images.
Use the code only to locate the cause of an observed difference; never infer a
visual error from suspicious code alone.

Assign exactly one category to each error:

- `entity_error`: a visual element is missing, extra, or duplicated.
- `attribute_content_error`: an existing element has wrong text, data, scale,
  shape, or semantic color mapping.
- `layout_relation_error`: position, order, orientation, connection, or
  spatial relation differs.
- `style_rendering_error`: appearance differs clearly while entities,
  content, and relations remain correct; this includes a conspicuous color
  change without semantic impact.

Boundary rules:

- A missing or extra label is `entity_error`; an existing label with wrong
  text is `attribute_content_error`.
- A missing or extra line or arrow is `entity_error`; an existing one with
  the wrong connection or direction is `layout_relation_error`.
- Wrong encoded data values and axis scales are `attribute_content_error`,
  even when they also change positions.

Count one error per distinct visual cause. Merge repeated manifestations of
the same mismatch, such as several shifted marks caused by one wrong axis
scale. Keep independent differences separate, even when they affect the same
object. Do not split one mismatch into several descriptions of its effects.
Do not impose a fixed limit on the number of errors.

For each error, locate the smallest causal code span using only the supplied
`l_n` values. Each marker must contain `start_line`, `end_line`, and an `anchor`
copied verbatim from the joined `t` text in that range. Omit `code_span` if
the cause cannot be located confidently; still report the visible error.
Before returning the result, remove speculative or duplicate errors. Keep
every distinct clear error even when its code cause is uncertain.

Return only JSON: `{"errors":[]}` if there are no qualifying differences,
otherwise `{"errors":[{"category":"...","description":"...",
"code_span":[{"start_line":1,"end_line":1,"anchor":"..."}]}]}`.
`code_span` is optional for each error. Add no other fields.

Input:
[[INPUT_BLOCK]]
"""
