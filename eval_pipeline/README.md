# 最终评测逻辑

统一使用根目录 `run_eval.sh`；完整推理、转换和评测方法见 [项目说明](../README.md)。

- `evaluate_i2c_hallu.py`：基础解析、行号与 API 工具。
- `evaluate_i2c_hallu_set_match.py`：一对一集合语义匹配、行号评分、聚合与完整性信息。
- `evaluate_i2c_hallu_localization_half_credit.py`：锚点 0/0.5/1 评分，版本 code_span_v6_half_credit_20261026；当前默认入口。
- `conversion/`：从 JSONL/Excel 转为标准预测，并验证 code_span。
- `prompt/`：judge 提示词。

