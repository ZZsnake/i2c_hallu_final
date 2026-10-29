# I2C-Hallu 最终推理与评测

默认数据为 `benchmark_update_1015`，1040 条（positive 289、negative 751）；推理文本仅使用 `data_pipline/prompt_v8_refined_dual_image.py`。现有 Python 环境为 `/root/miniforge3/envs/image2code`；模型权重继续读取现有训练目录。

## 目录与入口

- `benchmark_update_1015/`：benchmark.jsonl、ID 清单及完整图像/代码资源。
- `data_pipline/VLMEvalKit/`：推理框架、模型兼容补丁、最终版本模型启动脚本。
- `eval_pipeline/`：更新的集合匹配、0/0.5/1 锚点评分及单元测试。
- `eval_pipeline/conversion/`：独立 JSONL/Excel 预测转换器。
- `run_api.sh`、`run_local.sh`：推理；`run_convert.sh`：转换；`run_eval.sh`：使用当前 benchmark 评测。
- `eval_pipeline/run_evaluation.sh`：评测入口依赖的完整运行及结果检查脚本。

## 运行依赖

根目录 `requirements.txt` 统一列出推理、JSONL/Excel 转换、评测及 VLMEvalKit 共享导入所需的 79 个包，固定为现有 `image2code` 环境中的版本。基线：Python 3.10.18、Linux x86_64、PyTorch CUDA 12.8。pip 会继续安装这些包的传递依赖；本文件不是完整环境锁定文件。

已有环境可以直接使用，无需重新安装。重建同类环境时，在目标 Python 环境中执行：

```bash
python -m pip install -r requirements.txt
```

核心组合为 torch 2.10.0、torchvision 0.25.0、torchaudio 2.10.0、vLLM 0.19.0、Transformers 4.57.6。Toolkit 内的 requirements 转引根目录，旧 torch 2.8 约束已更新。需要可用的 NVIDIA GPU 驱动；系统驱动和 Conda/Python 本身不由 pip requirements 安装。

依赖核查中，现有环境的 `pip check` 唯一提示是 decord 0.6.0 的已安装 wheel 标记为 cp36，与当前 Python 3.10 不匹配。推理启动器和评测/转换导入检查均通过。重建环境时需从匹配当前 Python/平台的发行包安装 decord；本次未改动现有环境。当前任务处理双图，不使用视频解码。

## 推理

```bash
cd /mnt/disk1/data_group/zhoujiawei/i2c_hallu_final
# 输入验证，不运行正式推理或调用 API
DRY_RUN=1 bash run_api.sh
DRY_RUN=1 bash run_local.sh

# GPT-6-Sol / Micu，默认 16 workers
read -rsp 'MICU_API_KEY: ' MICU_API_KEY; echo
export MICU_API_KEY
bash run_api.sh

# Qwen35-27B checkpoint-1320，GPU 0,1,2,3，TP4
bash run_local.sh
```

本地任务按剩余显存分配预算，默认 batch=4、上下文 32768、最大输出 16384。`CHECKPOINT_PATH` 可覆盖权重路径。使用现有环境，无需重装依赖。

输出位于 `data_pipline/VLMEvalKit/outputs/benchmark_update_1015_v8_prompt_<prompt SHA256>/`：API 在 `gpt-6-sol_api_dual_image/`，本地在 `local/<model>/T<run>/`。两者会导出 Excel 和 `*_gen_results.jsonl`，日志位于 `data_pipline/VLMEvalKit/log/`。重新运行相同任务会复用断点；修改 prompt 后结果目录自动变化。

## 转换推理结果

推荐先转换和检查，再提交评测。支持 JSONL 或 Excel；使用独立的输出文件，转换失败时返回非零。

```bash
mkdir -p eval_pipeline/inputs
bash run_convert.sh \
 --input-file /absolute/path/to/predictions_gen_results.jsonl \
 --output-file "$PWD/eval_pipeline/inputs/model_eval.jsonl"
```

转换将 `errors` 转为 `is_consistent/gold_errors`，保留 record_id，检查并记录 code_span 修正；已有规范预测可直接评测。确认终端显示 `Conversion failures: 0`。

## 评测

更新包的 judge 默认是 DeepSeek：base_url=`https://api.deepseek.com`，model=`deepseek-flash`，16 workers，关闭 thinking。它使用 `DEEPSEEK_API_KEY`，与推理的 `MICU_API_KEY` 分开。

```bash
# 只准备请求，不调用 judge API
bash run_eval.sh "$PWD/eval_pipeline/inputs/model_eval.jsonl" --dry-run --limit 2

# 完整评测
read -rsp 'DEEPSEEK_API_KEY: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
bash run_eval.sh "$PWD/eval_pipeline/inputs/model_eval.jsonl"

# 同一命令和输出目录重跑，复用缓存
EVAL_WORKERS=16 bash run_eval.sh "$PWD/eval_pipeline/inputs/model_eval.jsonl"
```

`BASE_URL`、`MODEL` 可覆盖 judge 服务；其它 key 变量可追加 `--api-key-env VARIABLE_NAME`。更换 judge 模型或评分规则时指定新的 `EVAL_OUTPUT_DIR`。

默认输出：`eval_pipeline/outputs/code_span_v6_half_credit_20261026/<预测文件名>/`，包含 `summary.json`、`per_sample.jsonl`、`judge_cache/set_match_cache.jsonl` 和 `judge_cache/code_span_cache.jsonl`。请为不同模型的预测使用不同文件名或显式设置 `EVAL_OUTPUT_DIR`，避免结果目录冲突。

入口限制到 GT ID 集合，完成后检查重复预测、缺失预测、缺失/无效 judge 结果；不完整时返回非零，重跑同一命令补齐。默认使用独立的新评分缓存目录，旧二值锚点缓存不能混用。

显式四路径形式：

```bash
bash eval_pipeline/run_evaluation.sh PREDICTIONS.jsonl GT.jsonl CANDIDATE_CODE_DIR OUTPUT_DIR [评测参数]
```

## 评分逻辑

1. 按 record_id 匹配 benchmark，解析参考错误与预测错误；一致性由错误列表是否为空决定。
2. judge 对两组错误做语义一对一匹配，得到错误发现 precision、recall、F1；category/subtype 不作为硬性匹配条件。
3. 匹配错误的行定位按行号集合计算 F1。锚点按因果定位覆盖程度给 0、0.5、1 分：完整定位记 1，定位部分必要因果内容记 0.5，仅名称/容器或错误位置记 0。
4. 锚点 recall 为 GT 覆盖分均值；precision 为每个预测锚点最高贡献分的均值，然后计算 F1。单个错误的定位分为行 F1 与锚点 F1 的几何平均。端到端定位对所有可评测 GT 错误平均，漏检记 0。

锚点评分版本：`code_span_v6_half_credit_20261026`。详细实现见 `evaluate_i2c_hallu_localization_half_credit.py`。

## 后台运行

```bash
tmux new -s i2c_final_api
# 在 tmux 内输入并 export MICU_API_KEY 后：
bash run_api.sh
# Ctrl+B，再按 D，退出会话并保持运行

tmux new -s i2c_final_eval
# 在 tmux 内输入并 export DEEPSEEK_API_KEY 后：
bash run_eval.sh "$PWD/eval_pipeline/inputs/model_eval.jsonl"
```

## 验证记录

迁移时随包的 7 项测试全部通过；1040 条预测转换成功，评测 dry-run 成功。当前目录只保留运行入口及依赖，历史实验、修复、重跑、测试脚本已清理。清理后重新验证入口、转换和评测请求准备。校验记录见 VALIDATION.json；评测源码校验清单位于 eval_pipeline/CURRENT_SOURCE_MANIFEST.sha256。
