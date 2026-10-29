#!/usr/bin/env python3
"""Run VLMEvalKit with vLLM's Qwen3.5 config on Transformers 4.57.

Registration is process-local and also runs when vLLM spawns workers.
"""

from transformers import AutoConfig, AutoVideoProcessor, Qwen3VLVideoProcessor
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
AutoVideoProcessor.register(Qwen3_5Config, Qwen3VLVideoProcessor, exist_ok=True)

if __name__ == "__main__":
    from run import main
    main()
