#!/usr/bin/env python3
"""Check local model config, processor, weight shards, and TP4 compatibility."""
import json
from pathlib import Path
import sys
from transformers import AutoConfig, AutoProcessor, AutoVideoProcessor, Qwen3VLVideoProcessor
from vllm.model_executor.models import ModelRegistry

p = Path(sys.argv[1])
raw = json.loads((p / 'config.json').read_text())
if raw['model_type'] == 'qwen3_5':
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
    AutoConfig.register('qwen3_5', Qwen3_5Config, exist_ok=True)
    AutoVideoProcessor.register(Qwen3_5Config, Qwen3VLVideoProcessor, exist_ok=True)
config = AutoConfig.from_pretrained(p, local_files_only=True)
processor = AutoProcessor.from_pretrained(p, local_files_only=True)
assert config.architectures[0] in ModelRegistry.get_supported_archs(), config.architectures
text = raw.get('text_config', raw)
for field in ('num_attention_heads', 'num_key_value_heads'):
    assert text[field] % 4 == 0, (field, text[field], 'TP4')
shards = set(json.loads((p / 'model.safetensors.index.json').read_text())['weight_map'].values())
assert all((p / shard).is_file() for shard in shards), 'Missing weight shards'
print(f'Model preflight passed: {p}; architecture={config.architectures[0]}; processor={type(processor).__name__}; shards={len(shards)}; TP4 compatible')
