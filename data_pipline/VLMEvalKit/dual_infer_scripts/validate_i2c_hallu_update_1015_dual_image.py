#!/usr/bin/env python3
"""Validate the complete 1015 dataset and selected prompt without inference."""
import json
from pathlib import Path
import sys

KIT = Path(__file__).resolve().parents[1]
ROOT = KIT.parents[1]
sys.path.insert(0, str(KIT))


def main(version):
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.i2c_hallu import INPUT_BLOCK_MARKER, build_dual_image_question

    assert version == 'v8'
    name = f'I2C_Hallu_Benchmark_Update_1015_{version.upper()}_Dual_Image'
    bundle = ROOT / 'benchmark_update_1015'
    metadata = json.loads((bundle / 'metadata.json').read_text())
    dataset = build_dataset(name)
    assert len(dataset.data) == metadata['record_count'] == 1040
    ids = (bundle / 'benchmark_ids.txt').read_text().splitlines()
    assert dataset.data['record_id'].tolist() == ids
    for i, row in enumerate(dataset.data.to_dict(orient='records')):
        prompt = dataset.build_prompt(i)
        assert row['index'] == ids[i]
        assert [part['type'] for part in prompt] == ['image', 'image', 'text']
        assert [part['value'] for part in prompt[:2]] == [row['gt_image_path'], row['candidate_image_path']]
        for field in ('gt_image_path', 'candidate_image_path', 'candidate_code_path'):
            assert Path(row[field]).is_file(), (i, field)
        assert prompt[2]['value'] == build_dual_image_question(row, version)
        assert INPUT_BLOCK_MARKER not in prompt[2]['value']
    print(f'Validated {len(ids)} records, {version} prompt, GT/candidate image order')


if __name__ == '__main__':
    main(sys.argv[1])
