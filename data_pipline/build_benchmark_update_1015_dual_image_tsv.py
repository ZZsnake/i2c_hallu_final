"""Build the 1015 dual-image TSV directly from the final release manifest."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'benchmark_update_1015'
OUTPUT = ROOT / 'data_pipline/i2c_hallu_benchmark_update_1015_dual_image.tsv'
DATASET = 'I2C_Hallu_Benchmark_Update_1015_Dual_Image'
COLUMNS = (
    'index', 'record_id', 'source_sample_id', 'image_path', 'question', 'answer',
    'image_type', 'figure_type', 'code_type', 'candidate_code_path',
    'candidate_image_path', 'gt_image_path', 'group_id', 'benchmark_label',
    'is_consistent', 'source_model', 'candidate_source',
)


def build_rows():
    metadata = json.loads((BUNDLE / 'metadata.json').read_text(encoding='utf-8'))
    ids = (BUNDLE / 'benchmark_ids.txt').read_text(encoding='utf-8').splitlines()
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in benchmark_ids.txt')
    rows, seen, verified = [], set(), {}
    with (BUNDLE / 'benchmark.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            rid = record['record_id']
            if rid in seen:
                raise ValueError(f'Duplicate record_id: {rid}')
            seen.add(rid)
            for field in ('gt_image_path', 'candidate_image_path', 'candidate_code_path'):
                relative = Path(record[field])
                path = (ROOT / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(BUNDLE.resolve()):
                    raise ValueError(f'{rid}: resource outside release: {relative}')
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path not in verified:
                    verified[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                expected = record['asset_hashes'][field.replace('_path', '_sha256')]
                if verified[path] != expected:
                    raise ValueError(f'{rid}: SHA256 mismatch: {path}')
            label = record['benchmark_label']
            if label not in ('positive', 'negative') or record['is_consistent'] is not (label == 'positive'):
                raise ValueError(f'{rid}: inconsistent release label')
            rows.append({
                'index': rid, 'record_id': rid,
                'source_sample_id': record['sample_id'],
                'image_path': json.dumps([record['gt_image_path'], record['candidate_image_path']], ensure_ascii=False),
                'question': f'Constructed dynamically from prompt_v8_refined_dual_image.py by {DATASET}.',
                'answer': '', 'image_type': record['figure_type'],
                'figure_type': record['figure_type'], 'code_type': record['code_type'],
                'candidate_code_path': record['candidate_code_path'],
                'candidate_image_path': record['candidate_image_path'],
                'gt_image_path': record['gt_image_path'], 'group_id': record['group_id'],
                'benchmark_label': label, 'is_consistent': str(record['is_consistent']).lower(),
                'source_model': record.get('source_model') or rid.split('::')[1],
                'candidate_source': record['release_metadata']['candidate_source'],
            })
    if seen != set(ids) or len(rows) != metadata['record_count']:
        raise ValueError('Release records differ from benchmark_ids.txt or metadata record_count')
    if dict(Counter(row['benchmark_label'] for row in rows)) != metadata['labels']:
        raise ValueError('Label counts differ from release metadata')
    return rows


def main():
    rows = build_rows()
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                     dir=OUTPUT.parent, prefix=OUTPUT.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter='\t',
                                quoting=csv.QUOTE_ALL, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(OUTPUT)
    print(f'Wrote {len(rows)} rows to {OUTPUT}; resource SHA256 verified')


if __name__ == '__main__':
    main()
