import os
import re
from pathlib import Path

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset


DATASET_NAME = 'I2C_Reproduction_Training_V1_8K'
DATA_TSV = Path(
    '/mnt/disk1/data_group/zhoujiawei/i2c_hallu_data/'
    'I2C_Reproduction_Training_V1_8K.tsv'
)


def strip_code_fence(value):
    """Return generated source code without an optional Markdown fence."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    match = re.fullmatch(r'```[^\n]*\n(.*?)\n```', text, flags=re.DOTALL)
    return match.group(1) if match else text


class I2CReproduction(ImageBaseDataset):
    """GT-image-to-code inference dataset for the 8K reproduction corpus."""

    TYPE = 'VQA'
    DATASET_URL = {DATASET_NAME: str(DATA_TSV)}

    def load_data(self, dataset):
        if dataset not in self.DATASET_URL:
            raise ValueError(f'Unsupported I2C reproduction dataset: {dataset}')

        data_path = Path(self.DATASET_URL[dataset])
        if not data_path.is_file():
            raise FileNotFoundError(f'I2C reproduction TSV not found: {data_path}')

        # The source TSV embeds base64 images and is larger than 1 GB. Follow
        # ImageBaseDataset.prepare_tsv by using a localized image-path TSV.
        if data_path.stat().st_size > 1024 ** 3:
            local_path = data_path.with_name(f'{data_path.stem}_local.tsv')
            if not local_path.is_file() or os.environ.get('FORCE_LOCAL'):
                from ..tools import LOCALIZE

                LOCALIZE(str(data_path), str(local_path))
            data_path = local_path

        data = load(str(data_path))
        if isinstance(data, list):
            data = pd.DataFrame(data)

        required = {'index', 'question', 'code_type'}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f'I2C reproduction TSV is missing columns: {missing}')
        if 'image' not in data and 'image_path' not in data:
            raise ValueError('I2C reproduction TSV requires image or image_path')
        if data['index'].astype(str).duplicated().any():
            raise ValueError('I2C reproduction TSV contains duplicate indices')
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            image_paths = toliststr(line['image_path'])
        else:
            image_paths = self.dump_image(line)

        messages = [dict(type='image', value=path) for path in image_paths]
        messages.append(dict(type='text', value=str(line['question'])))
        return messages

    def evaluate(self, eval_file, **judge_kwargs):
        """Export raw predictions as JSONL when evaluation is requested."""
        data = load(eval_file)
        records = data.to_dict(orient='records') if hasattr(data, 'to_dict') else data
        for record in records:
            record['generated_code'] = strip_code_fence(record.get('prediction'))

        result_file = get_intermediate_file_path(eval_file, '_generated_code', 'jsonl')
        dump(records, result_file)
        return {'total_count': len(records), 'result_file': result_file}
