import os.path as osp
import random as rd
import string

import pandas as pd

from vlmeval.smp import dump, gpt_key_set, load
from vlmeval.smp.log import get_logger
from ..dataset.utils.judge_util import build_judge
from ..dataset.utils.multiple_choice import extract_answer_from_item
from .matching_util import can_infer
from .mp_util import track_progress_rich

logger = get_logger(__name__)


def MMMU_result_transfer(result_path):
    result = {}
    result_data = load(result_path)
    mcq = result_data['A'].notna()
    for index in range(len(result_data)):
        line = result_data.iloc[index]
        if mcq.iloc[index]:
            options = {
                candidate: line[candidate]
                for candidate in string.ascii_uppercase
                if candidate in line and not pd.isna(line[candidate])
            }
            result[line['id']] = can_infer(line['prediction'], options)
        else:
            result[line['id']] = line['prediction']

    result_json = result_path.replace('.xlsx', '.json')
    dump(result, result_json)
    return result_json


def MMTBench_result_transfer(eval_file, dataset='default', **judge_kwargs):
    del dataset
    nproc = judge_kwargs.pop('nproc', 4)

    rd.seed(2680)
    suffix = eval_file.split('.')[-1]
    model_name = judge_kwargs['model']
    name_map = {
        'chatgpt-0125': 'openai',
        'gpt-4-0125': 'gpt4',
    }
    name = name_map.get(model_name, model_name)

    if model_name == 'exact_matching':
        model = None
    elif gpt_key_set():
        model = build_judge(**judge_kwargs)
        if not model.working():
            logger.error(
                'The OpenAI API is not working properly; '
                'falling back to exact matching for evaluation.'
            )
            model = None
    else:
        logger.error(
            'OPENAI_API_KEY is not set properly; '
            'falling back to exact matching for evaluation.'
        )
        model = None

    logger.info(f'Evaluating {eval_file}')
    result_file = eval_file.replace(f'.{suffix}', f'_{name}_option.pkl')
    result = load(result_file) if osp.exists(result_file) else {}

    data = load(eval_file)
    assert 'index' in data, 'Essential columns missing in the eval_file.'

    data = data.sort_values(by='index')
    data['prediction'] = [str(value) for value in data['prediction']]
    for key in list(data.keys()):
        normalized_key = key.lower() if key not in string.ascii_uppercase else key
        data[normalized_key] = data.pop(key)

    index_to_line = {
        data.iloc[index]['index']: data.iloc[index]
        for index in range(len(data))
    }
    index_to_line = {
        key: value for key, value in index_to_line.items()
        if key not in result
    }

    indices = list(index_to_line.keys())
    lines = [index_to_line[index] for index in indices]
    tasks = [(model, line) for line in lines]
    outputs = track_progress_rich(
        extract_answer_from_item,
        tasks,
        nproc=nproc,
        chunksize=nproc,
        save=result_file,
        keys=indices,
    )

    for index, output in zip(indices, outputs):
        if index in result:
            assert result[index]['opt'] == output['opt']
            assert result[index]['log'] == output['log']
        else:
            result[index] = output

    data['opt'] = [result[index]['opt'] for index in data['index']]
    data['log'] = [result[index]['log'] for index in data['index']]

    output_path = eval_file.replace(f'.{suffix}', f'_{name}_submission.tsv')
    dump(data, output_path)
    return output_path
