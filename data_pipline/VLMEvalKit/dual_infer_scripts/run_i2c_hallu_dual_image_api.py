#!/usr/bin/env python3
"""V8 refined dual-image API inference, adapted from batch_api_runner_template.py."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import signal
import sys
from threading import Event, current_thread
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_api_helpers import (
    StatusLogger, parse_extra_params, content_part_text, serialize_usage_value,
    normalize_token_usage, empty_token_totals, add_item_token_usage, get_thread_client,
)
DATASET = 'I2C_Hallu_Benchmark_Update_1015_V8_Dual_Image'


def messages_for(prompt):
    content = []
    for part in prompt:
        if part['type'] == 'image':
            path = Path(part['value'])
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            content.append({'type': 'image_url', 'image_url': {
                'url': f'data:{mime};base64,{encoded}', 'detail': 'high'}})
        else:
            content.append({'type': 'text', 'text': part['value']})
    return [{'role': 'user', 'content': content}]


def anthropic_messages_for(prompt):
    content = []
    for part in prompt:
        if part['type'] == 'image':
            path = Path(part['value'])
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            content.append({'type': 'image', 'source': {
                'type': 'base64', 'media_type': mime,
                'data': base64.b64encode(path.read_bytes()).decode('ascii'),
            }})
        else:
            content.append({'type': 'text', 'text': part['value']})
    return [{'role': 'user', 'content': content}]


def stream_completion_content(response, logger, sid, attempt, stop):
    parts, usage, finish_reason = [], None, None
    chunk_count, chars, last_log = 0, 0, time.monotonic()
    with response:
        for chunk in response:
            if stop.is_set():
                raise InterruptedError('Inference stop requested')
            chunk_count += 1
            if getattr(chunk, 'usage', None) is not None:
                usage = serialize_usage_value(chunk.usage)
            choices = getattr(chunk, 'choices', None)
            if choices:
                choice = choices[0]
                text = content_part_text(getattr(choice.delta, 'content', None))
                parts.append(text)
                chars += len(text)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            now = time.monotonic()
            if chunk_count == 1:
                logger.emit('api_stream_first_chunk', sid, attempt=attempt)
            elif now - last_log >= 10:
                logger.emit('api_stream_receiving', sid, attempt=attempt,
                            chunks=chunk_count, response_chars=chars)
                last_log = now
    logger.emit('api_stream_finished', sid, attempt=attempt, chunks=chunk_count,
                response_chars=chars, finish_reason=finish_reason,
                token_usage=normalize_token_usage(usage))
    return ''.join(parts), usage, finish_reason


def error_fields(exc):
    # Keep errors useful without dumping request bodies or credentials.
    fields = {'error_type': type(exc).__name__}
    if getattr(exc, 'status_code', None) is not None:
        fields['status_code'] = exc.status_code
    if isinstance(exc, (ValueError, InterruptedError)):
        fields['error'] = str(exc)[:500]
    return fields


def call_with_retries(client, request_args, args, logger, sid, stop):
    for attempt_index in range(args.retries + 1):
        if stop.is_set():
            raise InterruptedError('Inference stop requested')
        attempt = attempt_index + 1
        started = time.monotonic()
        logger.emit('api_request_start', sid, attempt=attempt, stream=args.stream)
        try:
            if args.stream:
                stream_args = {'stream': True}
                if args.stream_include_usage:
                    stream_args['stream_options'] = {'include_usage': True}
                response = client.chat.completions.create(**request_args, **stream_args)
                raw, usage, reason = stream_completion_content(response, logger, sid, attempt, stop)
            else:
                completion = client.chat.completions.create(**request_args)
                raw = content_part_text(completion.choices[0].message.content)
                usage = serialize_usage_value(getattr(completion, 'usage', None))
                reason = completion.choices[0].finish_reason
            logger.emit('api_request_done', sid, attempt=attempt,
                        elapsed_seconds=round(time.monotonic()-started, 3),
                        response_chars=len(raw), finish_reason=reason,
                        token_usage=normalize_token_usage(usage))
            if reason != 'stop':
                raise ValueError(f'Incomplete response: finish_reason={reason}')
            if not raw.strip():
                raise ValueError('Empty API response')
            return raw, usage, reason
        except Exception as exc:
            logger.emit('api_request_error', sid, attempt=attempt, **error_fields(exc))
            status = getattr(exc, 'status_code', None)
            permanent = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
            if attempt_index >= args.retries or permanent or stop.is_set() or isinstance(exc, InterruptedError):
                raise
            delay = args.retry_delay * 2**attempt_index
            logger.emit('api_retry_sleep', sid, attempt=attempt, sleep_seconds=delay)
            if stop.wait(delay):
                raise InterruptedError('Inference stop requested')
    raise RuntimeError('unreachable retry state')


def process_record(openai_cls, args, api_key, request_base, record, prompt, logger, stop):
    from vlmeval.dataset.i2c_hallu import normalize_prediction_record_code_spans
    item = dict(record)
    sid = str(item['index'])
    item.update(sample_id=sid, api_success=False, status='error')
    started = time.monotonic()
    try:
        logger.emit('sample_start', sid, worker=current_thread().name)
        client = get_thread_client(openai_cls, args.base_url, api_key, args.timeout)
        request = {**request_base, 'messages': messages_for(prompt)}
        logger.emit('request_payload_ready', sid, worker=current_thread().name)
        raw, usage, reason = call_with_retries(client, request, args, logger, sid, stop)
        item.update(prediction=raw, prediction_raw=raw, api_usage=usage, usage=usage,
                    token_usage=normalize_token_usage(usage), api_finish_reason=reason)
        item = normalize_prediction_record_code_spans(item)
        if item['prediction_conversion_status'] != 'ok':
            raise ValueError(item['prediction_conversion_error'])
        item.update(api_success=True, status='ok', api_error_type=None)
        logger.emit('sample_done', sid, status='ok', token_usage=item['token_usage'],
                    worker=current_thread().name)
    except Exception as exc:
        item['api_error_type'] = type(exc).__name__
        item['api_error'] = error_fields(exc)
        logger.emit('sample_error', sid, worker=current_thread().name, **error_fields(exc))
    item['elapsed_seconds'] = round(time.monotonic()-started, 3)
    item['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    return item


def process_record_anthropic(anthropic_cls, args, api_key, request_base, record, prompt, logger, stop):
    from vlmeval.dataset.i2c_hallu import normalize_prediction_record_code_spans
    item = dict(record)
    sid = str(item['index'])
    item.update(sample_id=sid, api_success=False, status='error')
    started = time.monotonic()
    try:
        logger.emit('sample_start', sid, worker=current_thread().name)
        client = get_thread_client(anthropic_cls, args.base_url, api_key, args.timeout)
        request = {**request_base, 'messages': anthropic_messages_for(prompt)}
        logger.emit('request_payload_ready', sid, worker=current_thread().name)
        for attempt_index in range(args.retries + 1):
            if stop.is_set():
                raise InterruptedError('Inference stop requested')
            attempt = attempt_index + 1
            request_started = time.monotonic()
            logger.emit('api_request_start', sid, attempt=attempt, stream=False)
            try:
                message = client.messages.create(**request)
                raw = ''.join(block.text for block in message.content
                              if getattr(block, 'type', None) == 'text')
                usage = serialize_usage_value(getattr(message, 'usage', None))
                reason = getattr(message, 'stop_reason', None)
                logger.emit('api_request_done', sid, attempt=attempt,
                            elapsed_seconds=round(time.monotonic()-request_started, 3),
                            response_chars=len(raw), finish_reason=reason,
                            token_usage=normalize_token_usage(usage))
                if reason not in ('end_turn', 'stop_sequence'):
                    raise ValueError(f'Incomplete response: stop_reason={reason}')
                if not raw.strip():
                    raise ValueError('Empty API response')
                break
            except Exception as exc:
                logger.emit('api_request_error', sid, attempt=attempt, **error_fields(exc))
                status = getattr(exc, 'status_code', None)
                permanent = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
                if attempt_index >= args.retries or permanent or stop.is_set() or isinstance(exc, InterruptedError):
                    raise
                delay = args.retry_delay * 2**attempt_index
                logger.emit('api_retry_sleep', sid, attempt=attempt, sleep_seconds=delay)
                if stop.wait(delay):
                    raise InterruptedError('Inference stop requested')
        item.update(prediction=raw, prediction_raw=raw, api_usage=usage, usage=usage,
                    token_usage=normalize_token_usage(usage), api_finish_reason=reason)
        item = normalize_prediction_record_code_spans(item)
        if item['prediction_conversion_status'] != 'ok':
            raise ValueError(item['prediction_conversion_error'])
        item.update(api_success=True, status='ok', api_error_type=None)
        logger.emit('sample_done', sid, status='ok', token_usage=item['token_usage'],
                    worker=current_thread().name)
    except Exception as exc:
        item['api_error_type'] = type(exc).__name__
        item['api_error'] = error_fields(exc)
        logger.emit('sample_error', sid, worker=current_thread().name, **error_fields(exc))
    item['elapsed_seconds'] = round(time.monotonic()-started, 3)
    item['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    return item


def load_completed(checkpoint):
    completed = {}
    if checkpoint.exists():
        for line_no, line in enumerate(checkpoint.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid checkpoint at {checkpoint}:{line_no}; restore the backup or repair the last line') from exc
            if item.get('api_success'):
                completed[str(item['index'])] = item
    return completed


def export_results(records, rows, output):
    import pandas as pd
    from vlmeval.dataset.i2c_hallu import normalize_prediction_record_code_spans
    ordered = [records[str(row['index'])] for row in rows if str(row['index']) in records]
    results = output / 'predictions_gen_results.jsonl'
    temp = results.with_suffix('.jsonl.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for record in ordered:
            stream.write(json.dumps(normalize_prediction_record_code_spans(dict(record)),
                                    ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(results)
    temp_xlsx = output / 'predictions.tmp.xlsx'
    pd.DataFrame(ordered).to_excel(temp_xlsx, index=False)
    temp_xlsx.replace(output / 'predictions.xlsx')
    print(f'Exported {len(ordered)} successful records: {output / "predictions.xlsx"}\n{results}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DATASET, choices=[DATASET])
    parser.add_argument('--model', default=os.getenv('API_MODEL', 'gpt-6-sol'))
    parser.add_argument('--provider', choices=['openai', 'anthropic'], default='openai')
    parser.add_argument('--base-url', default=os.getenv('API_BASE_URL', 'https://www.micuapi.ai/v1'))
    parser.add_argument('--api-key-env', default='MICU_API_KEY')
    parser.add_argument('--api-key-fallback-env', default='OPENAI_API_KEY',
                        help='Fallback key variable; pass an empty string to require --api-key-env')
    parser.add_argument('--env-file', type=Path, help='Load API keys from this dotenv file without overriding the environment')
    parser.add_argument('--workers', type=int, default=int(os.getenv('API_NPROC', '4')))
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--max-retries', '--retries', dest='retries', type=int, default=2)
    parser.add_argument('--retry-delay', type=float, default=3)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--token-field', choices=['max_completion_tokens', 'max_tokens'], default='max_completion_tokens')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--response-format', choices=['none', 'json_object'], default='none')
    parser.add_argument('--extra-param', action='append', default=[], metavar='KEY=JSON_VALUE')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--stream-include-usage', action='store_true', help='Request streaming token usage if the provider supports it')
    parser.add_argument('--work-dir', '--output-dir', dest='work_dir', type=Path)
    parser.add_argument('--log-file', type=Path)
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true', default=True, help='Resume successful records (always enabled)')
    parser.add_argument('--validate-only', '--dry-run', dest='validate_only', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='Export saved results without API calls or a key')
    parser.add_argument('--shutdown-grace', type=float, default=15, help='Seconds to drain in-flight requests on Ctrl+C/SIGTERM')
    args = parser.parse_args()
    if args.workers < 1 or args.max_tokens < 1 or args.retries < 0 or args.timeout <= 0 or args.start < 0 or args.retry_delay < 0 or args.shutdown_grace < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid workers/token/timeout/retry/start/limit/shutdown values')
    if args.provider == 'anthropic' and (args.stream or args.response_format != 'none' or args.extra_param):
        parser.error('Anthropic provider supports non-streaming requests without response-format or extra-param')
    return args


def prepare_inputs(dataset_name=DATASET):
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(dataset_name)
    expected_count = 1040
    assert len(dataset.data) == expected_count
    rows = json.loads(dataset.data.to_json(orient='records', force_ascii=False))
    prompts, digest = [], hashlib.sha256()
    for i, row in enumerate(rows):
        prompt = dataset.build_prompt(i)
        assert [part['type'] for part in prompt] == ['image', 'image', 'text'], i
        assert prompt[0]['value'] == row['gt_image_path'], i
        assert prompt[1]['value'] == row['candidate_image_path'], i
        assert prompt[2]['value'] == row['question'], i
        digest.update(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode())
        for part in prompt[:2]:
            path = Path(part['value'])
            assert path.is_file(), path
            digest.update(path.read_bytes())
        prompts.append(prompt)
    print(f'Validated {len(rows)} rows; order=GT image / candidate image / {dataset_name} text', flush=True)
    return rows, prompts, digest.hexdigest()


def run_pending(args, pending, completed, rows, output, checkpoint, logger, key, request_base, process_fn=None, selected_count=None):
    if getattr(args, 'provider', 'openai') == 'anthropic':
        from anthropic import Anthropic as client_cls
    else:
        from openai import OpenAI as client_cls
    from tqdm import tqdm
    process_fn = process_fn or (process_record_anthropic if getattr(args, 'provider', 'openai') == 'anthropic' else process_record)
    stop = Event()
    interrupted_at = [None]
    def interrupt(signum, frame):
        if interrupted_at[0] is None:
            interrupted_at[0] = time.monotonic()
        else:
            interrupted_at[0] = time.monotonic() - args.shutdown_grace
        stop.set()
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='api-worker')
    iterator = iter(pending)
    inflight = {}
    failures, written = 0, 0
    totals = empty_token_totals()
    started = time.monotonic()
    total = selected_count if selected_count is not None else len(completed) + len(pending)
    skipped = total - len(pending)
    progress = tqdm(total=total, initial=skipped, desc=args.model,
                    unit='sample', dynamic_ncols=True, mininterval=0.5, file=sys.stderr)
    logger._progress_active = True
    progress.set_postfix(ok=skipped, failed=0, active=0, refresh=False)
    def submit_next():
        if stop.is_set():
            return
        item = next(iterator, None)
        if item is not None:
            row, prompt = item
            future = pool.submit(process_fn, client_cls, args, key, request_base, row, prompt, logger, stop)
            inflight[future] = row
    try:
        with checkpoint.open('a', encoding='utf-8') as stream:
            for _ in range(args.workers):
                submit_next()
            stop_logged = False
            while inflight:
                if stop.is_set() and not stop_logged:
                    logger.emit('run_stop_requested', in_flight=len(inflight), grace_seconds=args.shutdown_grace)
                    stop_logged = True
                progress.set_postfix(ok=skipped + written - failures, failed=failures,
                                     active=len(inflight), refresh=False)
                done, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                if not done:
                    progress.refresh()
                for future in done:
                    row = inflight.pop(future)
                    try:
                        item = future.result()
                    except Exception as exc:
                        item = {**row, 'api_success': False, 'status': 'error', 'api_error_type': type(exc).__name__}
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    written += 1
                    add_item_token_usage(totals, item)
                    if item['api_success']:
                        completed[str(item['index'])] = item
                    else:
                        failures += 1
                    rate = written / max(time.monotonic()-started, 0.001) * 60
                    logger.emit('sample_written', str(item['index']), status=item['status'],
                                written=written, pending_total=len(pending), completed_count=len(completed),
                                records_per_minute=round(rate, 2), token_usage=item.get('token_usage'), token_totals=totals)
                    submit_next()
                    progress.set_postfix(ok=skipped + written - failures, failed=failures,
                                         active=len(inflight), refresh=False)
                    progress.update(1)
                if interrupted_at[0] is not None and time.monotonic()-interrupted_at[0] >= args.shutdown_grace:
                    break
    finally:
        stop.set()
        for future in inflight:
            future.cancel()
        pool.shutdown(wait=not inflight, cancel_futures=True)
        progress.set_postfix(ok=skipped + written - failures, failed=failures,
                             active=len(inflight), refresh=False)
        progress.close()
        logger._progress_active = False
        # Serialize every completed result before exiting; never resubmit queued jobs on stop.
        export_results(completed, rows, output)
        logger.emit('run_stopped' if interrupted_at[0] is not None else 'run_done',
                    completed_count=len(completed), failed=failures, token_totals=totals,
                    unfinished_in_flight=len(inflight))
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if inflight:
        # Python waits for executor threads at normal exit. Grace has expired;
        # results are already flushed/exported and remaining requests stay pending.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    return 130 if interrupted_at[0] is not None else (1 if failures else 0)


def main():
    args = parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise ValueError(f'Environment file not found: {args.env_file}')
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    rows, prompts, input_hash = prepare_inputs(args.dataset)
    selected_rows = rows[args.start:args.start+args.limit if args.limit else None]
    selected_prompts = prompts[args.start:args.start+args.limit if args.limit else None]
    if not selected_rows:
        raise ValueError('No records selected')
    token_field = 'max_tokens' if args.provider == 'anthropic' else args.token_field
    request_base = {'model': args.model, token_field: args.max_tokens}
    if args.temperature is not None:
        request_base['temperature'] = args.temperature
    if args.top_p is not None:
        request_base['top_p'] = args.top_p
    if args.response_format == 'json_object':
        request_base['response_format'] = {'type': 'json_object'}
    extra = parse_extra_params(args.extra_param)
    if set(extra) & {'messages', 'model', 'stream', 'stream_options', 'max_tokens', 'max_completion_tokens'}:
        raise ValueError('Use dedicated options for messages/model/stream/token limit')
    request_base.update(extra)
    safe_model = ''.join(c if c.isalnum() or c in '-_' else '_' for c in args.model)
    output = (args.work_dir or ROOT / 'outputs' / f'{safe_model}_api_dual_image').resolve()
    checkpoint = output / 'progress.jsonl'
    if args.validate_only:
        completed = load_completed(checkpoint)
        skipped = sum(str(row['index']) in completed for row in selected_rows)
        print(f'Model={args.model}; workers={args.workers}; stream={args.stream}; timeout={args.timeout}; retries={args.retries}')
        print(f'Selected={len(selected_rows)}; saved_success={len(completed)}; skipped={skipped}; pending={len(selected_rows)-skipped}; no API calls')
        return 0
    key = os.getenv(args.api_key_env) or (
        os.getenv(args.api_key_fallback_env) if args.api_key_fallback_env else None
    )
    if not args.export_only and not key:
        names = args.api_key_env + (f' or {args.api_key_fallback_env}' if args.api_key_fallback_env else '')
        raise ValueError(f'Set {names}')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process is using {output}')
        completed = load_completed(checkpoint)
        if args.export_only:
            export_results(completed, rows, output)
            return 0
        manifest = {'model': args.model, 'base_url': args.base_url, 'dataset': args.dataset,
                    'input_sha256': input_hash, 'token_field': token_field, 'max_tokens': args.max_tokens}
        if args.provider != 'openai':
            manifest['provider'] = args.provider
        options = {k:v for k,v in request_base.items() if k not in ('model', token_field)}
        manifest['request_options'] = options
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            # Original checkpoints did not contain request_options; absent means defaults.
            saved.setdefault('request_options', {})
            if saved != manifest:
                raise ValueError('Existing output has different model/inputs/request settings; use another --work-dir')
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        logger = StatusLogger(args.log_file or output/'api_run.log', args.status_file or output/'api_status.jsonl')
        pending = [(row,prompt) for row,prompt in zip(selected_rows,selected_prompts) if str(row['index']) not in completed]
        logger.emit('run_start', model=args.model, base_url=args.base_url, workers=args.workers,
                    timeout=args.timeout, max_retries=args.retries, stream=args.stream,
                    output=str(output), completed_count=len(completed))
        logger.emit('run_resume_plan', selected=len(selected_rows), skipped=len(selected_rows)-len(pending), pending=len(pending))
        return run_pending(args, pending, completed, rows, output, checkpoint, logger, key, request_base, selected_count=len(selected_rows))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f'Error: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
