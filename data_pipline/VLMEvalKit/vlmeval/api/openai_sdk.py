import json

import requests

from ..smp import get_logger
from .base import BaseAPI

logger = get_logger(__name__)


def _parse_stream_chunk(chunk: dict):
    if chunk.get('type') == 'response.output_text.delta':
        return chunk.get('delta') or '', ''
    if chunk.get('type') == 'response.reasoning_summary_text.delta':
        return '', chunk.get('delta') or ''
    choices = chunk.get('choices', [])
    if not choices:
        return '', ''
    delta = choices[0].get('delta', {})
    if not isinstance(delta, dict):
        return '', ''
    return delta.get('content') or '', delta.get('reasoning_content') or ''


def _to_responses_input(messages):
    responses_input = []
    for message in messages:
        content = message.get('content', '')
        if not isinstance(content, list):
            responses_input.append(dict(role=message['role'], content=content))
            continue

        converted = []
        for item in content:
            item_type = item.get('type')
            if item_type == 'text':
                converted.append(dict(type='input_text', text=item.get('text', '')))
            elif item_type == 'image_url':
                image = item.get('image_url', {})
                if isinstance(image, str):
                    converted.append(dict(type='input_image', image_url=image))
                else:
                    image_item = dict(type='input_image', image_url=image.get('url', ''))
                    if image.get('detail') is not None:
                        image_item['detail'] = image['detail']
                    converted.append(image_item)
            else:
                converted.append(item)
        responses_input.append(dict(role=message['role'], content=converted))
    return responses_input


def _extract_responses_answer(response):
    output_text = response.get('output_text')
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts = []
    for output in response.get('output', []):
        if not isinstance(output, dict):
            continue
        for item in output.get('content', []):
            if not isinstance(item, dict) or item.get('type') not in ('output_text', 'text'):
                continue
            text = item.get('text', '')
            if isinstance(text, str) and text:
                parts.append(text)
    if not parts:
        raise KeyError('Responses API result does not contain output text.')
    return ''.join(parts).strip()


def _collect_streaming_content(response, verbose=False):
    answer_parts = []
    reasoning_parts = []
    answer_chunk_count = 0
    reasoning_chunk_count = 0
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            if verbose:
                logger.warning(f'Invalid streaming chunk: {data[:120]}')
            continue
        answer_part, reasoning_part = _parse_stream_chunk(chunk)
        if reasoning_part and not answer_parts:
            reasoning_parts.append(reasoning_part)
            reasoning_chunk_count += 1
        if answer_part:
            answer_parts.append(answer_part)
            answer_chunk_count += 1
    answer = ''.join(answer_parts).strip()
    if verbose:
        logger.info(
            'Streaming finished: '
            f'answer_chunks={answer_chunk_count}, answer_chars={len(answer)}, '
            f'reasoning_chunks={reasoning_chunk_count}, reasoning_chars={len("".join(reasoning_parts))}'
        )
    return answer


class OpenAISDKWrapper(BaseAPI):
    """Base class for OpenAI-compatible API wrappers.

    Provides a standard ``generate_inner`` implementation with optional
    :class:`~vlmeval.api.adapters.ModelAdapter` support for model-specific
    prompt building and payload processing.

    Subclasses **must** implement :meth:`prepare_inputs` to convert the
    internal message list into the HTTP message format expected by their
    specific API endpoint.

    Attributes:
        adapter: An optional :class:`ModelAdapter` instance. Set to
            ``None`` to disable adapter hooks.
        key (str): API authentication key.
        api_base (str): Full URL of the selected OpenAI-compatible endpoint.
        model (str): Model identifier sent in the request payload.
        timeout (int): HTTP request timeout in seconds.
    """

    is_api: bool = True

    def __init__(self,
                 retry=5,
                 wait=5,
                 verbose=True,
                 system_prompt=None,
                 model_adapter=None,
                 api_format='chat_completions',
                 stream=False,
                 **kwargs):
        """
        Args:
            model_adapter: Either a :class:`ModelAdapter` instance, a
                registered adapter name (``str``), or ``None``.
        """
        self.adapter = None
        if api_format not in ('chat_completions', 'responses'):
            raise ValueError(f'Unsupported API format: {api_format}')
        self.api_format = api_format
        self.stream = stream
        if model_adapter is not None:
            if isinstance(model_adapter, str):
                from .adapters import build_adapter
                self.adapter = build_adapter(model_adapter)
            else:
                self.adapter = model_adapter

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Adapter integration
    # ------------------------------------------------------------------

    def set_dump_image(self, dump_image_func):
        if self.adapter is not None:
            self.adapter.dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset) -> bool:
        if self.adapter is not None:
            return self.adapter.use_custom_prompt(dataset, self.system_prompt)
        return False

    def build_prompt(self, line, dataset=None):
        if self.adapter is not None:
            return self.adapter.build_prompt(line, dataset)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # HTTP formatting – subclasses must override
    # ------------------------------------------------------------------

    def prepare_inputs(self, inputs, system_prompt):
        """Convert the internal message list to HTTP message dicts.

        Args:
            inputs: List of ``{'type': ..., 'value': ...}`` dicts, or a
                list of role-keyed dicts for multi-turn conversations.
            system_prompt: System prompt string, or ``None``.

        Returns:
            List of OpenAI-style message dicts with ``role`` and
            ``content`` fields.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement prepare_inputs()'
        )

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    def generate_inner(self, inputs, dataset=None, **kwargs) -> tuple:
        if self.adapter is not None:
            model_args = self.adapter.override_model_args(dataset, kwargs)
            system_prompt = model_args.pop('system_prompt', self.system_prompt)
            inputs = self.adapter.process_inputs(inputs, dataset)
            kwargs.update(model_args)
        else:
            system_prompt = self.system_prompt

        input_msgs = self.prepare_inputs(inputs, system_prompt)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key}',
        }
        if self.api_format == 'responses':
            if 'max_tokens' in kwargs and 'max_output_tokens' not in kwargs:
                kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
            kwargs.pop('n', None)
            payload = dict(model=self.model, input=_to_responses_input(input_msgs), **kwargs)
        else:
            payload = dict(model=self.model, messages=input_msgs, n=1, **kwargs)
        stream = payload.pop('stream', self.stream)
        if stream:
            payload['stream'] = True

        if self.adapter is not None:
            payload = self.adapter.process_payload(payload, dataset=dataset)

        response = requests.post(
            self.api_base,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout * 1.1,
            stream=stream,
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            if stream and ret_code == 0:
                answer = _collect_streaming_content(response, verbose=self.verbose)
            else:
                resp_struct = json.loads(response.text)
                if self.api_format == 'responses':
                    answer = _extract_responses_answer(resp_struct)
                else:
                    answer = resp_struct['choices'][0]['message']['content'].strip()
            if self.adapter is not None:
                answer = self.adapter.postprocess(answer, dataset=dataset)
        except Exception as err:
            logger.error(f'{type(err)}: {err}')
            if self.verbose:
                logger.error(response.text if hasattr(response, 'text') else response)
        return ret_code, answer, response
