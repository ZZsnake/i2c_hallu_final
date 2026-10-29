from .lmdeploy import LMDeployAPI


class OpenAICompatibleClaudeAPI(LMDeployAPI):
    """Claude API exposed through an OpenAI-compatible chat endpoint.

    Some Claude routes reject ``temperature`` instead of ignoring it.  Keep
    the parameter out of both the default generation kwargs and every request
    payload, including calls that try to provide it explicitly.
    """

    def __init__(self, **kwargs):
        # Override LMDeployWrapper's default temperature before it filters
        # None-valued parameters.
        kwargs['temperature'] = None
        super().__init__(**kwargs)

    def generate_inner(self, inputs, dataset=None, **kwargs):
        kwargs.pop('temperature', None)
        return super().generate_inner(inputs, dataset=dataset, **kwargs)
