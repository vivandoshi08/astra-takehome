"""The real model client. Spends your OpenRouter credit — read `q3_v3/README.md` first.

We give you an OpenRouter key with $10 on it. One environment variable:

    export OPENROUTER_API_KEY=sk-or-...

That is the whole setup. The default model is Claude Haiku 4.5, which is what
the question's costs are quoted against; `EVO_MODEL` overrides it with any
OpenRouter model slug (`anthropic/claude-sonnet-5`, `openai/gpt-4o-mini`, ...).
Switching to a pricier model is allowed and costs you runs — that is a budget
decision, and the question asks you to justify it.

`EVO_REASONING` (low|medium|high) asks for reasoning effort where the model
supports it. Not every model does — Haiku 4.5 does not. Try one call before
you commit a run to it.

Token counts come from the provider's own `usage` field, never estimated, so
the budget in `budget.py` reflects what you are actually billed.
"""

from __future__ import annotations

import os

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 2048


class Response:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMClient:
    """One `complete(prompt, n)` method, same shape as MockClient."""

    def __init__(self, model: str | None = None):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key found. Set OPENROUTER_API_KEY with the key we sent "
                "you. To run without spending anything, use --client mock."
            )
        import openai
        self.model = model or os.environ.get("EVO_MODEL") or DEFAULT_MODEL
        self.reasoning = os.environ.get("EVO_REASONING") or None
        self._client = openai.OpenAI(api_key=key, base_url=BASE_URL)
        self.name = f"openrouter:{self.model}"

    def _one(self, prompt: str) -> Response:
        kwargs = {}
        if self.reasoning:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning}}
        r = self._client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return Response(r.choices[0].message.content or "",
                        r.usage.prompt_tokens, r.usage.completion_tokens)

    def complete(self, prompt: str, n: int = 1, **_) -> list[Response]:
        return [self._one(prompt) for _ in range(n)]
