"""A fake model client, so you can run the whole loop offline for free.

Use this to understand the loop, debug your task, and check your plumbing
before spending a cent:

    python -m evo.loop --client mock --generations 3

It returns syntactically valid candidates from a small canned pool and reports
plausible token counts, so budget accounting exercises the same code path as
the real client. It does not learn — the scores it produces mean nothing.
"""

from __future__ import annotations

import random

_RENDER = '''
def decode(z):
    import math
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = (1.0 - rho) * b + 0.5
            img.append(amp * (0.0 if p < 0.0 else (1.0 if p > 1.0 else p)))
    return img
'''.strip()

# Candidates for the default ellipses task: encode/decode pairs.
CANNED = [
    # predict a black image, ignore the input
    "def encode(img):\n    return [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]\n\n"
    "def decode(z):\n    return [0.0] * 256",
    # a fixed ellipse in the middle of the frame
    "def encode(img):\n    return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]\n\n" + _RENDER,
    # centroid of bright pixels, fixed shape
    "def encode(img):\n"
    "    sx = sy = m = 0.0\n"
    "    for i in range(16):\n"
    "        for j in range(16):\n"
    "            v = img[i * 16 + j]\n"
    "            if v > 0.3:\n"
    "                sx += v * j; sy += v * i; m += v\n"
    "    if m == 0:\n"
    "        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]\n"
    "    return [sx / m, sy / m, 3.0, 2.0, 0.0, 0.8]\n\n" + _RENDER,
    # centroid plus size from the bright mass, fixed roundness
    "def encode(img):\n"
    "    sx = sy = m = 0.0\n"
    "    for i in range(16):\n"
    "        for j in range(16):\n"
    "            v = img[i * 16 + j]\n"
    "            if v > 0.3:\n"
    "                sx += v * j; sy += v * i; m += v\n"
    "    if m == 0:\n"
    "        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]\n"
    "    r = (m / (0.8 * 3.141592653589793)) ** 0.5\n"
    "    return [sx / m, sy / m, r, r, 0.0, 0.8]\n\n" + _RENDER,
]


class MockResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockClient:
    """Same interface as LLMClient; never touches the network."""

    name = "mock"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def complete(self, prompt: str, n: int = 1, **_) -> list[MockResponse]:
        out = []
        for _ in range(n):
            body = self.rng.choice(CANNED)
            if self.rng.random() < 0.3:                 # a little jitter
                body = body.replace("> 0.3", "> 0.4").replace("0.8]", "0.7]")
            text = f"```python\n{body}\n```"
            out.append(MockResponse(
                text,
                input_tokens=len(prompt) // 4,
                output_tokens=len(text) // 4,
            ))
        return out
