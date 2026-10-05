"""No timeout anywhere between compile_candidate and evaluate: an infinite loop
in a candidate (at compile time or inside encode) hangs the run. We interrupt
it ourselves with SIGALRM after 2 s to prove it was still running."""
import signal
import time

import _common  # noqa: F401
from evo.loop import compile_candidate
from tasks import ellipses


class Alarm(BaseException):
    pass


def handler(*_):
    raise Alarm


signal.signal(signal.SIGALRM, handler)

CASES = {
    "infinite loop in encode": "def encode(img):\n    while True:\n        pass\n"
                               "def decode(z):\n    return [0.0]*256",
    "infinite loop at module level (compile time)":
        "while True:\n    pass\ndef encode(img):\n    return [0]*6\n"
        "def decode(z):\n    return [0.0]*256",
    "infinite loop wrapped in try/except Exception":
        "def encode(img):\n    while True:\n        try:\n            pass\n"
        "        except Exception:\n            pass\n"
        "def decode(z):\n    return [0.0]*256",
}

for name, code in CASES.items():
    t = time.time()
    signal.alarm(2)
    try:
        f = compile_candidate(code, ellipses.ENTRY_POINT, isolated=True)
        s = ellipses.evaluate(f)
        signal.alarm(0)
        print(f"  {name}: returned {s} after {time.time()-t:.1f}s")
    except Alarm:
        print(f"  {name}: STILL RUNNING after {time.time()-t:.1f}s "
              f"(killed by our alarm; the loop has no timeout)")

# An Exception-subclass timeout raised inside encode is swallowed by
# evaluate's `except Exception` -> -inf, which is what a fix would want.
class Soft(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Soft()))
f = compile_candidate(CASES["infinite loop in encode"], ellipses.ENTRY_POINT,
                      isolated=True)
signal.alarm(1)
t = time.time()
s = ellipses.evaluate(f)
signal.alarm(0)
print(f"  Exception-subclass alarm inside encode -> evaluate returns {s} "
      f"after {time.time()-t:.1f}s (usable as a fix, but defeatable by a "
      f"candidate's own `except Exception`)")
