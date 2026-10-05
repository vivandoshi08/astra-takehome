"""Try to break `compile_candidate(isolated=True)` ("the latent code is the only
channel between encode and decode") and the evaluator's integrity with
handcrafted candidates, scored exactly as the loop scores them
(compile_candidate -> tasks.ellipses.evaluate). Each exploit runs in its own
subprocess because several of them corrupt process-wide state.

    python check_isolation.py            # run all
    python check_isolation.py <name>     # run one (subprocess mode)
"""
import os
import subprocess
import sys
import textwrap

ZEROS = "    return [0.0] * 6\n"
SEED_DECODE = """
def decode(z):
    import math
    cx, cy, a, b, th, amp = z
    a = max(a, 0.5); b = max(b, 0.5)
    ct, st = math.cos(th), math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx; dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = (1.0 - rho) * b + 0.5
            img.append(amp * (0.0 if p < 0.0 else (1.0 if p > 1.0 else p)))
    return img
"""

EXPLOITS = {
    # --- side channels between encode and decode -------------------------
    "builtins_module": """
def encode(img):
    import builtins
    builtins._STASH = list(img)
    return [0.0] * 6
def decode(z):
    import builtins
    return builtins._STASH
""",
    "dunder_builtins_dict": """
def encode(img):
    __builtins__['_STASH'] = list(img)   # the exec namespace shares loop.py's builtins dict
    return [0.0] * 6
def decode(z):
    return _STASH                        # bare name resolves via builtins
""",
    "stdlib_module_attr": """
import math
def encode(img):
    math._img = list(img)
    return [0.0] * 6
def decode(z):
    import math
    return math._img
""",
    "sys_modules": """
def encode(img):
    import sys
    sys.modules['_stash'] = list(img)
    return [0.0] * 6
def decode(z):
    import sys
    return sys.modules['_stash']
""",
    "gc_other_namespace": """
LAST = None
def encode(img):
    global LAST
    LAST = list(img)
    return [0.0] * 6
def decode(z):
    import gc
    for o in gc.get_objects():
        if callable(o) and getattr(o, '__name__', '') == 'encode' \\
                and getattr(o, '__globals__', {}).get('LAST') is not None:
            return o.__globals__['LAST']
    return [0.0] * 256
""",
    "frame_peek_evaluator": """
def encode(img):
    return [0.0] * 6
def decode(z):
    import sys
    return sys._getframe(1).f_locals['img']   # evaluate()'s local: the noisy target
""",
    # --- recognising specific images via the repo's own data module ------
    "import_data_lookup_noisy": """
def encode(img):
    import data
    for i, row in enumerate(data.PUBLIC):
        if abs(float(row[0]) - img[0]) < 1e-12 and abs(float(row[255]) - img[255]) < 1e-12:
            return [float(i), 0, 0, 0, 0, 0]
    return [0.0] * 6
def decode(z):
    import data
    return [float(v) for v in data.PUBLIC[int(z[0])]]
""",
    "import_data_true_params": """
def encode(img):
    import data
    for i, row in enumerate(data.PUBLIC):
        if abs(float(row[0]) - img[0]) < 1e-12 and abs(float(row[255]) - img[255]) < 1e-12:
            return [float(v) for v in data.PUBLIC_PARAMS[i]]
    return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
""" + SEED_DECODE,
    # --- corrupting the evaluator / data / budget at compile time --------
    "patch_evaluate": """
import tasks.ellipses as _E
_E.evaluate = lambda fns, images=None: 0.0
def encode(img):
    return [0.0] * 6
def decode(z):
    return [0.0] * 256
""",
    "mutate_public_in_place": """
import data
data.PUBLIC[:] = 0.0
def encode(img):
    return [0.0] * 6
def decode(z):
    return [0.0] * 256
""",
    "raise_systemexit_in_encode": """
def encode(img):
    raise SystemExit(0)
def decode(z):
    return [0.0] * 256
""",
    "keyboardinterrupt_in_decode": """
def encode(img):
    return [0.0] * 6
def decode(z):
    raise KeyboardInterrupt
""",
    # --- determinism and namespace duplication ---------------------------
    "nondeterministic_encode": """
import random
def encode(img):
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.3:
                sx += v * j; sy += v * i; m += v
    r = random.random()
    return [sx / m, sy / m, 2.5 + r, 2.0 + r, 0.0, 0.8]
""" + SEED_DECODE,
    "module_state_differs_per_namespace": """
import random
W = random.random()
def encode(img):
    return [W, 0, 0, 0, 0, 0]
def decode(z):
    return [0.0 if z[0] == W else 1.0] * 256
""",
    # --- unsandboxed execution at compile time ---------------------------
    "compile_time_side_effects": """
import os
_p = os.environ['AUDIT_CANARY_FILE']
with open(_p, 'a') as _f:
    _f.write('exec ran; canary env=' + os.environ.get('AUDIT_CANARY', '?') + '\\n')
import subprocess as _sp, socket as _so, numpy as _np   # all importable
def encode(img):
    return [0.0] * 6
def decode(z):
    return [0.0] * 256
""",
}


def run_one(name):
    import _common  # noqa: F401
    from evo.loop import compile_candidate
    from tasks import ellipses
    import data

    code = textwrap.dedent(EXPLOITS[name]).strip()
    try:
        fns = compile_candidate(code, ellipses.ENTRY_POINT, isolated=True)
    except BaseException as e:   # noqa: BLE001
        print(f"compile_candidate raised {type(e).__name__}: {e!r}")
        return
    if fns is None:
        print("compile_candidate -> None (rejected)")
        return
    extra = ""
    if name == "module_state_differs_per_namespace":
        e, d = fns
        extra = (f"  encode W={e.__globals__['W']:.4f} decode W="
                 f"{d.__globals__['W']:.4f}")
    if name == "nondeterministic_encode":
        scores = [ellipses.evaluate(fns) for _ in range(5)]
        print("5 evaluations of the same program: "
              + ", ".join(f"{s:+.5f}" for s in scores))
        return
    try:
        s = ellipses.evaluate(fns)
    except BaseException as e:   # noqa: BLE001
        print(f"evaluate() did not catch it: {type(e).__name__}({e!r}) "
              f"escapes to the loop")
        return
    print(f"score {s:+.5f}{extra}")
    if name in ("patch_evaluate", "mutate_public_in_place"):
        # Does it persist for the NEXT, honest candidate in the same process?
        from _common import SEED
        honest = compile_candidate(SEED, ellipses.ENTRY_POINT, isolated=True)
        print(f"  -> the honest SEED scored afterwards in the same process: "
              f"{ellipses.evaluate(honest):+.5f} (true score -0.02719); "
              f"data.PUBLIC.sum()={float(data.PUBLIC.sum()):.2f}")
    if name == "compile_time_side_effects":
        p = os.environ["AUDIT_CANARY_FILE"]
        print(f"  canary file lines written during compile_candidate: "
              f"{open(p).read().splitlines()}")


def main():
    from _common import OUT
    canary = OUT / "isolation_canary.txt"
    if canary.exists():
        canary.unlink()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "AUDIT_CANARY": "visible-to-candidate",
           "AUDIT_CANARY_FILE": str(canary)}
    env.pop("OPENROUTER_API_KEY", None)
    w = max(map(len, EXPLOITS))
    for name in EXPLOITS:
        p = subprocess.run([sys.executable, __file__, name], capture_output=True,
                           text=True, env=env)
        out = (p.stdout.strip() or "(no output)").splitlines()
        tag = f"[exit {p.returncode}]" if p.returncode else ""
        print(f"  {name:<{w}}  {out[0]} {tag}")
        for line in out[1:]:
            print(f"  {'':<{w}}  {line}")
        if p.returncode and p.stderr.strip():
            print(f"  {'':<{w}}  stderr: {p.stderr.strip().splitlines()[-1]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
    else:
        main()
