"""Score an encode/decode candidate in fresh subprocesses (AUDIT.md A5-A7, C1-C3).

The shipped loop exec'd candidate code inside its own process. Two
namespaces there do not make the latent the only channel: encode and decode
could share state through `builtins`, a stdlib module, `sys.modules`, `gc`
or the evaluator's stack frame, read `data.PUBLIC` directly, or patch
`tasks.ellipses.evaluate` for the rest of the run.

Here encode runs in one fresh `python -I` process that sees the images, and
decode runs in a second one that sees only the latents. Neither can import
the repo's modules (`-I` keeps the CWD and the script directory off
sys.path; the CWD is an empty temp dir), and their environment has no API
key. Each process has a timeout. The score is then computed in this process
by the task's own, unmodified `evaluate`, replaying the two processes'
outputs — so an honest candidate scores exactly what the shipped loop gave
it, and `audit/` and `tests/test_loop_fixes.py` check that it does.

What this does not do: jail the filesystem or the network. A candidate can
still read files or open sockets; it just cannot see the key, the repo's
modules on its import path, or the scorer's memory.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile

# Runs in the child as `python -I -c WORKER spec.json out.json`. Passing it
# with -c (not as a file) also keeps the repo path out of sys.argv[0].
WORKER = r'''
import builtins, json, sys

def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as fh:
        spec = json.load(fh)
    role, items, recheck = spec["role"], spec["items"], spec["recheck"]
    sys.argv = ["candidate"]

    def finish(**res):
        with open(out_path, "w") as fh:
            json.dump(res, fh)

    # Same namespace shape as evo.loop.compile_candidate.
    ns = {"__builtins__": builtins.__dict__}
    try:
        exec(compile(spec["code"], "<candidate>", "exec"), ns)
    except BaseException as exc:
        return finish(status="compile_fail", detail=repr(exc)[:300])
    f = ns.get(role)
    if not callable(f):
        return finish(status="compile_fail", detail=f"no callable {role}")

    def call(item):
        r = f(list(item))
        if r is None:
            raise ValueError(f"{role} returned None")
        return [float(a) for a in r]

    outs = []
    for k, item in enumerate(items):
        try:
            outs.append(call(item))
        except BaseException as exc:
            return finish(status="call_fail",
                          detail=f"{role}[{k}]: {exc!r}"[:300])
    # Determinism: the same input must give the same output on a second call.
    for k in range(min(recheck, len(items))):
        try:
            again = call(items[k])
        except BaseException as exc:
            return finish(status="nondeterministic",
                          detail=f"{role}[{k}] raised on 2nd call: {exc!r}"[:300])
        if json.dumps(again) != json.dumps(outs[k]):
            return finish(status="nondeterministic",
                          detail=f"{role}[{k}] differs on 2nd call")
    finish(status="ok", outs=outs)

main()
'''

DEFAULT_TIMEOUT = 180.0     # seconds per process; the shipped loop had none
RECHECK = 8                 # inputs re-run for the determinism check


def _child_env(tmp: str) -> dict:
    # No OPENROUTER_API_KEY, nothing from the parent's environment.
    return {"PATH": os.defpath, "HOME": tmp, "TMPDIR": tmp,
            "PYTHONDONTWRITEBYTECODE": "1"}


def _run_worker(code, role, items, timeout, recheck):
    tmp = tempfile.mkdtemp(prefix="q3sbx_")
    try:
        spec, out = os.path.join(tmp, "spec.json"), os.path.join(tmp, "out.json")
        with open(spec, "w") as fh:
            json.dump({"role": role, "items": items, "code": code,
                       "recheck": recheck}, fh)
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", WORKER, spec, out],
            cwd=tmp, env=_child_env(tmp), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        def kill():
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # the child and anything it spawned
            except ProcessLookupError:
                pass
            proc.wait()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill()
            return {"status": "timeout", "detail": f"{role} > {timeout:.0f}s"}
        except BaseException:          # Ctrl-C: do not leave the child running
            kill()
            raise
        try:
            with open(out) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {"status": "crashed",
                    "detail": f"{role} process exited {proc.returncode} "
                              f"without a result"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score(code: str, task, images=None, timeout: float = DEFAULT_TIMEOUT,
          recheck: int = RECHECK) -> tuple[float, str, str]:
    """Score `code` on `images` (default task.PUBLIC) with the task's own
    evaluate. Returns (score, status, detail).

    status: "scored" (score is finite), "invalid" (evaluate gave -inf),
    "compile_fail", "call_fail", "nondeterministic", "timeout", "crashed".
    A "compile_fail" is what the shipped loop skipped as `fns is None`.
    """
    imgs = task.PUBLIC if images is None else images
    rows = [[float(a) for a in row] for row in imgs]

    enc = _run_worker(code, "encode", rows, timeout, recheck)
    if enc["status"] != "ok":
        return float("-inf"), enc["status"], enc.get("detail", "")
    latents = enc["outs"]
    # evaluate would stop at the first malformed latent before calling decode;
    # do not hand decode anything it would never have seen.
    if not all(len(z) == task.LATENT and all(math.isfinite(a) for a in z)
               for z in latents):
        return float("-inf"), "invalid", "a latent is not 6 finite numbers"

    dec = _run_worker(code, "decode", latents, timeout, recheck)
    if dec["status"] != "ok":
        return float("-inf"), dec["status"], dec.get("detail", "")
    recons = dec["outs"]

    # Replay both processes through the unmodified evaluator.
    it_z, it_q = iter(latents), iter(recons)

    def encode(img):
        return next(it_z)

    def decode(z):
        return next(it_q)

    s = task.evaluate((encode, decode), images=imgs)
    if s == float("-inf"):
        return s, "invalid", "evaluate rejected the reconstruction"
    return s, "scored", ""
