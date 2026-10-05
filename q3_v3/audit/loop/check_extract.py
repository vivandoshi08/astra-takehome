"""`extract_code` against plausible answer formats. For each format: what the
regex captures, whether the capture compiles to (encode, decode), and the score.
"Dropped" = a correct program in the answer, but the loop scores nothing."""
from _common import SEED
from programs import MOMENTS

from evo.loop import CODE_BLOCK, compile_candidate, extract_code
from tasks import ellipses

P = MOMENTS.strip()
ENC, DEC = P.split("def decode")
DEC = "def decode" + DEC

CASES = {
    "```python (canonical)":           f"```python\n{P}\n```",
    "prose + ```python + prose":       f"Here is my idea.\n\n```python\n{P}\n```\n\nThis uses moments.",
    "```python + trailing spaces":     f"```python   \n{P}\n```",
    "```python\\r\\n (CRLF)":          f"```python\r\n{P}\r\n```",
    "```Python":                       f"```Python\n{P}\n```",
    "```py":                           f"```py\n{P}\n```",
    "```python3":                      f"```python3\n{P}\n```",
    "```Python3":                      f"```Python3\n{P}\n```",
    "``` python (space before tag)":   f"``` python\n{P}\n```",
    "``` (untagged)":                  f"```\n{P}\n```",
    "~~~python (tilde fence)":         f"~~~python\n{P}\n~~~",
    "````python (4-backtick fence)":   f"````python\n{P}\n````",
    "no fence at all":                 P,
    "unclosed fence (truncated)":      f"```python\n{P}",
    "truncated mid-program":           f"```python\n{P[:len(P)//2]}",
    "two blocks: encode | decode":     f"Encoder:\n```python\n{ENC}\n```\nDecoder:\n```python\n{DEC}\n```",
    "usage snippet first, then prog":  f"Call it as:\n```python\nz = encode(img)\n```\nFull program:\n```python\n{P}\n```",
    "draft block, then fixed block":   f"```python\n{SEED.strip()}\n```\nActually, better:\n```python\n{P}\n```",
    "```text analysis block first":    f"```text\nmoments -> axes\n```\n```python\n{P}\n```",
    "inline ``` in prose before code": f"I'll answer in one ```python block as asked.\n```python\n{P}\n```",
    "``` inside a string in the code": f"```python\nNOTE = 'fence ``` here'\n{P}\n```",
    "```python, no newline after tag": f"```python{P}\n```",
    "fence inside a list item (indented)": "1. Program:\n   ```python\n" + "\n".join("   " + l for l in P.splitlines()) + "\n   ```",
}


def main():
    dropped = 0
    rows = []
    for name, text in CASES.items():
        code = extract_code(text)
        if code is None:
            rows.append((name, "no match", "-", "DROPPED"))
            dropped += 1
            continue
        fns = compile_candidate(code, ellipses.ENTRY_POINT, isolated=True)
        if fns is None:
            first = code.splitlines()[0][:28]
            rows.append((name, f"starts {first!r}", "no", "DROPPED"))
            dropped += 1
            continue
        s = ellipses.evaluate(fns)
        first = code.splitlines()[0][:28]
        status = "ok" if abs(s - (-0.01222)) < 1e-4 else f"WRONG PROGRAM ({s:+.4f})"
        if status != "ok":
            dropped += 0  # scored, but not the program the model meant
        rows.append((name, f"starts {first!r}", "yes", status))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"  {r[0]:<{w}}  {r[1]:<38} compiles={r[2]:<3} -> {r[3]}")
    print(f"\n  regex: {CODE_BLOCK.pattern!r}")
    print(f"  {dropped}/{len(CASES)} formats drop a correct program entirely")


if __name__ == "__main__":
    main()
