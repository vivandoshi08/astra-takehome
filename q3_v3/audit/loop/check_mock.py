"""What the MockClient can and cannot exercise."""
from collections import Counter

from _common import SEED

from evo.loop import compile_candidate, extract_code
from evo.mock_client import CANNED, MockClient
from tasks import ellipses


def score(body):
    f = compile_candidate(body, ellipses.ENTRY_POINT, isolated=True)
    return None if f is None else ellipses.evaluate(f)


print("=== A. jitter: body.replace('> 0.3','> 0.4').replace('0.8]','0.7]')")
for i, body in enumerate(CANNED):
    j = body.replace("> 0.3", "> 0.4").replace("0.8]", "0.7]")
    print(f"  CANNED[{i}]: '> 0.3' x{body.count('> 0.3')}, '0.8]' "
          f"x{body.count('0.8]')}, other '0.8' x{body.count('0.8') - body.count('0.8]')}"
          f" | changed={j != body} | score {score(body):+.4f} -> {score(j):+.4f}")

print(f"\n  seed score {score(SEED):+.4f}; CANNED[2] score {score(CANNED[2]):+.4f}"
      f" (CANNED[2] is the seed program re-formatted)")

print("\n=== B. 20,000 mock responses: how many fail extract/compile/evaluate?")
c = MockClient(seed=123)
outcomes = Counter()
texts = Counter()
for r in c.complete("x" * 4000, n=20_000):
    texts[r.text] += 1
for t, k in texts.items():
    code = extract_code(t)
    if code is None:
        outcomes["no code block"] += k
    elif (s := score(code)) is None:
        outcomes["compile fail"] += k
    elif s == float("-inf"):
        outcomes["-inf"] += k
    else:
        outcomes["valid"] += k
print(f"  distinct texts: {len(texts)}; outcomes: {dict(outcomes)}")
outs = sorted({len(t) // 4 for t in texts})
print(f"  mock output_tokens range: {outs[0]}-{outs[-1]} (real answers: "
      f"~550-2048; see check_cost.py)")

print("\n=== C. is the mock prompt-dependent / seed-dependent?")
a = [r.text for r in MockClient().complete("prompt A", n=8)]
b = [r.text for r in MockClient().complete("completely different", n=8)]
print(f"  same default seed, different prompts -> identical answers: {a == b}")
