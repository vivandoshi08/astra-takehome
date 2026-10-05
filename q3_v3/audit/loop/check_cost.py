"""Offline cost estimate for one default run (12 generations x 4 candidates)
at Haiku 4.5 rates from spend.py. The prompt is built with the real
build_prompt; tokens are approximated from characters (no tokenizer offline).
"""
from _common import SEED

from evo import loop
from evo.llm_client import MAX_TOKENS
from programs import MOMENTS, REFINED
from tasks import ellipses
import spend

GENS, N = 12, 4
IN_RATE, OUT_RATE = spend.RATES["anthropic/claude-haiku-4.5"]

header = ellipses.prompt_header()
grids = ellipses._images_block(ellipses.SHOW_IMAGES)
p0 = loop.build_prompt(ellipses, [(-0.0272, SEED.strip())])
pop = [(-0.00995, REFINED.strip()), (-0.01222, MOMENTS.strip()),
       (-0.0272, SEED.strip())]
p1 = loop.build_prompt(ellipses, pop)

# Bigger programs later in a run: 3 programs of ~1.5x REFINED length.
big = REFINED.strip() + "\n\n" + MOMENTS.strip()[: len(REFINED) // 2]
p_big = loop.build_prompt(ellipses, [(-0.0099, big)] * 3)

print("prompt pieces (chars):")
print(f"  header (instructions)        {len(header) - len(grids):6,}")
print(f"  3 digit grids                {len(grids):6,}")
print(f"  gen-0 prompt (seed only)     {len(p0):6,}")
print(f"  gen>=1 prompt (refined, moments, seed) {len(p1):6,}")
print(f"  late prompt (3 x ~{len(big):,}-char programs) {len(p_big):6,}")
print(f"  programs: seed {len(SEED):,}, moments {len(MOMENTS):,}, refined "
      f"{len(REFINED):,} chars")


def toks(prompt, cpt_text, cpt_grid):
    g = len(grids)
    return (len(prompt) - g) / cpt_text + g / cpt_grid


print("\ninput tokens per call (text at 3.5-4 chars/tok; digit grids at "
      "2-3 chars/tok):")
for name, p in (("gen 0", p0), ("gen>=1", p1), ("late", p_big)):
    lo, hi = toks(p, 4.0, 3.0), toks(p, 3.5, 2.0)
    print(f"  {name:7s} {lo:6.0f} - {hi:6.0f}   (mock client reports "
          f"len//4 = {len(p)//4})")

print("\nper-run estimate, 48 calls (4 in gen 0 with the short prompt, 44 "
      "with a full top-3):")
in_lo = N * toks(p0, 4.0, 3.0) + (GENS - 1) * N * toks(p1, 4.0, 3.0)
in_hi = N * toks(p0, 3.5, 2.0) + (GENS - 1) * N * toks(p_big, 3.5, 2.0)
resp_chars = len(f"Brief note.\n```python\n{REFINED}\n```")
print(f"  a REFINED-size answer is {resp_chars:,} chars ~ "
      f"{resp_chars/4:.0f}-{resp_chars/3.5:.0f} tokens")
for label, out_per in (("terse (600 tok/answer)", 600),
                       ("typical (1,000 tok/answer)", 1000),
                       ("verbose (1,500)", 1500),
                       (f"every answer hits MAX_TOKENS ({MAX_TOKENS})",
                        MAX_TOKENS)):
    out = GENS * N * out_per
    for il, tin in (("in-low", in_lo), ("in-high", in_hi)):
        cost = tin / 1e6 * IN_RATE + out / 1e6 * OUT_RATE
        share = (out / 1e6 * OUT_RATE) / cost
        print(f"  {label:34s} {il:7s}: in {tin/1e3:5.0f}K + out {out/1e3:4.0f}K "
              f"= {(tin+out)/1e3:4.0f}K tok, ${cost:.2f} (output = "
              f"{share:.0%} of $)")

print("\nREADME claim: ~200K tokens, ~$0.40 per 12-gen run. At $1/$5 per Mtok "
      "that pair implies in+out=200K and in+5*out=400K -> in 150K, out 50K "
      "(~1,040 out tok/answer; ~3,100 in tok/call).")
