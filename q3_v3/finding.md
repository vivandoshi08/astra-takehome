# Q3 — Finding

## Is the incumbent's number rational?

**No.** The VAE's −0.0051 comes from memorising the 128 images it is scored
on. It is not a six-number model of ellipses.

- **It beats the noise floor.** Every pixel carries N(0, 0.1²) noise, and
  the score is error against the *noisy* image. Redrawing the true ellipse
  exactly scores −0.0098 on PUBLIC. A per-image six-number fit also soaks up
  a little noise, which gets it to −0.0096. A smooth six-number model cannot
  reach −0.0051 on images it has not seen.
- **It copies the noise.** It reproduces 59% of each PUBLIC image's own
  noise. An honest fit reproduces about 6/256 = 2.3%, and the VAE itself
  reproduces 2.4% on HELD_OUT.
- **The same recipe does it to whatever it trains on.** Trained on 128
  *other* images, it scores −0.0049 on those and −0.0345 on PUBLIC. The
  shipped VAE scores −0.0346 on HELD_OUT, worse than PCA.

So everything below is judged against the honest floor and also reported on
HELD_OUT (`vae_verdict.py`).

## The observation

**The loop's ceiling is a missing idea, not missing ability.** Haiku's
encoders *measure* the ellipse once: moments, thresholds, the brightest
pixels. They never *search*, that is, draw a guess with the decoder, compare
it with the image and adjust. Searching is allowed and easy, because encode
and decode come from the same source, so encode can call decode. One
sentence saying so takes the loop from PCA's level to close to the floor.

## The evidence

| group (runs) | encoders that search | champion median (PUBLIC) | exact rank test |
|---|---|---|---|
| A: shipped loop (5) | **0 / 240** | −0.0173 | — |
| C1: reasoning effort "low" (5) | **0 / 240** | −0.0175 | vs A: p = 1.0 |
| B: + "encode can call decode" (7) | 241 / 334 | **−0.0102** | vs A: p = 0.010 |
| C2: GPT-6 Luna, reasoning "low", no hint (5) | 102 / 108 | **−0.0097** | vs C1: p = 0.008, no overlap |
| our model (per-image fit) | — | −0.0096 | |

HELD_OUT gives the same p-values (`compare_groups.py`, `search_census.py`).

**Where the error sits.** For each champion I fitted the best six numbers
*through its own decoder* (`decompose_champions.py`), which splits its gap
to the floor into the part its encoder loses and the part its decoder
loses.

| | encoder loses | decoder loses |
|---|---|---|
| A and C1 champions | 0.0020–0.0144 | 0.0001–0.0040 |
| B champions whose encoders search | 0.0000–0.0002 | 0.0001–0.0020 |
| b05, whose encoders mostly don't | 0.0060 | 0.0011 |

**What it looks like** (`figures/compare_models.png`). Searching models
(ours, the hint and Luna champions) draw crisp ellipses and leave only
noise. The best shipped champion draws blurred ones and is wrong
everywhere: its interior error against the clean image is 28× ours. The
blur is a hedge. With its own encoder, the true sharp decoder scores worse
than its blurred one in 3 of 5 shipped runs. Haiku edits the drawing to hide
the measuring.

## Trying to break it

- ***"Haiku can't write a search."*** With the hint it writes one in 72% of
  answers, and 6 of 7 runs reach −0.0097 to −0.0118.
- ***"It just needs more thinking."*** With reasoning on (C1), still 0 of 240
  encoders search, and the score does not move.
- ***"Every LLM optimiser stalls here."*** Luna searched in its very first
  answer, without the hint. All five Luna runs sit at the floor.
- ***"The hint works some other way."*** Inside B, the one run whose encoders
  rarely searched (b05, 6 of 48) stayed at A's level, −0.0167. The six that
  searched in 18 to 46 of 48 answers reached −0.0097 to −0.0118.
- ***My own instrument.*** The first census missed drawing functions nested
  inside encode, and called a Luna run at −0.0096 "measuring", a score no
  measuring encoder reaches. I fixed it, and Haiku's 0 of 480 stayed.
- ***Overfitting?*** Every B and C2 champion's HELD_OUT score is within
  0.0002 of its PUBLIC score.

## What I think is going on, and what next

The prompt shows the best three programs, and from the seed onward they all
measure. Haiku edits inside that template (new constants, a reshaped rim)
and never steps outside it. Luna steps outside at generation 0.

The test that would separate this from "any extra sentence nudges the
model" is a **placebo arm**: a neutral sentence of the same length in place
of the hint. It costs about $2.50, more than the $1 left. After that, I
would seed the loop with a *searching* program and see whether Haiku keeps
the idea.

---

| | |
|---|---|
| runs performed | 25 loop runs (A 5, B 7, C1 5, C2 5, Sonnet 3 stopped) + 3 two-call probes |
| total tokens | 4,866,836 (3,468,426 in + 1,398,410 out), 991 calls |
| approximate spend | $8.98 of $10 (`spend.py`); provider meter $8.95 |
| models used | Claude Haiku 4.5 (default); Claude Sonnet 5 (stopped: every answer cut off); GPT-6 Luna |
