# Q3 — Finding

## Is the incumbent's number rational?

**No.** The VAE's −0.0051 is memorisation of the 128 images it is scored
on, not a six-number model of ellipses.

- **It beats the noise floor.** Every pixel carries N(0, 0.1²) noise, and
  the score is error against the *noisy* image. Redrawing the true ellipse
  exactly scores −0.0098 on PUBLIC. A per-image fit of six numbers absorbs
  only the noise along the few directions it can move in: expected
  −σ²(1 − 6/256) = −0.0098, and −0.0096 on PUBLIC. A smooth six-number
  model cannot get to −0.0051 on images it has not seen.
- **It copies the noise.** Its reconstructions reproduce 59% of each PUBLIC
  image's own noise. An honest fit reproduces about 6/256 = 2.3%, and so
  does the VAE on HELD_OUT (2.4%).
- **The same recipe does it to whatever it trains on.** Retrained
  identically on 128 *other* images: −0.0049 on those images, −0.0345 on
  PUBLIC, −0.0344 on HELD_OUT. On 2,048 images the gap closes (−0.0113 on
  its training set, −0.0144 on PUBLIC). The shipped VAE scores −0.0346 on
  HELD_OUT, worse than PCA.

So the race is judged against the honest floor (−0.0096 on PUBLIC), never
against −0.0051, and every score below is reported on HELD_OUT too.
(`vae_verdict.py`, `audit_baselines.py`)

## The observation

**The loop's ceiling is a missing idea, not missing ability.** Haiku never
thinks of fitting: its encoders *measure* the ellipse once (moments,
thresholds, the brightest pixels) and never *search* (draw a guess with
the decoder, compare, adjust). Yet that is allowed and easy, because encode
and decode are compiled from the same source, so encode can call decode.
One sentence saying so moves the loop from PCA's level to the floor.

## The evidence

| group (5 runs each unless noted) | encoders that search | champion median (PUBLIC) |
|---|---|---|
| A — shipped loop | **0 / 240** | −0.0173 |
| C1 — reasoning effort "low" | **0 / 240** | −0.0175 (vs A: exact p = 1.0) |
| B — one-sentence hint: "encode can call decode" | [[PENDING:B_SEARCH]] | [[PENDING:B_MEDIAN]] (vs A: [[PENDING:B_P]]) |
| C2 — GPT-6 Luna, reasoning "low", no hint | [[PENDING:L_SEARCH]] | [[PENDING:L_MEDIAN]] |
| our model (per-image fit) | — | −0.0096 |

- **Where the error sits** (`decompose_champions.py`). For each champion I
  fit the best possible six numbers *through its own decoder*. In group A,
  0.0034–0.0133 of each champion's error is lost by its encoder and only
  0.0003–0.0027 by its decoder: their decoders could reach −0.0099 to
  −0.0123 with a searching encoder. [[PENDING:DECOMP_B]]
- **A ten-line search bolted onto the seed** (no other change) scores
  −0.0156, better than A's median champion.

## Trying to break it

- *"Haiku can't write a search."* With the hint, [[PENDING:B_SEARCH_PCT]] of
  its encoders search, and four of its first five runs reach −0.0098 to
  −0.0114. It can; it doesn't think to.
- *"It just needs more thinking."* Reasoning effort "low" (C1): still 0/240
  searching encoders, no change in score.
- *"Any LLM optimiser stalls here."* GPT-6 Luna, with no hint, searched in
  its very first answer and its runs sit at the floor (C2).
- *"The hint helps for some other reason."* Inside group B, the one run
  whose encoders rarely searched (b05: 6/48) stayed at A's level (−0.0167);
  the runs that searched reached the floor. [[PENDING:EXT]]
- *My own instrument.* The census first looked only at module-level
  drawing functions and called a Luna run at −0.0096 "measuring"; its
  renderer was nested inside encode. Fixed; Haiku's 0/480 did not change.
- *Overfitting?* Every champion's HELD_OUT score is within 0.0025 of its
  PUBLIC score, and for B within 0.0002.

## What I think is going on

The prompt shows the best three programs, and from the seed on they all
measure; Haiku edits within that template — new constants, a reshaped
rim — and never steps outside it. Luna does step outside it at generation 0.
What would separate this from "the hint's wording simply nudges any model"
is a **placebo control**: a neutral sentence of the same length in place of
the hint.

## What I would do next

Run that placebo arm (5 runs, ~$2), and seed the loop with a *searching*
program to see whether Haiku keeps the idea once it is in the population.

---

| | |
|---|---|
| runs performed | [[PENDING:RUNS]] |
| total tokens | [[PENDING:TOKENS]] |
| approximate spend | [[PENDING:SPEND]] of $10 |
| models used | Claude Haiku 4.5 (default), Claude Sonnet 5 (stopped), GPT-6 Luna |
