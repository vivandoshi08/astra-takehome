# Q3 — race the optimizer loop

An LLM-as-optimizer loop in the FunSearch / OpenEvolve / OPRO lineage. Each
generation it shows a model the best programs so far with their scores, asks
for better ones, evaluates them, and keeps the winners.

The programs it searches for are *models*: an `encode`/`decode` pair that
squeezes a 16x16 image through a six-number latent code and reconstructs it.
This time you are in the race too: you write a model of your own
(`student.py`), scored by exactly the same evaluator, and compare it against
the loop's champions and the baselines — including a small variational
autoencoder we trained for you.

## Run it for free first

```sh
python -m evo.loop --client mock --generations 3
```

`--client mock` never touches the network. Use it to understand the loop, read
the logs, and check your plumbing before spending anything.

## Running it for real

We send you an OpenRouter key with **$10** on it. One variable:

```sh
export OPENROUTER_API_KEY=sk-or-...
python -m evo.loop --client llm --generations 12 --run-id r1
```

That is the whole setup. The default model is `anthropic/claude-haiku-4.5`;
`EVO_MODEL` takes any OpenRouter slug if you want a different one, and
`EVO_REASONING=low|medium|high` asks for reasoning effort on models that
support it (Haiku 4.5 does not — try one call before betting a run on it).

**What it costs.** A 12-generation run is roughly 200K tokens, about
**\$0.40** at the default model — so the key is worth about twenty runs.
Switching to a bigger model is allowed and buys you fewer: Sonnet 5 is about
\$0.80 a run, Opus 5 about \$2.00. That is a real trade and the question asks
you to justify whichever way you go.

```sh
python spend.py        # what every run so far has cost, and what is left
```

The loop also hard-stops at 500,000 tokens per run, but at 200K you will not
reach it — the credit is the binding constraint, not the cap. Every call's
real usage is logged to `token_log.jsonl`.

The loop is stochastic; treat any conclusion from a single run with suspicion.
If you run the key dry and still need it, tell us — that is a normal thing to
happen, not a mark against you.

## What it writes

`runs/<run-id>/`

| file | contents |
|---|---|
| `run_log.jsonl` | one line per generation: best score, every scored candidate |
| `token_log.jsonl` | one line per model call, with the provider's own usage numbers |
| `best.py` | the best program found |

## The task

`tasks/ellipses.py` — write `encode(img)` mapping a flat list of 256 pixel
values to a six-number latent, and `decode(z)` mapping it back. Score is
negative mean squared reconstruction error per pixel on the `PUBLIC` images,
so higher is better and 0.0 is perfect. `encode` and `decode` are compiled
into separate namespaces: the six numbers are the only channel between them.

Every image shows one soft-edged ellipse — position, semi-axes, orientation,
and brightness all vary; `data.py` shows exactly how they are drawn, and
everything is fixed by seed. There are two image sets. `PUBLIC` is what the
loop optimises against. `HELD_OUT` is shipped but never shown to the
optimizer.

The loop starts from a seed program, FunSearch-style
(`tasks/ellipses.py::SEED`): a working decoder plus an encoder that finds the
ellipse's position and nothing else, placed in the population before
generation 0 at no token cost.

## Your model

`student.py` is the piece you implement: your own `encode`/`decode` pair, in
whatever style you like — a hand-designed estimator, a fitted model,
something trained with `vae.py`. The rules are the ones every loop candidate
plays under: encode sees only the one image, decode sees only the six
numbers, deterministic, no recognising specific images. Unlike loop
candidates you may use numpy, `data.render`, and anything you train yourself.
The placeholder in the file is just the loop's seed program — replace it.

```sh
python race.py runs/r1/best.py      # your model vs baselines vs champion
```

`race.py` scores every entry with the task evaluator and plots what each one
does to the images. Reference scores, also printed at the start of every
loop run:

```
mean        -0.0454     decode to the mean image, ignore the input
pca (k=6)   -0.0173     best 6-D linear projection
vae (k=6)   -0.0051     a trained variational autoencoder, 6-D latent
```

The VAE is the incumbent. It is provided, correct as an implementation, and
cheap to retrain (`baselines.py` caches its weights in `baseline_cache/`;
delete the file to retrain from the fixed seed, about a minute). `vae.py` is
plain numpy with hand-written backprop — it takes any width, latent size,
beta, or training set you like, so it doubles as an instrument for your own
measurements. `plots.py` will draw what any encode/decode pair — champion,
baseline, or yours — actually does to the images.

**Graded targets** for your model, scored by `race.py` on `PUBLIC`:

| tier | your model's score |
|---|---|
| T1 | beats the mean baseline |
| T2 | beats PCA |
| T3 | ≥ −0.0140 |
| Bonus | ≥ −0.0105 |

## Files

- `evo/loop.py` — the loop. Read it; it is about 150 lines.
- `evo/budget.py` — token accounting and the hard cap. **Do not edit this or
  its logs**; if the cap blocks something reasonable, say so in your report.
- `evo/llm_client.py` — the real client (OpenRouter). `evo/mock_client.py` —
  the offline one. `spend.py` — what your runs have cost.
- `tasks/ellipses.py` — the task, its evaluator, and the prompt the model sees.
- `data.py` — the ellipse generator and the image sets. `baselines.py` — the
  reference models. `vae.py` — the autoencoder. `plots.py` — plotting helpers.
- `student.py` — **yours to implement**. `race.py` — scores it against
  everything else.

## Your deliverables

1. **Your model** in `student.py`, its `race.py` table including at least one
   loop champion, and a few sentences on how it works and where its errors
   come from.
2. **Your run logs** — `runs/<run-id>/` for every run, including the ones
   that went nowhere.
3. **Your finding** — about one page; `finding.md` suggests a structure. Run
   the loop, race your model against it, watch what both do. Tell us one
   thing you noticed that we did not ask you to look for, with the evidence,
   and with whatever you tried in order to falsify it. An observation you
   deliberately tried to break is worth far more than three you did not.
