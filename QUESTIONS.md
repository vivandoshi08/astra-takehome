# The questions

Three questions. `README.md` covers setup, budgets, cost, and how to submit —
read it first. This file is the questions themselves: what you build, what we
provide, what you must deliver.

For each question, the required results are listed under **Deliver**. The
`report.md` in each question directory is a suggested structure for writing
them up — you may use it, restructure it, or replace it with your own document
in whatever format you prefer. What matters is that every required result is
present and evidenced.

Commit the logs your runs produce (`fitness_history.csv`, `budget_log.jsonl`)
— they are part of your submission, and `.gitignore` is set up to keep them.

---

## Q1 — Evolve a slime-volleyball agent with NEAT

**Directory:** `q1_slimevolley/` · **Time:** about 5–6 hours · **Compute:**
laptop CPU, roughly 30 minutes wall-clock for a full-budget run · **API cost:** $0

The repo contains a vendored slime-volleyball environment (`slimevolley.py` —
runs on modern Python; do not spend time on dependencies, we did that) with a
built-in baseline opponent: a 120-parameter network from the original 2015
demo. That is what you are evolving against, using `neat-python`.

**Provided:**

- `slimevolley.py` — the environment. 12-dimensional observation.
- `evolve.py` — the driver: population setup, parallel evaluation across
  workers, step accounting, writing results to `runs/<run-id>/`. As shipped,
  each genome's fitness is its score from a single episode.
- `budget.py` — step accounting. Do not edit it or the log it writes.
- `neat_config.ini` — a starting NEAT configuration. Tuning it is fair game.
- `replay.py` — writes a GIF of a champion playing, so you can watch it.
  (`env.render()` does not work: the vendored environment predates gymnasium
  and its renderer imports a module that no longer exists. This draws frames
  from the game state instead. Check `action_from()` matches your own output
  mapping.)
- `episode.py` — a stub. This is the piece you implement.

**You implement:** `play_episode(net, env)` in `episode.py` — run one episode
against the built-in opponent and return the total reward. That means working
out how a network's outputs become an action, and how an episode runs. From
there, everything else is your call: fitness design, reward shaping if you
want it, evaluation and replay tooling.

**Budget:** at most **100,000,000 environment steps** per run, across all
workers. The scaffold counts every step and hard-stops.

**Deliver:**

1. **Your training pipeline**, with the exact command that reproduces your
   best run.
2. **A fitness curve** — population best and mean per generation
   (`evolve.py` writes `fitness_history.csv`).
3. **A champion evaluation**: mean score with a 95% confidence interval over
   **100 fresh, held-out-seed episodes**. Not the training number.
4. **Champion architecture**: node count, connection count, activation
   functions in use.
5. **A behavioral analysis**: replay episodes and describe what your champion
   actually does, in plain language — including any degenerate or exploit-like
   behavior. If you shaped the reward, report what the shaping did to
   behavior.
6. **"What went wrong, and how you found it"**: at least two things that broke
   or gave a misleading result, how you noticed, and what you did about it.

**Graded targets** — you do not need to beat the built-in opponent. Random
play scores −5.0.

| tier | champion mean over the 100 held-out episodes |
|---|---|
| T1 | working pipeline and an honest fitness curve, within budget |
| T2 | ≥ −3.0 |
| T3 | ≥ −1.0 |
| Bonus | > 0 (beats the built-in opponent) |

---

## Q2 — Backprop-NEAT under a compute budget

**Directory:** `q2_backprop_neat/` · **Time:** about 5 hours · **Compute:**
laptop CPU, seconds to under a minute per run · **API cost:** $0

`neat-python` evolves topology. Gradient descent tunes weights. This question
is about combining them — and about spending a fixed amount of compute well.

Three 2-D classification datasets: **xor**, **spiral**, **spiral3**. The
train / held-out splits are frozen by seed so everyone is measured on the same
data. Do not regenerate them.

**Provided:**

- `main.py` — the driver. As shipped it is plain NEAT: fitness is training
  accuracy, no gradient refinement, no complexity term. Adding those is the
  question. The refinement site is marked with a TODO in `evaluate_genome`.
- `network.py` — `GenomeNetwork`, which runs a genome as a plain-numpy
  network and exposes its parameters as a flat vector: `forward()`,
  `get_params()`, `set_params()`, `write_back()`. `forward` is written so you
  can differentiate through it however you like — autograd, torch, JAX, or by
  hand.
- `datasets.py` — the three problems with frozen splits.
- `budget.py` — gradient-step accounting and `plan()`, which derives the
  generation count from your allocation. Do not edit it.
- `viz.py` — decision-boundary and fitness-curve plotting.

**You implement:** gradient refinement of genome weights inside the NEAT loop,
charging `budget.charge(steps)` for what you spend; a complexity penalty of
your own definition; and the experiments below.

**The inheritance choice:** once you have refined a genome's weights, you
decide what becomes of them. Write them back into the genome so its offspring
inherit them (**Lamarckian** — `net.write_back(genome)`), or discard them and
keep only the fitness the refined network earned, leaving the genome's own
weights untouched (**Baldwinian** — learning changes which genomes get
selected, not what is passed on). Both are implementable with what
`network.py` already gives you, and the scaffold does neither as shipped.

**Budget:** **30,000 gradient steps** per run, total, across the whole
evolutionary run. You choose how to spend it: many genomes refined shallowly,
few refined deeply, or only the elite refined at all. `main.py` derives the
generation count from `--steps` and `--refine-fraction` so every run costs the
same total compute. A run that exceeds the budget does not count. Report
`budget.summary()` with every result.

**Deliver:**

1. **Working refinement**: gradient descent on genome weights inside the NEAT
   loop, and held-out accuracy on all three datasets. Held-out accuracy is
   the number that counts, everywhere.
2. **An allocation study**: at least three allocations × both inheritance
   strategies × at least three seeds, reporting held-out accuracy with
   spread. Say which allocation you settled on and why.
3. **Lamarckian or Baldwinian — and evidence.** State which your
   implementation does, and show the evidence that it actually does that
   rather than that you intended it to. Then propose a *mechanism* for the
   difference you observe between the two, and design an experiment that
   tests your explanation. A well-tested wrong hypothesis scores better than
   an untested right one.
4. **A complexity frontier**: sweep at least four penalty strengths and
   report the accuracy-vs-size trade-off, not a single number. Include the
   no-penalty point — you cannot claim the penalty helped without it. Say
   what your penalty is: the backprop-NEAT write-up describes one in words
   but gives no formula, so state what you implemented and why.
5. **A small-network result**: the best held-out accuracy you can reach with
   **6 or fewer enabled connections**, on each dataset, alongside your
   unconstrained numbers.
6. **Decision-boundary plots** for each dataset, with a sentence on what the
   boundary shape tells you.
7. **"What went wrong, and how you found it"**: at least two things, as in Q1.

---


## Q3 — Race the optimizer loop

**Directory:** `q3_v3/` · **Time:** about 5 hours · **Compute:** laptop
CPU, minutes · **API cost:** a $10 OpenRouter key, which we give you —
about $0.40 a run, so roughly 20 runs

An LLM-as-optimizer loop in the FunSearch / OpenEvolve / OPRO lineage. Each
generation it shows a model the best programs so far with their scores and
asks for better ones. The programs are models: an `encode`/`decode` pair
that squeezes a 16x16 image — one soft-edged ellipse whose position, axes,
orientation, and brightness all vary — through a six-number latent code and
reconstructs it. This time you are in the race too.

**Provided:**

- `evo/loop.py` — the loop. Read it; it is short.
- `evo/budget.py` — token accounting and the hard cap. Do not edit it or its
  logs.
- `evo/llm_client.py` — the real client. `evo/mock_client.py` — the offline one.
- `tasks/ellipses.py` — the task and its evaluator.
- `data.py` — the ellipse generator, everything fixed by seed. `HELD_OUT`
  ships but is never shown to the optimizer.
- `vae.py`, `baselines.py` — the trained incumbent and the other baselines.
  Deterministic, and cheap to retrain in any configuration you like.
- `plots.py` — helpers for drawing what an `encode`/`decode` pair actually does.
- `race.py` — scores your model, the baselines, and any champion on the same
  terms, with the same evaluator.
- `student.py` — a stub. This is the piece you implement.

**You implement:** your own model in `student.py` — an `encode(img)` that
maps 256 pixels to exactly 6 numbers and a `decode(z)` that maps them back,
in whatever style you like: hand-designed estimator, fitted model, something
trained with `vae.py`. Same rules as every loop candidate (the 6 numbers are
the only channel; deterministic; no recognising specific images), except you
may use numpy, `data.render`, and models you train yourself.

**Budget:** **$10 of OpenRouter credit.** That is the real constraint, and
it is yours to allocate: about 20 runs at the default model, fewer and more
expensive ones if you switch. There is also a 500,000-token hard stop per
run, but at roughly 200K a run you will not meet it — the money runs out
first. `python spend.py` prices everything you have run so far.

Run it with `--client mock` before you spend anything; that is free, and it
is how you find your plumbing bugs on someone else's budget. The loop is
stochastic — be suspicious of any conclusion drawn from a single run.

**Deliver:**

1. **Your model and its race**: `student.py`, the `race.py` table including
   at least one loop champion, and a few sentences on how the model works
   and where its errors come from.
2. **Your runs, and how you spent the money.** `runs/<run-id>/{run_log.jsonl,
   token_log.jsonl, best.py}` for every run, including the ones that went
   nowhere, plus the `spend.py` table. You had $10 and roughly twenty runs:
   say how you decided to divide them, what each group was for, and what the
   spend bought you. A run that went nowhere still counts and is still worth
   reporting. If you changed models, say why and what it cost you in runs.
   Leftover credit is not a virtue and neither is an empty balance — we are
   reading the reasoning, not the total.
3. **Run-to-run variance, and one setting.** The loop is stochastic, and how
   much it varies is itself a measurement nobody has handed you. Run the
   shipped default at least **three** times and report the spread of final
   champion scores, not an average alone. Then change **one** thing — a
   setting, a model, a prompt, the seed program, `--candidates`, reasoning
   effort where the model supports it — and run that at least three times
   too. Report both arms with their spread and say whether the difference
   survives it. Six runs is about $2.50 at the default model, so this fits
   with room to spare; check `spend.py` before and after. "Arm B won" from
   one run each is the answer we are hoping you don't give.
4. **A verdict on the incumbent.** The loop prints three baseline scores when
   it starts, and the variational autoencoder's is far the best of them — it is
   the number your model and the loop are both chasing. **Do you think that
   number is rational?** Decide whether it is a plausible score for a
   six-number model on this data, show us whatever reasoning or measurement
   settles it for you, and say what follows for the rest of your results. An
   argument from the numbers is worth more than an impression.
5. **Your finding** — about one page. `finding.md` suggests a structure; use
   it, restructure it, or write your own document in whatever format you
   prefer. Tell us one thing you noticed that we did not ask you to look
   for, with the evidence, and with whatever you tried in order to falsify
   it. An observation you deliberately tried to break is worth far more than
   three you did not.

**Graded targets** — for `student.py`, scored by `race.py` on `PUBLIC`
(higher is better; the startup print gives you the baselines):

| tier | your model's score |
|---|---|
| T1 | beats the mean baseline |
| T2 | beats PCA |
| T3 | ≥ −0.0140 |
| Bonus | ≥ −0.0105 |
