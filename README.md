# ASTRA take-home

Thanks for your interest in ASTRA. This is a three-question take-home covering
neuroevolution and LLM-driven optimization.

**Read `QUESTIONS.md` for the questions themselves.** This file covers setup,
budget, and how to submit.

## What we're looking for

You are expected to use Claude Code (or a comparable coding agent) throughout.
That is not a loophole — it is the point. We work with these tools daily, and we
want to see how you work with them.

What distinguishes a strong submission is not how much code you produced. It is
whether you *drove* the work: whether you scoped the task, noticed when the
agent (or you) was wrong, verified results you didn't trust, and can explain why
you made the choices you made. Write freely about that process in each
question's report — there is no fixed format. The "what went wrong, and how you
found it" section is where your debugging judgment gets read, and it is read
carefully.

A submission where everything apparently worked on the first try and the report
says "asked Claude, it worked" scores poorly — not because using the agent is
bad, but because it tells us nothing about you.

## Time budget

**About 15 hours total**, across all three questions. If you are well past that,
stop and submit what you have with a note about where the time went. An honest
partial submission beats a padded complete one, and "this question took 3× longer
than budgeted" is useful signal for us.

## Setup

Python 3.10 or newer. No GPU. A laptop is fine.

```sh
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash sanity_check.sh
```

`sanity_check.sh` should print a line per question and exit 0. If it doesn't,
**stop and email us** — we would rather fix our starter code than have you spend
your budget on our bug.

## What this costs you

**Nothing in model spend — we give you the key.** Q1 and Q2 make zero API
calls; they are pure CPU. All model spend is in Q3, and it comes out of a
**$10 OpenRouter key that we send you** rather than your own account.

| | Model API spend | Compute |
|---|---|---|
| Q1 — slime volleyball | $0 | Laptop CPU, ~30 min wall-clock |
| Q2 — backprop-NEAT | $0 | Laptop CPU, a few minutes per run |
| Q3 — the optimizer loop | our $10 key, ~$0.40 a run | Negligible |
| **Total to you** | **$0** | **No GPU** |

$10 is roughly twenty runs at Q3's default model, and Q3 asks you to spend it
deliberately rather than to spend it all. `q3_v3/spend.py` shows what you have
used at any point. The scaffold also hard-stops each run at 500K tokens and
logs every call. If you run the key dry and still need it, tell us — that
happens, and asking will not count against you.

Separately, you'll want Claude Code access for the work itself. A Pro
subscription covers a weekend of this comfortably. If that is a barrier, email
us before you start; we will sort it out, and asking will not count against
you either.

## Submitting

Work in this repo and **commit as you go** — we read the history, and a single
squashed "final" commit loses the most interesting part of your submission. Tag
the commit where you first got a working result for each question:

```sh
git tag q1-first-result
```

When you're done, run `bash package_submission.sh` and send us the tarball it
produces. It collects your repo history, reports, run logs, budget logs, and
plots, and tells you if anything expected is missing.

Each question specifies its own deliverables; see `QUESTIONS.md`.

### Optional: your coding-agent transcript

If you used a coding agent, you are welcome to include its transcript or
session logs alongside your submission. Part of what ASTRA works on is how
people actually collaborate with these tools, and real traces of someone
solving a real problem are worth more to us than anything we could stage.

**This is entirely optional and plays no part in grading.** Including it will
not help your application, leaving it out will not hurt it, and we do not read
it while assessing your work. It is research material, nothing else. Nobody has
ever been disadvantaged by declining, and we would rather you said no than felt
obliged.

If you do include it, **scrub it first**. Transcripts routinely contain API
keys, absolute paths from your machine, and anything else you happened to paste
in. Put it in a top-level `agent_trace/` directory in any format — the
packaging script will pick it up.

## Ground rules

- **Don't edit the budget enforcement** (`q1_slimevolley/budget.py`,
  `q2_backprop_neat/budget.py`, `q3_v3/evo/budget.py`) or the logs they
  write. If a budget genuinely blocks something reasonable, say so in your
  report and we'll take that as a finding about our scaffold.
- Use any library, any model, any reference material.
- Shortlisted candidates do a **20-minute live walkthrough** — you'll explain one
  debugging decision and make a small change on the spot. Knowing your own
  submission is what matters there.

Questions about the setup are always fair game. Email us.
