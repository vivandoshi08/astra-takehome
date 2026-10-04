# Q1 — Slime volleyball with NEAT

This file is a suggested structure, not a form. Write into it, reorder it, or
replace it with your own document in whatever format suits you — a different
layout, a notebook, a PDF. What matters is that every result listed under
**Deliver** in `QUESTIONS.md` is present and evidenced, and that we can find
it. About 1.5 pages is the right length.

Write freely about your process — what you tried, what you checked, what you
refused to take on faith. The "what went wrong" section is where we read your
debugging judgment, so be concrete there.

## Pipeline

What you built, and how to reproduce your best run (exact command).

## Fitness curve

Population best and mean per generation. Embed the plot or point at the file.
Say what the curve does and why it does that — a plateau is a finding, not a
failure.

## Champion evaluation

Mean score ± 95% CI over 100 fresh, held-out-seed episodes. Not the training
number. A table like this is one way to present it:

| | value |
|---|---|
| mean score | |
| 95% CI | |
| episodes | 100 |
| tier reached | |

## Champion architecture

Node count, connection count, activation functions in use.

## Behavioral analysis

Watch your champion play. What does it actually do? Describe the strategy in
plain language. If you shaped the reward, say what the shaping did to behavior.
Degenerate or exploit-like behavior is interesting — report it rather than
hiding it.

## What went wrong, and how you found it

At least two things that broke or gave a misleading result — what happened, how
you noticed, and what you did about it. Include what you verified rather than
assumed, and what you ran to verify it.

## Anything surprising

Optional. Anything you noticed that we didn't ask about.
