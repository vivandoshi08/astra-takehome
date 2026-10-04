# Q2 — Backprop-NEAT under a compute budget

This file is a suggested structure, not a form. Write into it, reorder it, or
replace it with your own document in whatever format suits you. What matters
is that every result listed under **Deliver** in `QUESTIONS.md` is present and
evidenced, and that we can find it. Aim for about two pages.

Write freely about your process — what you tried, what you checked, what you
refused to take on faith. The "what went wrong" section is where we read your
debugging judgment, so be concrete there. The tables below are one way to
present the results, not the only way.

## 1. What you built

How refinement works in your version: which optimiser, how you got gradients,
what you differentiated. And how your complexity penalty is defined — the
backprop-NEAT write-up describes one in words but gives no formula, so say what
you implemented and why.

## 2. Results

Held-out accuracy is the number that counts. Report your best allocation, for
example:

| dataset | train acc | held-out acc | nodes | conns | steps used |
|---|---|---|---|---|---|
| xor | | | | | |
| spiral | | | | | |
| spiral3 | | | | | |

## 3. Allocation study

You have 30,000 gradient steps. At least three allocations x both inheritance
strategies x at least three seeds. Report held-out accuracy with spread —
a row per allocation works well:

| allocation (steps / refine-fraction / generations) | strategy | held-out (mean ± sd, n seeds) |
|---|---|---|
| | | |
| | | |
| | | |

Which allocation did you settle on, and why?

## 4. Lamarckian or Baldwinian

State which one your implementation does, and show the evidence that it
actually does that rather than that you intended it to.

Then: propose a **mechanism** for the difference you observe between the two,
and design an experiment that tests your explanation. A well-tested wrong
hypothesis scores better than an untested right one.

## 5. Complexity frontier

Sweep at least four penalty strengths. Report the accuracy-vs-size trade-off,
not a single number. Include the no-penalty point — you cannot claim the
penalty helped without it.

| penalty strength | held-out acc | conns |
|---|---|---|
| 0 (none) | | |
| | | |
| | | |
| | | |

What shape is the frontier, and does it differ between the two strategies?

## 6. Small-network track

Best held-out accuracy you can reach with **6 or fewer enabled connections**,
on each dataset, alongside your unconstrained numbers.

| dataset | ≤6 conns | unconstrained | difference |
|---|---|---|---|
| xor | | | |
| spiral | | | |
| spiral3 | | | |

## 7. Decision boundaries

Plots for each dataset. Say what the boundary shape tells you — a boundary
that is the wrong shape but scores well is worth remarking on.

## 8. What went wrong, and how you found it

At least two things that broke or gave a misleading result, how you noticed,
and what you did about it. Include what you verified rather than assumed, and
what you ran to verify it.

## 9. Anything surprising

Optional. Anything you noticed that we did not ask about.
