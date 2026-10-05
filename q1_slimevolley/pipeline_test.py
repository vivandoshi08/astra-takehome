"""Does the evolutionary pipeline optimise anything? Swap in an easy, dense fitness.

    python pipeline_test.py --objective contacts --budget 5000000 --run-id ptest_contacts

Runs evolve.py unchanged except for the selection fitness, which becomes a
dense behavioural count (mean ball contacts, or steps survived) instead of
game score. If NEAT + our evaluation + selection work, that number should
climb quickly; if it stays flat, the pipeline itself is broken and every
score-based run is suspect. Logs land in runs/<run-id>/ as usual (the
`fitness` column is the test objective; raw score is still logged).
Not a training method: champions from here are not reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import evolve  # noqa: E402

OBJECTIVES = {
    "contacts": lambda ep: ep["contacts"],
    "survival": lambda ep: ep["steps"] / 3000,
    "returns": lambda ep: ep["returns"],
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    i = argv.index("--objective")
    objective = argv[i + 1]
    del argv[i:i + 2]
    # Fitness is computed in the driver process, so patching here is enough.
    evolve.selection_fitness = lambda ep, args: OBJECTIVES[objective](ep)
    sys.argv = ["evolve.py", *argv]
    raise SystemExit(evolve.main())
