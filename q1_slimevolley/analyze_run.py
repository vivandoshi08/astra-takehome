"""Verify a training run's logs and re-derive its headline numbers.

    python analyze_run.py runs/best
    python analyze_run.py runs/best --out runs/best/analysis

Reads only the files described in LOGS.md (run_args.json, budget_log.jsonl,
fitness_history.csv, genomes.jsonl, validation.jsonl, champion.json). Every
check prints its name, the formula used, the numbers plugged in, the result
and a verdict:

    PASS  the logged number is consistent / the property holds
    WARN  suspicious or uninformative; read the detail before trusting it
    FAIL  the logs contradict themselves or a hard rule (budget, seeds) is broken
    INFO  descriptive only, no threshold
    SKIP  a file or column the check needs is missing

Also reads stdout.log (NEAT stagnation / extinction lines) when present. Checks
on the newer episode-behaviour fields (contacts, returns, act_both, out_sat, and
their history columns) SKIP on older runs that did not log them.

Writes <out>/analysis.md (summary table first, then the full detail) and
five plots: fitness_curve.png, seed_noise.png, behavior.png, complexity.png,
contacts.png.
No scipy: the Student-t distribution is computed here from the incomplete beta
function. 95% intervals use evaluate.t_crit_95 (so numbers agree with
evaluate.py) when df >= 9, where it is within 0.005 of exact; below that the
exact value from the incomplete beta is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

try:
    from evaluate import t_crit_95 as _t_crit_95_eval
except Exception:  # evaluate.py unavailable or broken: fall back to exact values
    _t_crit_95_eval = None

EPISODE_STEP_LIMIT = 3000
SCORE_MIN, SCORE_MAX = -5, 5
TOL_EXACT = 1e-9      # recomputed from raw lists in the same file
TOL_CSV = 1e-6        # values that went through CSV formatting
ACT_KEYS = ("act_fwd", "act_back", "act_jump", "act_idle")
RANDOM_CONTACTS, RANDOM_RETURNS = 1.4, 0.1      # random play, per episode (LOGS.md)
BUILTIN_CONTACTS, BUILTIN_RETURNS = 47.0, 25.5  # built-in opponent policy playing our side


# --------------------------------------------------------------------------
# statistics helpers (no scipy)
# --------------------------------------------------------------------------

def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_sf2(t, df):
    """Two-sided p-value P(|T| > |t|) for Student t with df degrees of freedom."""
    if not math.isfinite(t) or df <= 0:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, conf=0.95):
    """Two-sided critical value: P(|T| > c) = 1 - conf. Bisection on t_sf2."""
    alpha = 1 - conf
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_ci(xs, conf=0.95):
    """(mean, sd, se, t_crit, lo, hi) for a t-interval on the mean."""
    n = len(xs)
    m, s = mean(xs), sd(xs)
    if n < 2:
        return m, s, float("nan"), float("nan"), float("nan"), float("nan")
    se = s / math.sqrt(n)
    if conf == 0.95 and n - 1 >= 9 and _t_crit_95_eval is not None:
        tc = _t_crit_95_eval(n - 1)
    else:
        tc = t_crit(n - 1, conf)
    return m, s, se, tc, m - tc * se, m + tc * se


def ols(xs, ys):
    """Simple linear regression y = a + b x. Returns (b, se_b, t, df, p)."""
    n = len(xs)
    if n < 3:
        return (float("nan"),) * 5
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (float("nan"),) * 5
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rss = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    df = n - 2
    se_b = math.sqrt(rss / df / sxx)
    t = b / se_b if se_b > 0 else (math.inf if b > 0 else -math.inf if b < 0 else 0.0)
    return b, se_b, t, df, t_sf2(t, df) if math.isfinite(t) else 0.0


def welch(xs, ys):
    """Welch two-sample t test, mean(ys) - mean(xs). Returns (diff, t, df, p)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return (float("nan"),) * 4
    vx, vy = statistics.variance(xs), statistics.variance(ys)
    diff = mean(ys) - mean(xs)
    se2 = vx / nx + vy / ny
    if se2 == 0:
        return diff, float("nan"), float("nan"), float("nan")
    t = diff / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return diff, t, df, t_sf2(t, df)


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def fmt(x, nd=4):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if abs(x) >= 1e5 and x == int(x):
            return f"{int(x):,}"
        return f"{x:+.{nd}f}" if x < 0 else f"{x:.{nd}f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _num(v):
    if v is None:
        return None
    v = v.strip()
    if v == "" or v.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f == int(f) and "." not in v and "e" not in v.lower() else f


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_jsonl(path):
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def read_csv(path):
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]


class Run:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.args = read_json(run_dir / "run_args.json")
        self.budget = read_jsonl(run_dir / "budget_log.jsonl")
        self.hist = read_csv(run_dir / "fitness_history.csv")
        self.genomes = read_jsonl(run_dir / "genomes.jsonl")
        self.val = read_jsonl(run_dir / "validation.jsonl")
        self.champ = read_json(run_dir / "champion.json")
        log = run_dir / "stdout.log"
        self.stdout = log.read_text(errors="replace") if log.exists() else None

    def bonus(self):
        """Survival bonus B from run_args (selection fitness = mean(score + B*steps/3000 + R*returns))."""
        b = find_key(self.args or {}, ("survival_bonus",))
        return float(b) if isinstance(b, (int, float)) else 0.0

    def return_bonus(self):
        """Return bonus R from run_args (selection fitness += R*returns per episode); missing = 0."""
        r = find_key(self.args or {}, ("return_bonus",))
        return float(r) if isinstance(r, (int, float)) else 0.0

    def return_cap(self):
        """Returns per episode that earn the bonus (--return-cap); missing = uncapped
        (runs before the cap existed: their max returns per episode was 7, below 9)."""
        c = find_key(self.args or {}, ("return_cap",))
        return int(c) if isinstance(c, (int, float)) else None

    def worker_cap(self):
        """Per-process step cap the scaffold enforces: budget // workers."""
        budget = find_key(self.args or {}, ("budget",))
        workers = find_key(self.args or {}, ("workers",))
        if isinstance(budget, int) and isinstance(workers, int) and workers > 0:
            return max(1, budget // workers)
        return None

    def genomes_by_gen(self):
        by = defaultdict(list)
        for g in self.genomes or []:
            by[g["gen"]].append(g)
        return dict(sorted(by.items()))

    def val_by_gen(self):
        by = defaultdict(list)
        for v in self.val or []:
            by[v["gen"]].append(v)
        return dict(sorted(by.items()))

    def hist_col(self, col):
        """(generations, values) for rows where `col` is present and numeric."""
        if not self.hist or col not in self.hist[0]:
            return None
        gens, vals = [], []
        for row in self.hist:
            v = row.get(col)
            if isinstance(v, (int, float)):
                gens.append(row["generation"])
                vals.append(float(v))
        return gens, vals


def raw_of(g):
    """True objective: mean game score. Older logs have no raw_fitness column."""
    if "raw_fitness" in g:
        return g["raw_fitness"]
    return mean(g["scores"]) if g.get("scores") else g["fitness"]


def capped(r, cap):
    return r if cap is None else min(r, cap)


def selection_fitness(scores, steps, bonus, returns=None, rbonus=0.0, cap=None):
    """mean_i(score_i + B*steps_i/3000 + R*min(returns_i, cap))."""
    returns = returns if returns is not None else [0] * len(scores)
    return mean([s + bonus * t / EPISODE_STEP_LIMIT + rbonus * capped(r, cap)
                 for s, t, r in zip(scores, steps, returns)])


def find_key(obj, names):
    """First value whose key is in `names`, searching nested dicts depth-first."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names:
                return v
        for v in obj.values():
            found = find_key(v, names)
            if found is not None:
                return found
    return None


def champion_key(champ):
    return find_key(champ, ("key", "champion_key", "genome_key"))


def champion_gen(champ):
    return find_key(champ, ("gen", "generation", "generation_found", "found_gen"))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

class Report:
    """Collects lines for stdout and analysis.md, plus one summary row per check."""

    def __init__(self):
        self.lines: list[str] = []
        self.summary: list[tuple[str, str, str]] = []
        self.section = ""

    def out(self, text=""):
        print(text)
        self.lines.append(text)

    def begin(self, title):
        self.section = title
        self.out("")
        self.out(f"## {title}")
        self.out("")

    def note(self, text):
        self.out(f"> note: {text}")

    def check(self, name, formula, numbers, result, verdict, threshold=""):
        self.out(f"### [{verdict}] {name}")
        self.out(f"- formula:   {formula}")
        if numbers:
            self.out(f"- numbers:   {numbers}")
        self.out(f"- result:    {result}")
        if threshold:
            self.out(f"- threshold: {threshold}")
        self.out("")
        self.summary.append((f"{self.section}: {name}", result, verdict))

    def skip(self, name, reason):
        self.out(f"### [SKIP] {name}")
        self.out(f"- {reason}")
        self.out("")
        self.summary.append((f"{self.section}: {name}", reason, "SKIP"))

    def markdown(self, run_dir):
        counts = defaultdict(int)
        for _, _, v in self.summary:
            counts[v] += 1
        head = [f"# Run analysis: `{run_dir}`", "",
                "Counts: " + ", ".join(f"{k} {counts[k]}" for k in
                                       ("PASS", "WARN", "FAIL", "INFO", "SKIP") if counts[k]),
                "", "| check | result | verdict |", "|---|---|---|"]
        for name, result, verdict in self.summary:
            cell = result.replace("|", "\\|")
            head.append(f"| {name} | {cell} | **{verdict}** |")
        head += ["", "Plots: fitness_curve.png, seed_noise.png, behavior.png, complexity.png, contacts.png", ""]
        body = []
        for line in self.lines:
            body.append(line if line.startswith(("#", ">")) or line == "" else line)
        return "\n".join(head + body) + "\n"


# --------------------------------------------------------------------------
# 1. budget
# --------------------------------------------------------------------------

def budget_total(budget_rows):
    latest = {}
    for r in budget_rows:
        latest[r["pid"]] = max(latest.get(r["pid"], 0), r["cum_steps"])
    return sum(latest.values()), latest


GAP_THRESHOLD = ("PASS if 0; WARN if > 0 and some worker hit its cap (discarded partial generation); "
                 "FAIL if < 0 (logs claim steps the budget never counted) or > 0 with no worker at cap "
                 "(steps spent that no log row explains)")


def _gap_verdict(gap, capped):
    if gap == 0:
        return "PASS"
    if gap > 0 and capped:
        return "WARN"
    return "FAIL"


def check_budget(run: Run, rep: Report):
    rep.begin("1. Budget")
    if not run.budget:
        rep.skip("budget log", "budget_log.jsonl missing or empty")
        return
    total, latest = budget_total(run.budget)
    rep.out(f"budget_log.jsonl: {len(run.budget)} flush records from {len(latest)} processes")

    limit = find_key(run.args or {}, ("budget", "budget_limit", "step_budget", "limit"))
    if isinstance(limit, (int, float)):
        verdict = "PASS" if total <= limit else "FAIL"
        rep.check("total steps within limit",
                  "total = sum over pid of max(cum_steps); require total <= budget",
                  " + ".join(f"{v:,}" for v in latest.values()) + f" = {total:,}; budget = {int(limit):,}",
                  f"{total:,} / {int(limit):,} ({100 * total / limit:.2f}%)", verdict,
                  "FAIL if total > budget")
    else:
        rep.skip("total steps within limit", "no budget value in run_args.json")

    cap = run.worker_cap()
    capped = sorted(pid for pid, c in latest.items() if cap is not None and c >= cap)
    if cap is not None:
        rep.out(f"per-process cap = budget // workers = {cap:,}; processes that hit it: {len(capped)} of {len(latest)}")
    max_gen = None
    cum = run.hist_col("cum_steps")
    gs = run.hist_col("gen_steps")
    if gs:
        max_gen = max(gs[1]) if gs[1] else None
    if cum and cum[1]:
        cum_last = int(cum[1][-1])
        gap = total - cum_last
        tol = max_gen if max_gen is not None else 0
        verdict = _gap_verdict(gap, capped)
        rep.check("budget total vs last cum_steps",
                  "gap = total - cum_steps[last row]; the run stops at a generation boundary, so expect gap == 0, "
                  "unless a worker hit its cap mid-generation (that partial generation is counted but not logged)",
                  f"{total:,} - {cum_last:,} = {gap:,}; max(gen_steps) = {fmt(tol)}; workers at cap: {len(capped)}",
                  f"gap {gap:,} steps ({100 * gap / total:.2f}% of budget spent on a discarded generation)" if gap
                  else "gap 0", verdict, GAP_THRESHOLD)
        used = find_key(run.args or {}, ("steps_used",))
        if isinstance(used, (int, float)):
            rep.check("run_args.steps_used == budget total", "steps_used == sum over pid of max(cum_steps)",
                      f"{int(used):,} vs {total:,}", "equal" if used == total else f"differs by {int(used) - total:,}",
                      "PASS" if used == total else "FAIL", "exact equality")
    else:
        rep.skip("budget total vs last cum_steps", "fitness_history.csv has no cum_steps column")

    if gs and cum and gs[1] and cum[1]:
        s = int(sum(gs[1]))
        cum_last = int(cum[1][-1])
        verdict = "PASS" if s == cum_last else "WARN"
        rep.check("sum(gen_steps) vs last cum_steps",
                  "sum_g gen_steps[g] == cum_steps[last]",
                  f"sum = {s:,}; cum_steps[last] = {cum_last:,}; diff = {s - cum_last:,}",
                  "equal" if s == cum_last else f"differs by {s - cum_last:,}", verdict,
                  "exact equality expected")
        diffs = [int(c) - int(p) for p, c in zip([0] + cum[1][:-1], cum[1])]
        bad = [(g, d, int(x)) for g, d, x in zip(cum[0], diffs, gs[1]) if d != int(x)]
        rep.check("gen_steps == diff(cum_steps) per row",
                  "gen_steps[g] == cum_steps[g] - cum_steps[g-1]",
                  f"{len(bad)} mismatching rows of {len(diffs)}"
                  + (f"; first: gen {bad[0][0]} diff {bad[0][1]:,} vs gen_steps {bad[0][2]:,}" if bad else ""),
                  "consistent" if not bad else f"{len(bad)} rows inconsistent",
                  "PASS" if not bad else "WARN", "every row must match")
    else:
        rep.skip("sum(gen_steps) vs last cum_steps", "gen_steps or cum_steps column missing")

    has_val_steps = run.hist_col("val_best") is not None
    if run.genomes is not None and run.val is None and has_val_steps:
        rep.skip("episode-level steps vs budget total",
                 "validation.jsonl missing but the run validated (val_best column): its steps can't be accounted")
    elif run.genomes is not None:
        g_steps = sum(sum(g.get("steps", [])) for g in run.genomes)
        v_steps = sum(sum(v.get("val_steps", [])) for v in run.val or [])
        logged = g_steps + v_steps
        gap = total - logged
        tol = max_gen if max_gen is not None else 0
        verdict = _gap_verdict(gap, capped)
        rep.check("episode-level steps vs budget total",
                  "logged = sum(genomes.steps[]) + sum(validation.val_steps[]); "
                  "gap = total - logged; expect gap == 0",
                  f"{g_steps:,} + {v_steps:,} = {logged:,}; total = {total:,}; gap = {gap:,}"
                  + ("" if run.val is not None else " (validation.jsonl missing: val steps = 0)"),
                  f"gap {gap:,} steps ({100 * gap / total:.3f}% of total)", verdict,
                  GAP_THRESHOLD)
        if cum and cum[1]:
            last_gen = cum[0][-1]
            in_hist = (sum(sum(g.get("steps", [])) for g in run.genomes if g["gen"] <= last_gen)
                       + sum(sum(v.get("val_steps", [])) for v in run.val or [] if v["gen"] <= last_gen))
            cum_last = int(cum[1][-1])
            rep.check("episode-level steps vs last cum_steps",
                      "sum of steps[] + val_steps[] over gens <= last history gen == cum_steps[last]",
                      f"{in_hist:,} vs {cum_last:,}; diff = {in_hist - cum_last:,}",
                      "equal" if in_hist == cum_last else f"differs by {in_hist - cum_last:,}",
                      "PASS" if in_hist == cum_last else "WARN", "exact equality expected")
    else:
        rep.skip("episode-level steps vs budget total", "genomes.jsonl missing")


# --------------------------------------------------------------------------
# 2. fitness consistency
# --------------------------------------------------------------------------

def check_fitness(run: Run, rep: Report):
    rep.begin("2. Fitness consistency")
    if not run.genomes:
        rep.skip("genome rows", "genomes.jsonl missing or empty")
        return

    n = len(run.genomes)
    B, R = run.bonus(), run.return_bonus()
    has_raw = "raw_fitness" in run.genomes[0]
    rep.out(f"survival bonus B = {B}, return bonus R = {R} (run_args; missing = 0); "
            f"raw_fitness column {'present' if has_raw else 'absent (older log)'}")
    no_returns = 0
    bad_mean, bad_raw, bad_diff, bad_range, bad_len, bad_timeout = [], [], [], [], [], []
    for g in run.genomes:
        sc = g.get("scores", [])
        tag = (g["gen"], g["key"])
        if sc:
            if R and "returns" not in g:
                no_returns += 1
            want = selection_fitness(sc, g.get("steps", [0] * len(sc)), B, g.get("returns"), R, run.return_cap())
            if abs(g["fitness"] - want) > TOL_EXACT:
                bad_mean.append((tag, g["fitness"], want))
            if has_raw and abs(g["raw_fitness"] - mean(sc)) > TOL_EXACT:
                bad_raw.append((tag, g["raw_fitness"], mean(sc)))
        pf, pa = g.get("points_for"), g.get("points_against")
        if pf is not None and pa is not None:
            if any(s != f - a for s, f, a in zip(sc, pf, pa)):
                bad_diff.append(tag)
        if any((s != int(s)) or s < SCORE_MIN or s > SCORE_MAX for s in sc):
            bad_range.append((tag, sc))
        lens = {len(g.get(k, sc)) for k in ("scores", "steps", "points_for", "points_against")}
        if len(lens) > 1:
            bad_len.append(tag)
        if "timeouts" in g and "steps" in g:
            if g["timeouts"] != sum(1 for s in g["steps"] if s >= EPISODE_STEP_LIMIT):
                bad_timeout.append(tag)

    def ex(lst):
        return f"; first: {lst[0]}" if lst else ""

    if has_raw:
        rep.check("raw_fitness == mean(scores)", f"|raw_fitness - mean(scores)| <= {TOL_EXACT}",
                  f"{len(bad_raw)} of {n:,} genome rows violate{ex(bad_raw)}",
                  "consistent" if not bad_raw else f"{len(bad_raw)} rows inconsistent",
                  "PASS" if not bad_raw else "FAIL", "zero violations")
    rep.check("fitness == mean(score + B*steps/3000 + R*min(returns, cap))",
              f"|fitness - mean_i(scores[i] + B*steps[i]/3000 + R*returns[i])| <= {TOL_EXACT}, B = {B}, R = {R}",
              f"{len(bad_mean)} of {n:,} genome rows violate{ex(bad_mean)}"
              + (f"; {no_returns} rows have R > 0 but no returns[] (R term taken as 0)" if no_returns else ""),
              "consistent" if not bad_mean else f"{len(bad_mean)} rows inconsistent",
              "PASS" if not bad_mean else "FAIL", "zero violations")
    rep.check("scores == points_for - points_against",
              "for every episode i: scores[i] == points_for[i] - points_against[i]",
              f"{len(bad_diff)} of {n:,} rows violate{ex(bad_diff)}",
              "consistent" if not bad_diff else f"{len(bad_diff)} rows inconsistent",
              "PASS" if not bad_diff else "FAIL", "zero violations")
    rep.check("scores are integers in [-5, 5]", "s == int(s) and -5 <= s <= 5 for every score",
              f"{len(bad_range)} of {n:,} rows violate{ex(bad_range)}",
              "all in range" if not bad_range else f"{len(bad_range)} rows out of range",
              "PASS" if not bad_range else "FAIL", "zero violations")
    rep.check("per-episode list lengths agree", "len(scores) == len(steps) == len(points_for) == len(points_against)",
              f"{len(bad_len)} rows with mismatched lengths{ex(bad_len)}",
              "consistent" if not bad_len else f"{len(bad_len)} rows inconsistent",
              "PASS" if not bad_len else "FAIL", "zero violations")
    rep.check("timeouts == count(steps >= 3000)", f"timeouts == #{{i : steps[i] >= {EPISODE_STEP_LIMIT}}}",
              f"{len(bad_timeout)} rows violate{ex(bad_timeout)}",
              "consistent" if not bad_timeout else f"{len(bad_timeout)} rows inconsistent",
              "PASS" if not bad_timeout else "FAIL", "zero violations")

    _check_history_recompute(run, rep)
    _check_validation_rows(run, rep)


def _check_history_recompute(run: Run, rep: Report):
    if not run.hist:
        rep.skip("fitness_history recompute", "fitness_history.csv missing")
        return
    by_gen = run.genomes_by_gen()
    stats = {
        "best_fitness": ("max(fitness)", max),
        "mean_fitness": ("mean(fitness)", mean),
        "median_fitness": ("median(fitness)", statistics.median),
        "min_fitness": ("min(fitness)", min),
        "best_raw": ("max(raw_fitness)", max),
        "mean_raw": ("mean(raw_fitness)", mean),
        "mean_ep_steps": ("mean over all episodes of steps[]", None),
        "timeout_frac": ("sum(timeouts) / #episodes", None),
    }
    cols = run.hist[0].keys()
    for col, (formula, fn) in stats.items():
        if col not in cols:
            rep.skip(f"{col} recompute", f"column {col} not in fitness_history.csv")
            continue
        bad, checked, first = 0, 0, None
        for row in run.hist:
            gen, logged = row["generation"], row.get(col)
            if gen not in by_gen or not isinstance(logged, (int, float)):
                continue
            pop = by_gen[gen]
            if col == "mean_ep_steps":
                val = mean([s for g in pop for s in g.get("steps", [])])
            elif col == "timeout_frac":
                eps = sum(len(g.get("steps", [])) for g in pop)
                val = sum(g.get("timeouts", 0) for g in pop) / eps if eps else float("nan")
            elif col.endswith("_raw"):
                val = fn([raw_of(g) for g in pop])
            else:
                val = fn([g["fitness"] for g in pop])
            checked += 1
            if abs(val - logged) > TOL_CSV * max(1.0, abs(val)):
                bad += 1
                first = first or (gen, logged, val)
        rep.check(f"{col} recompute", f"history[{col}] == {formula} over genomes.jsonl rows of that gen (tol {TOL_CSV}, relative)",
                  f"{bad} of {checked} generations mismatch" + (f"; first: gen {first[0]} logged {first[1]} vs recomputed {first[2]}" if first else ""),
                  "consistent" if not bad else f"{bad} gens inconsistent",
                  "PASS" if not bad and checked else ("SKIP" if not checked else "FAIL"), "zero mismatches")

    if "std_fitness" in cols:
        bad_pop = bad_samp = checked = 0
        for row in run.hist:
            gen, logged = row["generation"], row.get("std_fitness")
            if gen not in by_gen or not isinstance(logged, (int, float)):
                continue
            f = [g["fitness"] for g in by_gen[gen]]
            checked += 1
            if abs(statistics.pstdev(f) - logged) > TOL_CSV * max(1.0, abs(logged)):
                bad_pop += 1
            if len(f) > 1 and abs(statistics.stdev(f) - logged) > TOL_CSV * max(1.0, abs(logged)):
                bad_samp += 1
        ok = bad_pop == 0 or bad_samp == 0
        which = "population sd (ddof=0)" if bad_pop == 0 else "sample sd (ddof=1)" if bad_samp == 0 else "neither"
        rep.check("std_fitness recompute", "history[std_fitness] == pstdev(fitness) or stdev(fitness)",
                  f"{checked} gens; mismatches: pstdev {bad_pop}, stdev {bad_samp}",
                  f"matches {which}", "PASS" if ok else "FAIL", "all gens match one convention")

    for col, gcol in (("best_nodes", "nodes"), ("best_conns", "conns")):
        if col not in cols:
            continue
        bad = checked = 0
        for row in run.hist:
            gen, logged = row["generation"], row.get(col)
            if gen not in by_gen or not isinstance(logged, (int, float)):
                continue
            pop = by_gen[gen]
            top = max(g["fitness"] for g in pop)
            candidates = {g.get(gcol) for g in pop if g["fitness"] == top}
            checked += 1
            if logged not in candidates:
                bad += 1
        rep.check(f"{col} matches a best-fitness genome",
                  f"history[{col}] in {{{gcol} of genomes with fitness == max(fitness)}} (ties allowed)",
                  f"{bad} of {checked} gens mismatch", "consistent" if not bad else f"{bad} gens inconsistent",
                  "PASS" if not bad else "WARN", "zero mismatches")


def _check_validation_rows(run: Run, rep: Report):
    if run.val is None:
        rep.skip("validation rows", "validation.jsonl missing")
        return
    bad_mean, bad_range, bad_train = [], [], []
    fit = {(g["gen"], g["key"]): (g["fitness"], raw_of(g)) for g in run.genomes or []}
    n_sel = n_raw = 0
    for v in run.val:
        sc = v.get("val_scores", [])
        if sc and abs(v["val_mean"] - mean(sc)) > TOL_EXACT:
            bad_mean.append((v["gen"], v["key"]))
        if any(s != int(s) or s < SCORE_MIN or s > SCORE_MAX for s in sc):
            bad_range.append((v["gen"], v["key"]))
        pair = fit.get((v["gen"], v["key"]))
        if pair is not None:
            sel_ok = abs(pair[0] - v["train_fitness"]) <= TOL_EXACT
            raw_ok = abs(pair[1] - v["train_fitness"]) <= TOL_EXACT
            n_sel += sel_ok
            n_raw += raw_ok
            if not (sel_ok or raw_ok):
                bad_train.append((v["gen"], v["key"], v["train_fitness"], pair))
    rep.check("val_mean == mean(val_scores)", f"|val_mean - mean(val_scores)| <= {TOL_EXACT}",
              f"{len(bad_mean)} of {len(run.val)} rows violate" + (f"; first {bad_mean[0]}" if bad_mean else ""),
              "consistent" if not bad_mean else f"{len(bad_mean)} rows inconsistent",
              "PASS" if not bad_mean else "FAIL", "zero violations")
    rep.check("val_scores integers in [-5, 5]", "s == int(s) and -5 <= s <= 5",
              f"{len(bad_range)} rows violate", "all in range" if not bad_range else "out of range",
              "PASS" if not bad_range else "FAIL", "zero violations")
    rep.check("validation.train_fitness == genomes fitness or raw_fitness",
              "same (gen, key) row in both files; matches selection fitness or raw_fitness",
              f"{len(bad_train)} of {len(run.val)} rows match neither; matches selection fitness: {n_sel}, raw: {n_raw}"
              + (f"; first {bad_train[0]}" if bad_train else ""),
              "consistent" if not bad_train else f"{len(bad_train)} rows inconsistent",
              "PASS" if not bad_train else "FAIL", "zero violations")

    vb = run.hist_col("val_best")
    by_gen = run.val_by_gen()
    if vb:
        bad = [(g, x, max(r["val_mean"] for r in by_gen[g])) for g, x in zip(*vb)
               if g in by_gen and abs(x - max(r["val_mean"] for r in by_gen[g])) > TOL_CSV]
        rep.check("history val_best == max(val_mean) of that gen", "val_best[g] == max over validation rows of gen g",
                  f"{len(bad)} of {len(vb[0])} gens mismatch" + (f"; first {bad[0]}" if bad else ""),
                  "consistent" if not bad else f"{len(bad)} gens inconsistent",
                  "PASS" if not bad else "FAIL", "zero mismatches")
    cv = run.hist_col("champion_val")
    if cv:
        drops = [(g, b, a) for g, a, b in zip(cv[0][1:], cv[1][1:], cv[1][:-1]) if a < b - TOL_CSV]
        running, bad = float("-inf"), 0
        for g, x in zip(*cv):
            if g in by_gen:
                running = max(running, max(r["val_mean"] for r in by_gen[g]))
            if math.isfinite(running) and abs(x - running) > TOL_CSV:
                bad += 1
        rep.check("champion_val is the running max of val_mean",
                  "champion_val[g] == max over gens <= g of val_mean, hence non-decreasing",
                  f"{len(drops)} decreases; {bad} gens != running max",
                  "consistent" if not drops and not bad else "inconsistent",
                  "PASS" if not drops and not bad else "FAIL", "zero violations")

    if run.champ:
        key, gen = champion_key(run.champ), champion_gen(run.champ)
        best = max(run.val, key=lambda r: r["val_mean"]) if run.val else None
        cval = find_key(run.champ, ("val_mean", "champion_val"))
        if best is not None:
            tied = [r for r in run.val if r["val_mean"] == best["val_mean"]]
            ok = any(r["key"] == key for r in tied) and (cval is None or abs(cval - best["val_mean"]) <= TOL_EXACT)
            rep.check("champion.json is the argmax of val_mean",
                      "champion.key in {key : val_mean == max(val_mean)} and champion.val_mean == max(val_mean)",
                      f"champion key {key} (gen {gen}), val_mean {cval}; validation max {best['val_mean']} "
                      f"by key(s) {sorted({r['key'] for r in tied})}",
                      "consistent" if ok else "champion is not the best-validated genome",
                      "PASS" if ok else "FAIL", "must match")
    else:
        rep.skip("champion.json is the argmax of val_mean", "champion.json missing")


# --------------------------------------------------------------------------
# 3. seed hygiene
# --------------------------------------------------------------------------

def _as_range(v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return int(v[0]), int(v[1])
    if isinstance(v, dict):
        lo = find_key(v, ("lo", "start", "min", "low", "first"))
        hi = find_key(v, ("hi", "stop", "end", "max", "high"))
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return int(lo), int(hi)
    return None


def _role(name):
    n = name.lower()
    if "train" in n:
        return "training"
    if "val" in n:
        return "validation"
    if "held" in n or "test" in n or "eval" in n:
        return "held-out"
    return None


HELDOUT_EPISODES_DEFAULT = 100   # evaluate.py --episodes default


def collect_seed_ranges(obj, found=None, n_heldout=HELDOUT_EPISODES_DEFAULT):
    """Every [lo, hi) range in run_args whose key names a role. hi is exclusive.

    A bare start value (e.g. `heldout_start: 1000000`) becomes
    [start, start + n_heldout): held-out seeds are start + i for i < episodes.
    """
    found = {} if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            r, role = _as_range(v), _role(str(k))
            if r is None and isinstance(v, int) and str(k).endswith("_start"):
                r = (v, v + n_heldout)
            if r and role and role not in found:
                found[role] = (k, r)
            else:
                collect_seed_ranges(v, found, n_heldout)
    return found


def check_seeds(run: Run, rep: Report):
    rep.begin("3. Seed hygiene")
    if run.args is None:
        rep.skip("seed ranges disjoint", "run_args.json missing")
        return
    ranges = collect_seed_ranges(run.args)
    if any(str(k).endswith("_start") for k, _ in ranges.values()):
        rep.note(f"a range given only by its start is taken as [start, start + {HELDOUT_EPISODES_DEFAULT}) "
                 "(evaluate.py's default episode count)")
    for role, (k, (lo, hi)) in ranges.items():
        rep.out(f"- {role:<10} run_args[{k!r}] = [{lo:,}, {hi:,})")
    missing = [r for r in ("training", "validation", "held-out") if r not in ranges]
    if len(ranges) < 2:
        rep.skip("seed ranges disjoint", f"found {len(ranges)} seed range(s) in run_args.json; need >= 2 "
                 "(expected keys naming train/val/held-out with [lo, hi) or {lo, hi})")
        return
    overlaps = []
    roles = list(ranges)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            (alo, ahi), (blo, bhi) = ranges[a][1], ranges[b][1]
            lo, hi = max(alo, blo), min(ahi, bhi)
            if lo < hi:
                overlaps.append(f"{a} & {b} overlap on [{lo:,}, {hi:,})")
    numbers = "; ".join(f"{r}=[{lo:,},{hi:,})" for r, (_, (lo, hi)) in ranges.items())
    if overlaps:
        verdict, result = "FAIL", "; ".join(overlaps)
    elif missing:
        verdict, result = "WARN", f"disjoint, but no range logged for: {', '.join(missing)}"
    else:
        verdict, result = "PASS", "all three ranges disjoint"
    rep.check("seed ranges disjoint", "for every pair: max(lo_a, lo_b) >= min(hi_a, hi_b)  (hi exclusive)",
              numbers, result, verdict, "FAIL on any overlap; WARN if a role is unlogged")


# --------------------------------------------------------------------------
# 4. seed difficulty confound
# --------------------------------------------------------------------------

def elite_deltas(run: Run):
    """Per gen g: list of fitness_g(k) - fitness_{g-1}(k) for keys present in both."""
    by_gen = run.genomes_by_gen()
    gens = list(by_gen)
    out = {}
    for prev, cur in zip(gens, gens[1:]):
        before = {g["key"]: g["fitness"] for g in by_gen[prev]}
        out[cur] = [g["fitness"] - before[g["key"]] for g in by_gen[cur] if g["key"] in before]
    return out


def check_seed_noise(run: Run, rep: Report):
    rep.begin("4. Seed difficulty confound")
    if not run.genomes:
        rep.skip("elite re-evaluation delta", "genomes.jsonl missing")
        return None
    deltas = elite_deltas(run)
    all_d = [d for ds in deltas.values() for d in ds]
    if not all_d:
        rep.skip("elite re-evaluation delta", "no genome key appears in two consecutive generations")
        return None
    per_gen = {g: mean(ds) for g, ds in deltas.items() if ds}
    m, s = mean(all_d), sd(all_d)
    rep.check("elite delta (individual)", "delta_k,g = fitness_g(k) - fitness_{g-1}(k) for keys in both gens; "
              "genome unchanged, so delta is pure seed/episode noise",
              f"n = {len(all_d)} pairs over {len(per_gen)} gens; mean = {fmt(m)}; sd = {fmt(s)}",
              f"seed noise sd per genome = {fmt(s, 3)} fitness units", "INFO")

    if all(d == 0 for d in all_d):
        rep.check("elites re-evaluated on fresh seeds", "any(delta != 0)", "every delta is exactly 0",
                  "elites carry identical fitness: not re-evaluated, or seeds not fresh per gen", "WARN",
                  "WARN if all deltas are 0")
        return deltas

    best = run.hist_col("best_fitness")
    if not best:
        by_gen = run.genomes_by_gen()
        best = (list(by_gen), [max(g["fitness"] for g in p) for p in by_gen.values()])
    bmap = dict(zip(*best))
    common = [g for g in per_gen if g in bmap and (g - 1) in bmap]
    d_best = [bmap[g] - bmap[g - 1] for g in common]
    d_elite = [per_gen[g] for g in common]
    sd_gen, sd_best = sd(d_elite), sd(d_best)
    r = pearson(d_elite, d_best)
    ratio = sd_gen / sd_best if sd_best and sd_best > 0 else float("nan")
    verdict = "WARN" if (math.isfinite(ratio) and ratio >= 0.5) else "PASS"
    rep.check("seed noise vs best_fitness movement",
              "D_g = mean_k delta_k,g (generation's seed-difficulty shift); dB_g = best_g - best_{g-1}; "
              "ratio = sd(D) / sd(dB); r = corr(D, dB)",
              f"{len(common)} gens; sd(D) = {fmt(sd_gen)}; sd(dB) = {fmt(sd_best)}; ratio = {fmt(ratio, 3)}; "
              f"r = {fmt(r, 3)} (r^2 = {fmt(r * r, 3) if math.isfinite(r) else 'nan'})",
              f"{fmt(100 * ratio, 1)}% of best-fitness jitter is matched by seed shifts; "
              f"r^2 = share of dB variance explained by seed difficulty",
              verdict, "WARN if ratio >= 0.5 (generation-to-generation best_fitness mostly tracks which seeds were drawn)")
    return deltas


# --------------------------------------------------------------------------
# 5. winner's curse, 6. validation noise
# --------------------------------------------------------------------------

def check_winners_curse(run: Run, rep: Report):
    rep.begin("5. Winner's curse (train vs validation)")
    if not run.val:
        rep.skip("train - val gap", "validation.jsonl missing or empty")
        return
    raw = {(g["gen"], g["key"]): raw_of(g) for g in run.genomes or []}

    def train_raw(v):
        return raw.get((v["gen"], v["key"]), v["train_fitness"])

    gaps = [train_raw(v) - v["val_mean"] for v in run.val]
    rep.out("train side = raw_fitness from genomes.jsonl (true objective), falling back to validation.train_fitness")
    m, s, se, tc, lo, hi = mean_ci(gaps)
    verdict = "WARN" if lo > 0 else "PASS"
    rep.check("train raw - val_mean over validated genomes",
              "gap_i = train_raw_i - val_mean_i; CI = mean +- t_{0.975, n-1} * sd / sqrt(n)",
              f"n = {len(gaps)}; mean = {fmt(m)}; sd = {fmt(s)}; se = {fmt(se)}; t = {fmt(tc, 3)}",
              f"mean gap {fmt(m, 3)}, 95% CI [{fmt(lo, 3)}, {fmt(hi, 3)}] (positive = training overstates)",
              verdict, "WARN if CI lower bound > 0: training fitness of selected genomes is optimistic; report val/held-out, not train")
    if run.champ:
        key, gen = champion_key(run.champ), champion_gen(run.champ)
        row = next((v for v in run.val if v["key"] == key and (gen is None or v["gen"] == gen)), None)
        if row is None:
            row = next((v for v in run.val if v["key"] == key), None)
        if row:
            rep.check("champion train raw vs val", "gap = train_raw - val_mean for the champion row",
                      f"{fmt(train_raw(row))} - {fmt(row['val_mean'])}",
                      f"gap {fmt(train_raw(row) - row['val_mean'], 3)}", "INFO")
    rep.note("the champion's val_mean is itself a max over every validated genome on the same fixed validation seeds, "
             "so it is optimistic too; only the held-out evaluation (evaluate.py) is unbiased.")


def check_validation_noise(run: Run, rep: Report):
    rep.begin("6. Validation noise")
    if not run.val or not run.champ:
        rep.skip("champion val_mean CI", "validation.jsonl or champion.json missing")
        return
    key, gen = champion_key(run.champ), champion_gen(run.champ)
    rows = [v for v in run.val if v["key"] == key]
    row = next((v for v in rows if gen is None or v["gen"] == gen), rows[0] if rows else None)
    if row is None:
        rep.skip("champion val_mean CI", f"champion key {key} not found in validation.jsonl")
        return
    sc = row["val_scores"]
    m, s, se, tc, lo, hi = mean_ci(sc)
    half = tc * se
    verdict = "PASS" if half <= 0.5 else "WARN"
    rep.check("champion val_mean 95% CI", "SE = sd(val_scores) / sqrt(n); CI = mean +- t_{0.975, n-1} * SE",
              f"n = {len(sc)}; mean = {fmt(m)}; sd = {fmt(s)}; SE = {fmt(s)}/sqrt({len(sc)}) = {fmt(se)}; t = {fmt(tc, 3)}",
              f"{fmt(m, 3)} +- {fmt(half, 3)}  [{fmt(lo, 3)}, {fmt(hi, 3)}]", verdict,
              "WARN if half-width > 0.5 points")
    others = {}
    for v in run.val:
        if v["key"] != key:
            others[v["key"]] = max(others.get(v["key"], float("-inf")), v["val_mean"])
    tied = sorted(k for k, x in others.items() if x >= lo)
    rep.check("candidates statistically tied with champion",
              "#{other keys k : max val_mean_k >= champion CI lower bound}",
              f"lower bound {fmt(lo, 3)}; {len(tied)} of {len(others)} other validated genomes reach it",
              f"{len(tied)} tied candidates", "INFO" if len(tied) < 5 else "WARN",
              "WARN if >= 5: champion choice is largely luck of the validation seeds")


# --------------------------------------------------------------------------
# 7. reward hacking / degenerate behavior
# --------------------------------------------------------------------------

def behavior_series(run: Run):
    """Per gen: population mean act_* and timeout fraction."""
    by_gen = run.genomes_by_gen()
    series = {"gen": list(by_gen), "timeout_frac": []}
    for k in ACT_KEYS:
        series[k] = []
    for pop in by_gen.values():
        eps = sum(len(g.get("steps", [])) for g in pop)
        series["timeout_frac"].append(sum(g.get("timeouts", 0) for g in pop) / eps if eps else float("nan"))
        for k in ACT_KEYS:
            vals = [g[k] for g in pop if k in g]
            series[k].append(mean(vals) if vals else float("nan"))
    return series


def champion_rows(run: Run):
    key = champion_key(run.champ) if run.champ else None
    if key is None:
        return []
    return [g for g in run.genomes if g["key"] == key]


def check_behavior(run: Run, rep: Report):
    rep.begin("7. Reward-hacking / degenerate-behavior signals")
    if not run.genomes:
        rep.skip("behavior", "genomes.jsonl missing")
        return
    ser = behavior_series(run)
    q = max(1, len(ser["gen"]) // 4)
    last_to = mean(ser["timeout_frac"][-q:])
    first_to = mean(ser["timeout_frac"][:q])
    rep.check("timeout fraction over time", "timeout_frac_g = sum(timeouts) / #episodes; compare first vs last quarter of gens",
              f"first {q} gens: {fmt(first_to, 3)}; last {q} gens: {fmt(last_to, 3)}",
              f"{fmt(100 * last_to, 1)}% of late episodes run to {EPISODE_STEP_LIMIT} steps",
              "WARN" if last_to > 0.25 else "PASS",
              "WARN if > 25% late: possible stalling for draws; watch a replay")

    last_gen = ser["gen"][-1]
    eps = [(st, sc) for g in run.genomes if g["gen"] == last_gen for st, sc in zip(g.get("steps", []), g.get("scores", []))]
    to_scores = [sc for st, sc in eps if st >= EPISODE_STEP_LIMIT]
    r = pearson([float(a) for a, _ in eps], [float(b) for _, b in eps])
    rep.check("episode length vs score (last gen)", "r = pearson(steps_i, score_i) over every episode in the last generation",
              f"{len(eps)} episodes; r = {fmt(r, 3)}; timed-out episodes: {len(to_scores)}, "
              f"their mean score {fmt(mean(to_scores), 3) if to_scores else 'n/a'}, "
              f"share at exactly 0: {fmt(sum(1 for s in to_scores if s == 0) / len(to_scores), 3) if to_scores else 'n/a'}",
              f"r = {fmt(r, 3)}", "INFO")

    present = [k for k in ACT_KEYS if k in run.genomes[0]]
    if present:
        start = ", ".join(f"{k}={fmt(ser[k][0], 2)}" for k in present)
        end = ", ".join(f"{k}={fmt(ser[k][-1], 2)}" for k in present)
        rep.check("population action fractions", "mean over genomes of act_* per generation",
                  f"gen {ser['gen'][0]}: {start} | gen {last_gen}: {end}", "see behavior.png", "INFO")

    crow = champion_rows(run)
    if crow and present:
        acts = {k: mean([g[k] for g in crow]) for k in present}
        saturated = all(v > 0.95 or v < 0.05 for v in acts.values())
        dominant = [k for k, v in acts.items() if v > 0.95]
        idle = acts.get("act_idle", 0)
        if idle > 0.95:
            verdict, result = "WARN", "champion is nearly always idle (never moves)"
        elif saturated:
            verdict, result = "WARN", f"champion's action is ~constant (dominant: {', '.join(dominant) or 'none'})"
        elif dominant:
            verdict, result = "WARN", f"champion does {', '.join(dominant)} > 95% of steps"
        else:
            verdict, result = "PASS", "champion's actions vary"
        rep.check("champion action pattern", "mean of champion's act_* over its genomes.jsonl rows; "
                  "degenerate if any pattern > 0.95, or every pattern is > 0.95 or < 0.05 (constant action)",
                  ", ".join(f"{k}={fmt(v, 3)}" for k, v in acts.items()) + f" ({len(crow)} rows)",
                  result, verdict, "WARN if any act_* > 0.95")
    elif not crow:
        rep.skip("champion action pattern", "champion.json missing or champion key not in genomes.jsonl")

    pf_rows = crow if crow else [g for g in run.genomes if g["gen"] == last_gen]
    label = "champion's training episodes" if crow else "last generation"
    pf = sorted((p for g in pf_rows for p in g.get("points_for", [])), reverse=True)
    if pf:
        total = sum(pf)
        k20 = max(1, len(pf) // 5)
        share = sum(pf[:k20]) / total if total else float("nan")
        hist = {v: pf.count(v) for v in sorted(set(pf))}
        rep.check(f"points_for concentration ({label})",
                  "share = sum of points_for in the top 20% of episodes / total points_for",
                  f"{len(pf)} episodes; distribution {hist}; top {k20} episodes hold {sum(pf[:k20])} of {total} points",
                  f"share {fmt(share, 3)} (0.20 = evenly spread)" if total else "no points won",
                  "INFO")


# --------------------------------------------------------------------------
# 8. learning signal
# --------------------------------------------------------------------------

def check_learning(run: Run, rep: Report):
    rep.begin("8. Learning signal")
    vb = run.hist_col("val_best")
    if not vb or len(vb[0]) < 3:
        if run.val:
            by_gen = run.val_by_gen()
            vb = (list(by_gen), [max(r["val_mean"] for r in rows) for rows in by_gen.values()])
            rep.note("val_best column missing; recomputed from validation.jsonl")
        else:
            rep.skip("val_best slope", "no val_best column and no validation.jsonl")
            return
    gens, vals = vb
    b, se, t, df, p = ols(gens, vals)
    if not math.isfinite(t):
        verdict = "WARN"
    elif t > 2:
        verdict = "PASS"
    else:
        verdict = "WARN"
    rep.check("val_best slope", "OLS val_best_g = a + b*g; SE(b) = sqrt(RSS/(n-2) / Sxx); t = b/SE",
              f"n = {len(gens)}; b = {fmt(b, 5)}/gen; SE = {fmt(se, 5)}; t = {fmt(t, 2)}; df = {df}; p = {fmt(p, 4)}",
              f"{fmt(b * 100, 3)} points per 100 generations", verdict,
              "PASS if t > 2 (validation score improving); WARN otherwise")
    q = max(2, len(vals) // 4)
    if len(vals) >= 2 * q:
        first, last = vals[:q], vals[-q:]
        diff, tw, dfw, pw = welch(first, last)
        rep.check("val_best first vs last quartile", "Welch t: diff = mean(last) - mean(first); t = diff / sqrt(v1/n1 + v2/n2)",
                  f"first {q} gens mean {fmt(mean(first), 3)}; last {q} gens mean {fmt(mean(last), 3)}; "
                  f"t = {fmt(tw, 2)}; df = {fmt(dfw, 1)}; p = {fmt(pw, 4)}",
                  f"diff {fmt(diff, 3)}", "PASS" if math.isfinite(tw) and tw > 2 else "WARN",
                  "PASS if t > 2")
    cv = run.hist_col("champion_val")
    if cv and cv[1]:
        rep.check("champion_val start -> end", "champion_val[last] - champion_val[first]",
                  f"{fmt(cv[1][0], 3)} -> {fmt(cv[1][-1], 3)}", f"+{fmt(cv[1][-1] - cv[1][0], 3)}", "INFO")


# --------------------------------------------------------------------------
# 9. population health
# --------------------------------------------------------------------------

def complexity_series(run: Run):
    by_gen = run.genomes_by_gen()
    return {
        "gen": list(by_gen),
        "nodes": [mean([g.get("nodes", float("nan")) for g in p]) for p in by_gen.values()],
        "conns": [mean([g.get("conns", float("nan")) for g in p]) for p in by_gen.values()],
        "frac_min": [sum(1 for g in p if raw_of(g) == SCORE_MIN) / len(p) for p in by_gen.values()],
    }


def check_population(run: Run, rep: Report):
    rep.begin("9. Population health")
    ns = run.hist_col("n_species")
    if ns and ns[1]:
        final = int(ns[1][-1])
        rep.check("n_species", "min / final of history[n_species]",
                  f"min {int(min(ns[1]))}, max {int(max(ns[1]))}, final {final}",
                  f"{final} species at the end", "WARN" if final <= 1 else "PASS",
                  "WARN if the final generation has a single species (diversity collapse)")
    else:
        rep.skip("n_species", "n_species column missing")
    if not run.genomes:
        rep.skip("complexity", "genomes.jsonl missing")
        return
    cs = complexity_series(run)
    n0, n1, c0, c1 = cs["nodes"][0], cs["nodes"][-1], cs["conns"][0], cs["conns"][-1]
    ratio = c1 / c0 if c0 else float("nan")
    rep.check("mean genome size growth", "ratio = mean(conns) last gen / mean(conns) first gen",
              f"nodes {fmt(n0, 1)} -> {fmt(n1, 1)}; conns {fmt(c0, 1)} -> {fmt(c1, 1)}; ratio {fmt(ratio, 2)}",
              f"x{fmt(ratio, 2)} connections", "WARN" if ratio > 3 else "PASS", "WARN if > 3x (bloat)")
    f0, f1 = cs["frac_min"][0], cs["frac_min"][-1]
    rep.check("fraction of population at exactly -5", "#{genomes : raw_fitness == -5} / pop size, first vs last gen",
              f"{fmt(f0, 3)} -> {fmt(f1, 3)}", f"{fmt(100 * f1, 1)}% of the final population scores -5 every episode",
              "WARN" if f1 > 0.5 else "PASS", "WARN if > 50% at the end")


# --------------------------------------------------------------------------
# 10. reward shaping (survival bonus)
# --------------------------------------------------------------------------

def check_shaping(run: Run, rep: Report):
    rep.begin("10. Reward shaping (survival / return bonus)")
    B, R = run.bonus(), run.return_bonus()
    if B == 0 and R == 0:
        rep.skip("bonus share of selection fitness",
                 "survival_bonus and return_bonus are 0 (or absent): selection fitness == raw score")
        return
    if not run.genomes:
        rep.skip("bonus share of selection fitness", "genomes.jsonl missing")
        return
    shares, cov_shares, corrs, gens, surv_sh, ret_sh = [], [], [], [], [], []
    for gen, pop in run.genomes_by_gen().items():
        raw = [raw_of(g) for g in pop]
        bonus = [g["fitness"] - r for g, r in zip(pop, raw)]
        fit = [g["fitness"] for g in pop]
        vf = statistics.pvariance(fit)
        if vf == 0:
            continue
        mr, mb = mean(raw), mean(bonus)
        cov = sum((r - mr) * (b - mb) for r, b in zip(raw, bonus)) / len(pop)
        shares.append(statistics.pvariance(bonus) / vf)
        cov_shares.append(2 * cov / vf)
        corrs.append(pearson(raw, bonus))
        gens.append(gen)
        surv_sh.append(statistics.pvariance([B * mean(g.get("steps", [0])) / EPISODE_STEP_LIMIT for g in pop]) / vf)
        if R and all("returns" in g for g in pop):
            ret_sh.append(statistics.pvariance(
                [R * mean(capped(r, run.return_cap()) for r in g["returns"]) for g in pop]) / vf)
    if not shares:
        rep.skip("bonus share of selection fitness", "selection fitness has zero variance in every generation")
        return
    q = max(1, len(shares) // 4)
    late = mean(shares[-q:])
    rep.check("bonus share of selection-fitness variance",
              "per gen: bonus_k = fitness_k - raw_k (= mean(B*steps/3000 + R*returns), both shaping terms); "
              "Var(fit) = Var(raw) + Var(bonus) + 2Cov(raw, bonus); share = Var(bonus)/Var(fit)",
              f"B = {B}; R = {R}; {len(shares)} gens; mean share {fmt(mean(shares), 3)} (last {q} gens {fmt(late, 3)}); "
              f"mean 2Cov/Var(fit) {fmt(mean(cov_shares), 3)}; mean corr(raw, bonus) "
              f"{fmt(mean([c for c in corrs if math.isfinite(c)]), 3) if any(math.isfinite(c) for c in corrs) else 'nan'}",
              f"{fmt(100 * late, 1)}% of late selection spread is shaping bonus, not score",
              "WARN" if late > 0.5 else "PASS",
              "WARN if > 50% in the last quarter: selection mostly rewards the shaping terms")
    if R:
        rep.check("shaping variance split: survival vs return term",
                  "per gen: Var(B*mean(steps)/3000)/Var(fit) and Var(R*mean(returns))/Var(fit); last-quarter means "
                  "(they need not sum to the bonus share: the two terms covary)",
                  f"B = {B}, R = {R}; survival term {fmt(mean(surv_sh[-q:]), 3)}; "
                  f"return term {fmt(mean(ret_sh[-q:]), 3) if ret_sh else 'n/a (no returns[])'}",
                  "see numbers", "INFO")

    to = run.hist_col("timeout_frac")
    mr = run.hist_col("mean_raw") or run.hist_col("best_raw")
    if not to or not mr:
        ser = behavior_series(run)
        by_gen = run.genomes_by_gen()
        to = (ser["gen"], ser["timeout_frac"])
        mr = (list(by_gen), [mean([raw_of(g) for g in p]) for p in by_gen.values()])
    bt, _, tt, _, _ = ols(*to)
    br, _, tr, _, _ = ols(*mr)
    stalling = math.isfinite(tt) and tt > 2 and not (math.isfinite(tr) and tr > 2)
    rep.check("stalling for survival",
              "OLS slopes over generations: timeout_frac_g and mean_raw_g; "
              "flag if timeout slope t > 2 while raw-score slope t <= 2",
              f"timeout_frac slope {fmt(bt, 5)}/gen (t = {fmt(tt, 2)}); mean_raw slope {fmt(br, 5)}/gen (t = {fmt(tr, 2)})",
              "timeouts rising without raw-score gain: likely reward hacking" if stalling else "no stalling signature",
              "WARN" if stalling else "PASS", "WARN on the signature above")


# --------------------------------------------------------------------------
# 11-14. episode-behaviour fields (contacts, returns, act_both, out_sat)
# --------------------------------------------------------------------------

def has_contacts(run: Run):
    return bool(run.genomes) and "contacts" in run.genomes[0] and "returns" in run.genomes[0]


def contact_series(run: Run):
    """Per gen, recomputed from genomes.jsonl: mean contacts / returns per episode."""
    by_gen = run.genomes_by_gen()
    gens, c, r = [], [], []
    for gen, pop in by_gen.items():
        cs = [x for g in pop for x in g.get("contacts", [])]
        rs = [x for g in pop for x in g.get("returns", [])]
        if cs:
            gens.append(gen)
            c.append(mean(cs))
            r.append(mean(rs) if rs else float("nan"))
    return gens, c, r


def _step_weighted(pop, key):
    """sum_k key_k * steps_k / sum_k steps_k, steps_k = genome k's total episode steps.

    act_* and out_sat are already step-weighted fractions within each genome
    (episode counts / that genome's steps), so this reproduces the population
    fraction exactly (out_sat's factor 3 cancels)."""
    tot = sum(sum(g.get("steps", [])) for g in pop)
    if not tot:
        return float("nan")
    return sum(g[key] * sum(g.get("steps", [])) for g in pop) / tot


def _hist_or_recompute(run: Run, col, recompute):
    s = run.hist_col(col)
    if s and s[1]:
        return s, "fitness_history.csv"
    return recompute(), "recomputed from genomes.jsonl"


def check_behavior_logs(run: Run, rep: Report):
    rep.begin("11. Behaviour-log consistency (contacts, returns, cancel, saturation)")
    if not run.genomes:
        rep.skip("behaviour fields", "genomes.jsonl missing")
        return
    g0 = run.genomes[0]
    n = len(run.genomes)
    if has_contacts(run):
        bad_len, bad_ret, bad_neg = [], [], []
        for g in run.genomes:
            tag = (g["gen"], g["key"])
            ne = len(g.get("scores", []))
            lists = [g.get(k) for k in ("contacts", "opp_contacts", "returns") if k in g]
            if any(len(x) != ne for x in lists):
                bad_len.append(tag)
            if any(r > c for c, r in zip(g["contacts"], g["returns"])):
                bad_ret.append(tag)
            if any(x < 0 or x != int(x) for lst in lists for x in lst):
                bad_neg.append(tag)
        n_eps = sum(len(g["contacts"]) for g in run.genomes)
        rep.check("returns <= contacts per episode", "for every episode i: returns[i] <= contacts[i] "
                  "(a return is a contact after which the ball crossed)",
                  f"{len(bad_ret)} of {n:,} genome rows violate ({n_eps:,} episodes)"
                  + (f"; first {bad_ret[0]}" if bad_ret else ""),
                  "consistent" if not bad_ret else f"{len(bad_ret)} rows inconsistent",
                  "PASS" if not bad_ret else "FAIL", "zero violations")
        rep.check("contact lists: lengths and integrality",
                  "len(contacts) == len(opp_contacts) == len(returns) == len(scores); all non-negative integers",
                  f"length mismatches {len(bad_len)}; negative/non-integer {len(bad_neg)}",
                  "consistent" if not (bad_len or bad_neg) else "inconsistent",
                  "PASS" if not (bad_len or bad_neg) else "FAIL", "zero violations")
    else:
        rep.skip("returns <= contacts per episode", "genomes.jsonl has no contacts/returns (older log)")

    if "act_both" in g0 and "act_fwd" in g0 and "act_back" in g0:
        bad = [(g["gen"], g["key"]) for g in run.genomes
               if g["act_both"] > min(g["act_fwd"], g["act_back"]) + TOL_EXACT]
        rep.check("act_both <= min(act_fwd, act_back)", "both-pressed steps are a subset of fwd steps and of back steps",
                  f"{len(bad)} of {n:,} rows violate" + (f"; first {bad[0]}" if bad else ""),
                  "consistent" if not bad else f"{len(bad)} rows inconsistent",
                  "PASS" if not bad else "FAIL", "zero violations")
    if "out_sat" in g0:
        bad = [(g["gen"], g["key"]) for g in run.genomes if not 0 <= g["out_sat"] <= 1]
        rep.check("out_sat in [0, 1]", "0 <= out_sat <= 1", f"{len(bad)} of {n:,} rows violate",
                  "consistent" if not bad else "out of range", "PASS" if not bad else "FAIL", "zero violations")

    if run.val and has_contacts(run):
        v_rows = [v for v in run.val if "val_contacts" in v]
        if v_rows:
            bad_ret = [(v["gen"], v["key"]) for v in v_rows
                       if any(r > c for c, r in zip(v["val_contacts"], v.get("val_returns", [])))]
            bad_len = [(v["gen"], v["key"]) for v in v_rows
                       if any(len(v.get(k, v["val_scores"])) != len(v["val_scores"])
                              for k in ("val_contacts", "val_opp_contacts", "val_returns"))]
            rep.check("validation: val_returns <= val_contacts, lengths agree",
                      "per validation episode: val_returns[i] <= val_contacts[i]; list lengths == len(val_scores)",
                      f"{len(bad_ret)} rows with returns > contacts; {len(bad_len)} rows with length mismatch "
                      f"(of {len(v_rows)})",
                      "consistent" if not (bad_ret or bad_len) else "inconsistent",
                      "PASS" if not (bad_ret or bad_len) else "FAIL", "zero violations")

    if not run.hist:
        rep.skip("behaviour history recompute", "fitness_history.csv missing")
        return
    cols = run.hist[0].keys()
    by_gen = run.genomes_by_gen()

    def best_raw_contacts(pop):
        top = max(raw_of(g) for g in pop)
        return {mean(g["contacts"]) for g in pop if raw_of(g) == top and g.get("contacts")}

    recompute = {
        "mean_contacts": ("mean over every episode of contacts[]", ("contacts",),
                          lambda p: mean([x for g in p for x in g["contacts"]])),
        "mean_returns": ("mean over every episode of returns[]", ("returns",),
                         lambda p: mean([x for g in p for x in g["returns"]])),
        "mean_cancel": ("sum_k act_both_k * S_k / sum_k S_k, S_k = sum(steps_k)", ("act_both",),
                        lambda p: _step_weighted(p, "act_both")),
        "mean_sat": ("sum_k out_sat_k * S_k / sum_k S_k, S_k = sum(steps_k)", ("out_sat",),
                     lambda p: _step_weighted(p, "out_sat")),
        "mean_nodes": ("mean over genomes of nodes", ("nodes",), lambda p: mean([g["nodes"] for g in p])),
        "mean_conns": ("mean over genomes of conns", ("conns",), lambda p: mean([g["conns"] for g in p])),
        "best_raw_contacts": ("mean(contacts) of a genome with max raw_fitness (ties allowed)", ("contacts",),
                              best_raw_contacts),
    }
    for col, (formula, need, fn) in recompute.items():
        if col not in cols:
            rep.skip(f"{col} recompute", f"column {col} not in fitness_history.csv (older log)")
            continue
        if not all(k in g0 for k in need):
            rep.skip(f"{col} recompute", f"genomes.jsonl lacks {', '.join(need)}")
            continue
        bad = checked = 0
        first = None
        for row in run.hist:
            gen, logged = row["generation"], row.get(col)
            if gen not in by_gen or not isinstance(logged, (int, float)):
                continue
            val = fn(by_gen[gen])
            checked += 1
            vals = val if isinstance(val, set) else {val}
            if not any(abs(v - logged) <= TOL_CSV * max(1.0, abs(v)) for v in vals):
                bad += 1
                first = first or (gen, logged, sorted(vals))
        rep.check(f"{col} recompute", f"history[{col}] == {formula} over genomes.jsonl rows of that gen "
                  f"(tol {TOL_CSV}, relative)",
                  f"{bad} of {checked} generations mismatch"
                  + (f"; first: gen {first[0]} logged {first[1]} vs recomputed {first[2]}" if first else ""),
                  "consistent" if not bad else f"{bad} gens inconsistent",
                  "PASS" if not bad and checked else ("SKIP" if not checked else "FAIL"), "zero mismatches")


def check_contacts(run: Run, rep: Report):
    rep.begin("12. Ball contacts as a learning signal")
    if not has_contacts(run):
        rep.skip("contacts slope", "genomes.jsonl has no contacts/returns (older log)")
        return
    rgens, rc, rr = contact_series(run)
    for col, label, rec, ref_rand, ref_bi in (("mean_contacts", "contacts", rc, RANDOM_CONTACTS, BUILTIN_CONTACTS),
                                              ("mean_returns", "returns", rr, RANDOM_RETURNS, BUILTIN_RETURNS)):
        (gens, vals), src = _hist_or_recompute(run, col, lambda rec=rec: (rgens, rec))
        if len(gens) < 3:
            rep.skip(f"{col} slope", f"only {len(gens)} generations")
            continue
        b, se, t, df, p = ols(gens, vals)
        rep.check(f"{col} slope", f"OLS {col}_g = a + b*g ({src}); t = b/SE(b)",
                  f"n = {len(gens)}; b = {fmt(b, 5)}/gen; SE = {fmt(se, 5)}; t = {fmt(t, 2)}; df = {df}; p = {fmt(p, 4)}",
                  f"{fmt(b * 100, 3)} {label}/episode per 100 generations",
                  "PASS" if math.isfinite(t) and t > 2 else "WARN",
                  f"PASS if t > 2 (agent touches/returns the ball more over time); WARN otherwise")
        q = max(2, len(vals) // 4)
        if len(vals) >= 2 * q:
            first, last = vals[:q], vals[-q:]
            diff, tw, dfw, pw = welch(first, last)
            rep.check(f"{col} first vs last quarter",
                      "Welch t: diff = mean(last q) - mean(first q); t = diff / sqrt(v1/n1 + v2/n2)",
                      f"q = {q}; first mean {fmt(mean(first), 3)}; last mean {fmt(mean(last), 3)}; "
                      f"t = {fmt(tw, 2)}; df = {fmt(dfw, 1)}; p = {fmt(pw, 4)}",
                      f"diff {fmt(diff, 3)}", "PASS" if math.isfinite(tw) and tw > 2 else "WARN", "PASS if t > 2")
            late = mean(last)
            rep.check(f"{col} vs reference policies",
                      f"late = mean of last {q} gens; ratio_random = late / {ref_rand}; "
                      f"progress = (late - {ref_rand}) / ({ref_bi} - {ref_rand})",
                      f"late {fmt(late, 3)}; random {ref_rand}; built-in policy {ref_bi}",
                      f"x{fmt(late / ref_rand, 2)} random play; {fmt(100 * (late - ref_rand) / (ref_bi - ref_rand), 1)}% "
                      f"of the way from random to the built-in policy", "INFO")

    best = run.hist_col("best_raw_contacts")
    if best and best[1]:
        q = max(1, len(best[1]) // 4)
        rep.check("best_raw_contacts start -> late", "mean(contacts) of the best-raw genome, first vs last quarter",
                  f"first {q} gens {fmt(mean(best[1][:q]), 3)}; last {q} gens {fmt(mean(best[1][-q:]), 3)}",
                  f"x{fmt(mean(best[1][-q:]) / RANDOM_CONTACTS, 2)} random play (late)", "INFO")

    # contact-score link, pooled over the last quarter of generations
    q = max(1, len(rgens) // 4)
    late_gens = set(rgens[-q:])
    eps = [(c, r, s) for g in run.genomes if g["gen"] in late_gens
           for c, r, s in zip(g["contacts"], g["returns"], g.get("scores", []))]
    if len(eps) < 3:
        rep.skip("contacts vs score", "fewer than 3 episodes in the last quarter")
        return
    cs, rs, ss = ([float(e[i]) for e in eps] for i in range(3))
    for name, xs in (("contacts", cs), ("returns", rs)):
        r = pearson(xs, ss)
        n = len(xs)
        t = r * math.sqrt((n - 2) / (1 - r * r)) if math.isfinite(r) and abs(r) < 1 else float("nan")
        rep.check(f"{name} vs score (episode level)",
                  f"r = pearson({name}_i, score_i) over every episode of the last {q} gens; t = r sqrt((n-2)/(1-r^2))",
                  f"n = {n}; mean {name} {fmt(mean(xs), 3)}; mean score {fmt(mean(ss), 3)}; r = {fmt(r, 3)}; "
                  f"t = {fmt(t, 2)}; p = {fmt(t_sf2(t, n - 2), 4)}",
                  f"r = {fmt(r, 3)} (positive = touching/returning the ball is what earns points)", "INFO")


def _champion_val_behaviour(run: Run):
    """Champion's validation rows (val_* fields); step-weighted act/sat, pooled contacts."""
    key = champion_key(run.champ) if run.champ else None
    if key is None or not run.val:
        return None
    rows = [v for v in run.val if v["key"] == key and "val_act_both" in v]
    if not rows:
        return None
    steps = [sum(v["val_steps"]) for v in rows]
    tot = sum(steps)
    out = {"rows": len(rows), "episodes": sum(len(v["val_steps"]) for v in rows)}
    for k in ("val_act_both", "val_out_sat", "val_act_idle", "val_act_fwd", "val_act_back", "val_act_jump"):
        if all(k in v for v in rows):
            out[k] = sum(v[k] * s for v, s in zip(rows, steps)) / tot
    for k in ("val_contacts", "val_returns", "val_opp_contacts"):
        if all(k in v for v in rows):
            out[k] = mean([x for v in rows for x in v[k]])
    return out


def check_degenerate(run: Run, rep: Report):
    rep.begin("13. Degenerate behaviour (cancel, juggling, saturation)")
    cb = _champion_val_behaviour(run)
    if cb and "val_act_both" in cb:
        ab = cb["val_act_both"]
        rep.check("champion stands still (fwd+back cancel)",
                  "act_both = sum_rows val_act_both * sum(val_steps) / sum(val_steps) over the champion's validation rows",
                  f"{cb['rows']} rows, {cb['episodes']} episodes; act_both {fmt(ab, 3)}"
                  + (f"; act_fwd {fmt(cb['val_act_fwd'], 3)}, act_back {fmt(cb['val_act_back'], 3)}"
                     if "val_act_fwd" in cb and "val_act_back" in cb else "")
                  + (f"; contacts/ep {fmt(cb['val_contacts'], 2)}, returns/ep {fmt(cb.get('val_returns', float('nan')), 2)}"
                     if "val_contacts" in cb else ""),
                  "champion presses fwd+back together most of the time: it stands still" if ab > 0.5
                  else f"cancels on {fmt(100 * ab, 1)}% of steps",
                  "WARN" if ab > 0.5 else "PASS", "WARN if act_both > 0.5")
    elif run.genomes and "act_both" in run.genomes[0] and champion_rows(run):
        crow = champion_rows(run)
        ab = _step_weighted(crow, "act_both")
        rep.check("champion stands still (fwd+back cancel)",
                  "act_both step-weighted over the champion's genomes.jsonl rows (no val_act_both logged)",
                  f"{len(crow)} rows; act_both {fmt(ab, 3)}",
                  "champion stands still" if ab > 0.5 else f"cancels on {fmt(100 * ab, 1)}% of steps",
                  "WARN" if ab > 0.5 else "PASS", "WARN if act_both > 0.5")
    else:
        rep.skip("champion stands still (fwd+back cancel)",
                 "no val_act_both / act_both for the champion (older log or champion.json missing)")

    if has_contacts(run):
        gens, c, r = contact_series(run)
        ser = behavior_series(run)
        to = dict(zip(ser["gen"], ser["timeout_frac"]))
        q = max(1, len(gens) // 4)
        late_c, late_r = mean(c[-q:]), mean(r[-q:])
        ratio = late_r / late_c if late_c > 0 else float("nan")
        to_first = mean([to[g] for g in gens[:q]])
        to_last = mean([to[g] for g in gens[-q:]])
        high = late_c >= 3 * RANDOM_CONTACTS
        juggling = high and math.isfinite(ratio) and ratio < 0.2 and to_last > to_first
        rep.check("juggling exploit (touch without returning)",
                  f"last {q} gens: contacts/ep, returns/ep, ratio = returns/contacts; timeout_frac first vs last quarter; "
                  f"flag if contacts >= 3 x random ({3 * RANDOM_CONTACTS:.1f}) and ratio < 0.2 and timeouts rising",
                  f"contacts {fmt(late_c, 3)}; returns {fmt(late_r, 3)}; ratio {fmt(ratio, 3)}; "
                  f"timeout_frac {fmt(to_first, 3)} -> {fmt(to_last, 3)}",
                  "agent keeps the ball on its side to run out the clock" if juggling
                  else ("contacts not high enough to juggle" if not high else "no juggling signature"),
                  "WARN" if juggling else "PASS", "WARN on the signature above")
    else:
        rep.skip("juggling exploit (touch without returning)", "no contacts/returns (older log)")

    sat = run.hist_col("mean_sat")
    if not (sat and sat[1]) and run.genomes and "out_sat" in run.genomes[0]:
        by_gen = run.genomes_by_gen()
        sat = (list(by_gen), [_step_weighted(p, "out_sat") for p in by_gen.values()])
    if sat and sat[1]:
        q = max(1, len(sat[1]) // 4)
        late = mean(sat[1][-q:])
        rep.check("output saturation", "mean_sat = fraction of network outputs < 0.02 or > 0.98; last-quarter mean",
                  f"first {q} gens {fmt(mean(sat[1][:q]), 3)}; last {q} gens {fmt(late, 3)}",
                  (f"{fmt(100 * late, 1)}% of outputs are all-or-nothing: most small weight mutations change "
                   "no action, so the fitness landscape is flat locally") if late > 0.8
                  else f"{fmt(100 * late, 1)}% saturated", "INFO", "note if > 0.8")
    else:
        rep.skip("output saturation", "no mean_sat column or out_sat field (older log)")
    cancel = run.hist_col("mean_cancel")
    if cancel and cancel[1]:
        q = max(1, len(cancel[1]) // 4)
        rep.check("population fwd+back cancel", "mean_cancel = step-weighted act_both over the population",
                  f"first {q} gens {fmt(mean(cancel[1][:q]), 3)}; last {q} gens {fmt(mean(cancel[1][-q:]), 3)}",
                  f"{fmt(100 * mean(cancel[1][-q:]), 1)}% of late steps press both directions (no movement)", "INFO")


def check_search_health(run: Run, rep: Report):
    rep.begin("14. Search health (structure drift, NEAT stagnation)")
    s = run.hist_col("mean_conns")
    src = "fitness_history.csv"
    if not (s and s[1]) and run.genomes and "conns" in run.genomes[0]:
        cs = complexity_series(run)
        s, src = (cs["gen"], cs["conns"]), "recomputed from genomes.jsonl"
    if s and len(s[1]) >= 3:
        b, se, t, df, p = ols(*s)
        c0, c1 = s[1][0], s[1][-1]
        change = (c1 - c0) / c0 if c0 else float("nan")
        drift = math.isfinite(change) and change < -0.5
        rep.check("mean_conns trend", f"OLS mean_conns_g = a + b*g ({src}); change = (end - start) / start",
                  f"start {fmt(c0, 2)} -> end {fmt(c1, 2)}; b = {fmt(b, 4)}/gen; t = {fmt(t, 2)}; p = {fmt(p, 4)}",
                  f"{fmt(100 * change, 1)}% connections"
                  + (": structural drift toward deletion (delete-conn outpaces add-conn under weak selection)"
                     if drift else ""),
                  "WARN" if drift else "PASS", "WARN if mean_conns falls by > 50%")
    else:
        rep.skip("mean_conns trend", "no mean_conns column and no conns field")

    if run.stdout is None:
        rep.skip("NEAT stagnation / extinction", "stdout.log missing (older run)")
        return
    lines = run.stdout.splitlines()
    stagnant = [ln for ln in lines if "stagnat" in ln.lower()]
    all_extinct = sum(1 for ln in lines if "all species extinct" in ln.lower())
    totals = []
    for ln in lines:
        if ln.strip().lower().startswith("total extinctions:"):
            try:
                totals.append(int(ln.split(":", 1)[1]))
            except ValueError:
                pass
    rep.check("NEAT stagnation / extinction (stdout.log)",
              "count lines containing 'stagnat'; last 'Total extinctions: N'; count 'All species extinct'",
              f"{len(lines):,} lines; stagnation lines {len(stagnant)}; final total extinctions "
              f"{totals[-1] if totals else 'n/a'} ({len(totals)} reports); 'all species extinct' {all_extinct}"
              + (f"; first: {stagnant[0].strip()!r}" if stagnant else ""),
              f"{len(stagnant)} species removed for stagnation, {totals[-1] if totals else 0} total extinctions",
              "INFO")


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def _cum_steps_axis(ax, run: Run):
    cum = run.hist_col("cum_steps")
    if not cum or len(cum[0]) < 2:
        return
    gens, steps = cum

    def g2s(g):
        import numpy as np
        return np.interp(g, gens, steps) / 1e6

    def s2g(s):
        import numpy as np
        return np.interp(s * 1e6, steps, gens)

    top = ax.secondary_xaxis("top", functions=(g2s, s2g))
    lo, hi = steps[0] / 1e6, steps[-1] / 1e6
    ticks = [t for t in matplotlib.ticker.MaxNLocator(6).tick_values(lo, hi) if lo <= t <= hi]
    top.set_xticks(ticks)
    top.set_xlabel("cumulative env steps after generation (millions)")


def plot_fitness(run: Run, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    shaped = run.bonus() > 0 or run.return_bonus() > 0
    for col, label, style in (("best_fitness", "train best (selection)" if shaped else "train best", "-"),
                              ("mean_fitness", "train mean (selection)" if shaped else "train mean", "-"),
                              ("best_raw", "train best raw", ":"), ("mean_raw", "train mean raw", ":"),
                              ("val_best", "val best (this gen)", "--"), ("champion_val", "champion val", "-")):
        series = run.hist_col(col)
        if series and (shaped or not col.endswith("_raw")):
            ax.plot(series[0], series[1], style, label=label, lw=1.5)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("generation")
    ax.set_ylabel("score (points won - lost)")
    ax.set_ylim(-5.2, 5.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    _cum_steps_axis(ax, run)
    fig.tight_layout()
    fig.savefig(out / "fitness_curve.png", dpi=120)
    plt.close(fig)


def plot_seed_noise(deltas, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_d = [d for ds in (deltas or {}).values() for d in ds]
    if all_d:
        ax.hist(all_d, bins=min(40, max(10, len(set(all_d)))), color="#4a78b5", alpha=0.85)
        ax.axvline(mean(all_d), color="k", lw=1, label=f"mean {mean(all_d):+.3f}, sd {sd(all_d):.3f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no elite pairs", ha="center", transform=ax.transAxes)
    ax.set_xlabel("fitness_g(k) - fitness_{g-1}(k), same genome, new seeds")
    ax.set_ylabel("count")
    ax.set_title("Elite re-evaluation delta (pure seed noise)")
    fig.tight_layout()
    fig.savefig(out / "seed_noise.png", dpi=120)
    plt.close(fig)


def plot_behavior(run: Run, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if run.genomes:
        ser = behavior_series(run)
        for k in ACT_KEYS:
            if not all(math.isnan(v) for v in ser[k]):
                ax.plot(ser["gen"], ser[k], label=k)
        ax.plot(ser["gen"], ser["timeout_frac"], "k--", label="timeout frac")
        for col, label, style in (("mean_cancel", "fwd+back cancel (act_both)", "-."),
                                  ("mean_sat", "outputs saturated", ":")):
            s = run.hist_col(col)
            if s and s[1]:
                ax.plot(s[0], s[1], style, label=label, lw=1.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("generation")
    ax.set_ylabel("fraction (population mean)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "behavior.png", dpi=120)
    plt.close(fig)


def plot_contacts(run: Run, out: Path):
    """mean_contacts, mean_returns, best_raw_contacts per generation; skipped on older logs."""
    series = {}
    for col in ("mean_contacts", "mean_returns", "best_raw_contacts"):
        s = run.hist_col(col)
        if s and s[1]:
            series[col] = s
    if not series and has_contacts(run):
        gens, c, r = contact_series(run)
        series = {"mean_contacts": (gens, c), "mean_returns": (gens, r)}
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, label, style in (("mean_contacts", "mean contacts / episode", "-"),
                              ("mean_returns", "mean returns / episode", "-"),
                              ("best_raw_contacts", "contacts of best-raw genome", ":")):
        if col in series:
            ax.plot(series[col][0], series[col][1], style, label=label, lw=1.5)
    ax.axhline(RANDOM_CONTACTS, color="grey", ls="--", lw=1, label=f"random play contacts ({RANDOM_CONTACTS})")
    ax.set_xlabel("generation")
    ax.set_ylabel("per episode")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Ball contacts (built-in policy on our side: ~{BUILTIN_CONTACTS:g} contacts, "
                 f"{BUILTIN_RETURNS:g} returns)", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    _cum_steps_axis(ax, run)
    fig.tight_layout()
    fig.savefig(out / "contacts.png", dpi=120)
    plt.close(fig)


def plot_complexity(run: Run, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if run.genomes:
        cs = complexity_series(run)
        ax.plot(cs["gen"], cs["nodes"], label="mean nodes")
        ax.plot(cs["gen"], cs["conns"], label="mean conns")
    for col, label in (("best_nodes", "best nodes"), ("best_conns", "best conns")):
        s = run.hist_col(col)
        if s:
            ax.plot(s[0], s[1], ":", label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    ns = run.hist_col("n_species")
    if ns:
        ax2 = ax.twinx()
        ax2.plot(ns[0], ns[1], color="tab:purple", alpha=0.7, label="n_species")
        ax2.set_ylabel("species", color="tab:purple")
        ax2.legend(loc="upper right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "complexity.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_check(fn, run, rep):
    try:
        return fn(run, rep)
    except Exception as exc:  # one broken check must not hide the others
        rep.out(f"### [ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
        rep.out("```")
        rep.out(traceback.format_exc().rstrip())
        rep.out("```")
        rep.summary.append((f"{rep.section}: {fn.__name__}", f"{type(exc).__name__}: {exc}", "ERROR"))
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="output directory (default <run_dir>/analysis)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    run = Run(run_dir)
    rep = Report()
    rep.out(f"# analyze_run: {run_dir}")
    for name, obj in (("run_args.json", run.args), ("budget_log.jsonl", run.budget),
                      ("fitness_history.csv", run.hist), ("genomes.jsonl", run.genomes),
                      ("validation.jsonl", run.val), ("champion.json", run.champ)):
        size = "missing" if obj is None else (f"{len(obj)} rows" if isinstance(obj, list) else "present")
        rep.out(f"- {name}: {size}")

    run_check(check_budget, run, rep)
    run_check(check_fitness, run, rep)
    run_check(check_seeds, run, rep)
    deltas = run_check(check_seed_noise, run, rep)
    run_check(check_winners_curse, run, rep)
    run_check(check_validation_noise, run, rep)
    run_check(check_behavior, run, rep)
    run_check(check_learning, run, rep)
    run_check(check_population, run, rep)
    run_check(check_shaping, run, rep)
    run_check(check_behavior_logs, run, rep)
    run_check(check_contacts, run, rep)
    run_check(check_degenerate, run, rep)
    run_check(check_search_health, run, rep)

    for plot in (lambda: plot_fitness(run, out), lambda: plot_seed_noise(deltas, out),
                 lambda: plot_behavior(run, out), lambda: plot_complexity(run, out),
                 lambda: plot_contacts(run, out)):
        try:
            plot()
        except Exception as exc:
            print(f"plot failed: {type(exc).__name__}: {exc}")

    (out / "analysis.md").write_text(rep.markdown(run_dir))
    print("")
    print("SUMMARY")
    for name, result, verdict in rep.summary:
        print(f"  [{verdict:<5}] {name}: {result}")
    print(f"\nwrote {out}/analysis.md and plots")
    return 1 if any(v in ("FAIL", "ERROR") for _, _, v in rep.summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
