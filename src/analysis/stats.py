"""Statistical analysis of the master results.

Outputs:
    results/stats_summary.csv   mean +/- std distance & gap per method/instance/delta
    results/wilcoxon.csv        Wilcoxon signed-rank tests, with effect size and a
                                 bootstrap CI on the mean paired difference, for
                                 every comparison cited in the dissertation
    results/degradation.csv     % cost degradation under dynamism per method

Pairing for Wilcoxon: per-instance mean distance (n=8 pairs per test),
the standard treatment when methods have different stochasticity
structures (deterministic solvers vs. seeded learners).

Every comparison named in the dissertation text (Chapter 3 Section 3.5.4,
Chapter 4 Sections 4.2/4.4/4.7) is computed here, including the two that
were previously calculated ad hoc rather than by this script: DQN vs.
static-trained DQN (the decomposition finding) and fixed-plan vs.
live-priced-replan (the "no significant gain from live information"
finding). Reproducibility requires every cited number to come from a
committed script, not a one-off calculation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, rankdata, wilcoxon

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Every comparison reported in the dissertation, beyond the default "DQN vs
# every other method" sweep that wilcoxon_tests() already performs.
NAMED_COMPARISONS = [
    ("dqn", "dqn_static_policy"),        # decomposition: training vs. execution
    ("ortools_fixed", "ortools_replan_live"),  # value of live-priced re-planning
    ("ortools_fixed", "ortools_replan"),       # value of static-matrix re-planning
]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["delta", "method", "instance"])
    out = g.agg(
        mean_distance=("distance", "mean"),
        std_distance=("distance", "std"),
        mean_gap_pct=("gap_pct", "mean"),
        std_gap_pct=("gap_pct", "std"),
        mean_wall_s=("wall_time_s", "mean"),
        mean_replan_s=("replan_time_s", "mean"),
        n=("distance", "size"),
    ).reset_index()
    return out


def rank_biserial_r(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the standard effect size for
    a Wilcoxon signed-rank test (Kerby, 2014). +/-1 = every pair agrees in
    direction; 0 = ranks split evenly between the two directions. Ties
    (d == 0) are dropped, matching scipy.stats.wilcoxon's default handling."""
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 9999,
                  seed: int = 0) -> tuple[float, float, float]:
    """95% percentile bootstrap CI on the mean paired difference (a - b).
    Percentile rather than BCa: at n=8 pairs, BCa's bias-correction and
    acceleration estimates are themselves too noisy to trust; percentile
    is the more honest choice at this sample size."""
    d = a - b
    rng = np.random.default_rng(seed)
    res = bootstrap((d,), np.mean, n_resamples=n_resamples,
                     confidence_level=0.95, method="percentile",
                     random_state=rng)
    return float(d.mean()), float(res.confidence_interval.low), \
        float(res.confidence_interval.high)


def _paired_test(sub: pd.DataFrame, method_a: str, method_b: str) -> dict | None:
    """One fully-specified paired comparison: Wilcoxon p, effect size, and a
    bootstrap CI on the mean difference in both distance units and as a
    percentage of the baseline (method_b) mean — the operational-significance
    reading, since a statistically real effect can still be too small to act
    on, and vice versa at small n."""
    if method_a not in sub or method_b not in sub:
        return None
    paired = sub[[method_a, method_b]].dropna()
    if len(paired) < 5:
        return None
    a, b = paired[method_a].to_numpy(), paired[method_b].to_numpy()
    stat, p = wilcoxon(a, b)
    r = rank_biserial_r(a, b)
    mean_diff, ci_low, ci_high = bootstrap_ci(a, b)
    baseline_mean = b.mean()
    return dict(
        n_pairs=len(paired), statistic=stat, p_value=p,
        rank_biserial_r=r,
        a_mean=a.mean(), b_mean=baseline_mean,
        a_better=bool(a.mean() < baseline_mean),
        mean_diff=mean_diff, mean_diff_ci_low=ci_low, mean_diff_ci_high=ci_high,
        mean_diff_pct_of_baseline=100 * mean_diff / baseline_mean,
        mean_diff_pct_ci_low=100 * ci_low / baseline_mean,
        mean_diff_pct_ci_high=100 * ci_high / baseline_mean,
    )


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """DQN vs. every other method (the original sweep), plus every named
    comparison in NAMED_COMPARISONS, each per dynamism level available."""
    rows = []
    per_inst = (
        df.groupby(["delta", "method", "instance"])["distance"].mean().reset_index()
    )
    for delta in per_inst["delta"].unique():
        sub = per_inst[per_inst.delta == delta].pivot(
            index="instance", columns="method", values="distance"
        )
        named_baselines = {b for a, b in NAMED_COMPARISONS if a == "dqn"}
        if "dqn" in sub:
            for baseline in [c for c in sub.columns
                              if c != "dqn" and c not in named_baselines]:
                result = _paired_test(sub, "dqn", baseline)
                if result:
                    rows.append(dict(delta=delta,
                                      comparison=f"dqn_vs_{baseline}", **result))
        for method_a, method_b in NAMED_COMPARISONS:
            result = _paired_test(sub, method_a, method_b)
            if result:
                rows.append(dict(delta=delta,
                                  comparison=f"{method_a}_vs_{method_b}", **result))
    out = pd.DataFrame(rows)
    if not out.empty:
        # a_mean/b_mean/a_better are always correct regardless of which two
        # methods a row compares. dqn_mean/baseline_mean/dqn_better are kept
        # ONLY as a convenience alias for rows that actually compare against
        # DQN (comparison starts with "dqn_vs_") — populating them for every
        # row (including the ortools_fixed-vs-ortools_replan* named rows,
        # where neither method is DQN) previously mislabelled those rows'
        # OR-Tools means as "DQN mean" in Table 3.7 of the dissertation
        # (found in the 2026-08-05 audit). Consumers of the non-DQN named
        # comparisons must use a_mean/b_mean/a_better directly.
        is_dqn_row = out["comparison"].str.startswith("dqn_vs_")
        out["dqn_mean"] = out["a_mean"].where(is_dqn_row)
        out["baseline_mean"] = out["b_mean"].where(is_dqn_row)
        out["dqn_better"] = out["a_better"].where(is_dqn_row)
    return out


def degradation(df: pd.DataFrame) -> pd.DataFrame:
    """% increase in mean distance vs. the method's own static performance.

    ortools_replan has no static rows (it equals ortools_fixed there), so
    its reference is ortools_fixed's static solution — the plan it would
    have produced before any disruption.
    """
    per = df.groupby(["method", "delta", "instance"])["distance"].mean().unstack("delta")
    rows = []
    static_ref = per.xs("ortools_fixed", level="method")["static"]
    for method in per.index.get_level_values("method").unique():
        block = per.xs(method, level="method")
        ref = block["static"] if "static" in block and block["static"].notna().all() \
            else static_ref
        for delta in ("low", "high"):
            if delta not in block:
                continue
            deg = 100 * (block[delta] - ref) / ref
            rows.append(dict(
                method=method, delta=delta,
                mean_degradation_pct=deg.mean(), std_degradation_pct=deg.std(),
            ))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RESULTS / "results_master.csv")

    summary = summarise(df)
    summary.to_csv(RESULTS / "stats_summary.csv", index=False)

    # Merge in the decomposition condition (static-trained policies on
    # dynamic episodes) so dqn_vs_dqn_static_policy is a directly testable
    # pairwise comparison, not something computed outside this script.
    wilcoxon_df = df
    cross = RESULTS / "cross_eval.csv"
    if cross.exists():
        wilcoxon_df = pd.concat([df, pd.read_csv(cross)], ignore_index=True)

    wtests = wilcoxon_tests(wilcoxon_df)
    wtests.to_csv(RESULTS / "wilcoxon.csv", index=False)

    # Decomposition condition, again, for the degradation table (Table 3.8):
    # its degradation reference is the same checkpoints' static performance,
    # i.e. the dqn static rows relabelled.
    deg_df = df
    if cross.exists():
        x = pd.read_csv(cross)
        ref = df[(df.method == "dqn") & (df.delta == "static")].assign(
            method="dqn_static_policy")
        deg_df = pd.concat([df, x, ref], ignore_index=True)

    deg = degradation(deg_df)
    deg.to_csv(RESULTS / "degradation.csv", index=False)

    print("=== mean optimality gap %% (static) by method ===")
    static = summary[summary.delta == "static"]
    print(static.groupby("method")["mean_gap_pct"].mean().round(1).to_string())
    print("\n=== Wilcoxon signed-rank, with effect size and 95% bootstrap CI ===")
    if not wtests.empty:
        cols = ["delta", "comparison", "n_pairs", "p_value", "rank_biserial_r",
                "mean_diff", "mean_diff_ci_low", "mean_diff_ci_high"]
        print(wtests[cols].round(4).to_string(index=False))
    print("\n=== degradation under dynamism ===")
    print(deg.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
