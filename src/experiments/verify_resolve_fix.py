"""Verify the OR-Tools re-solve mispricing fix's headline claim (Chapter 3,
Section 3.4.3): "forced re-planning on the static matrix reproduces the
best-known solution on six of the eight instances and is within one percent
on the remaining two."

Rationale (restated from build_ch3.py's own prose, which is the source of
this claim): if the mid-route re-solve machinery is correct, forcing a
re-solve every three steps on the *unperturbed static* matrix must reproduce
the known optimum, because each re-solve sees exactly the problem the
one-shot solver saw — any drift is re-solve implementation error, not a
property of dynamism.

`ORToolsReplanPolicy` (src/baselines/ortools_replan.py) only re-solves when
`info["traffic_version"]` changes, which never happens on a plain (static)
`RouteEnv` — so it re-solves once and stops. This script forces a re-solve
every `REPLAN_EVERY` steps regardless, on the static environment, and checks
the resulting tour cost against each instance's published best-known
solution. This was previously an unbacked prose claim (found in the
2026-08-05 audit — no committed script, test or CSV supported it); this
script and its output CSV are that missing evidence.

Run:  python -m src.experiments.verify_resolve_fix
Output: results/resolve_fix_verification.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..baselines.ortools_replan import ORToolsReplanPolicy
from ..envs.instances import INSTANCES, load
from ..envs.route_env import RouteEnv
from .rollout import rollout_policy

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

REPLAN_EVERY = 3  # matches the "every three steps" claim in build_ch3.py §3.4.3
GAP_TOLERANCE_PCT = 1.0  # "within one percent" threshold from the same claim


class ForcedReplanPolicy(ORToolsReplanPolicy):
    """Same re-solve machinery as the production replanner, but triggers a
    re-solve every `every` actions taken, instead of only when
    `info["traffic_version"]` changes (which never happens on a static
    env — the production trigger is event-based, not time-based, and is
    exactly right for the dynamic evaluation it's built for; this variant
    exists only to stress-test the re-solve machinery itself in isolation
    from the disruption model)."""

    def __init__(self, every: int = REPLAN_EVERY, **kwargs):
        super().__init__(**kwargs)
        self.every = every
        self._steps_since_replan = 0

    def act(self, env, obs, info) -> int:
        if not self.queue or self._steps_since_replan >= self.every:
            self._replan(env, info)
            self._steps_since_replan = 0
        if not self.queue:
            return 0
        nxt = self.queue[0]
        if nxt != 0 and not info["action_mask"][nxt]:
            self._steps_since_replan += 1
            return 0
        self._steps_since_replan += 1
        return self.queue.pop(0)


def main():
    rows = []
    for name in INSTANCES:
        instance = load(name)
        env = RouteEnv(instance, seed=0)
        policy = ForcedReplanPolicy(mode="expectation")
        result = rollout_policy(env, policy, seed=0)
        bks = instance.best_known_cost
        gap_pct = 100 * (result["distance"] - bks) / bks
        rows.append(dict(
            instance=name, n_customers=instance.n_customers, bks=bks,
            distance=round(result["distance"], 4), gap_pct=round(gap_pct, 4),
            replan_count=policy.replan_count,
            exact_bks=bool(abs(result["distance"] - bks) < 1e-6),
            within_tolerance=bool(gap_pct <= GAP_TOLERANCE_PCT + 1e-9),
        ))

    df = pd.DataFrame(rows)
    out = RESULTS / "resolve_fix_verification.csv"
    df.to_csv(out, index=False)

    n_exact = int(df.exact_bks.sum())
    n_within = int(df.within_tolerance.sum())
    print(df.to_string(index=False))
    print(f"\nExact BKS match: {n_exact}/{len(df)} instances")
    print(f"Within {GAP_TOLERANCE_PCT}% of BKS: {n_within}/{len(df)} instances")
    print(f"Max gap among non-exact instances: "
          f"{df[~df.exact_bks].gap_pct.max() if n_exact < len(df) else 0:.2f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
