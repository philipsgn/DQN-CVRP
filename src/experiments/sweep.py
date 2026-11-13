"""Parallel training sweep: all instances x seeds x dynamism levels.

Runs training jobs as single-threaded subprocesses on a worker pool.
Resumable: jobs whose final checkpoint already exists are skipped.

    python -m src.experiments.sweep --workers 6 --seeds 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..envs.instances import INSTANCES

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "results" / "training"


def run_job(instance: str, delta: str, seed: int) -> tuple[str, bool, float]:
    run = f"{instance}_{delta}_s{seed}"
    if (TRAINING / f"{run}_final.pt").exists():
        return run, True, 0.0
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "src.experiments.train",
         "--instance", instance, "--delta", delta, "--seed", str(seed), "--quiet"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    if not ok:
        (TRAINING / f"{run}_ERROR.log").write_text(proc.stdout + "\n" + proc.stderr)
    return run, ok, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--deltas", nargs="+", default=["static", "low", "high"])
    args = p.parse_args()

    TRAINING.mkdir(parents=True, exist_ok=True)
    # static first: it is the safety-net result set
    jobs = [
        (inst, delta, seed)
        for delta in args.deltas
        for inst in INSTANCES
        for seed in range(args.seeds)
    ]
    print(f"{len(jobs)} jobs on {args.workers} workers", flush=True)

    done = failed = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, *job): job for job in jobs}
        for fut in as_completed(futures):
            run, ok, dt = fut.result()
            done += 1
            failed += (not ok)
            print(f"[{done}/{len(jobs)}] {'ok ' if ok else 'FAIL'} {run} "
                  f"({dt:.0f}s, elapsed {(time.perf_counter() - t0) / 60:.0f}m)",
                  flush=True)
    print(f"sweep complete: {done - failed} ok, {failed} failed, "
          f"{(time.perf_counter() - t0) / 60:.0f} minutes", flush=True)


if __name__ == "__main__":
    main()
