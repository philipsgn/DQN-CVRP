"""Dissertation figures -> figures/ (300 dpi PNG + vector PDF).

Fig 1  DQN training convergence, small multiples per instance
Fig 2  Optimality gap distribution by method (static), box plot
Fig 3  Mean optimality gap by method (static), bars + error bars
Fig 4  Adaptability: cost degradation vs. dynamism level  (headline)
Fig 5  Computational cost per method (log scale)
Fig 6  Route maps: OR-Tools static plan vs. DQN under disruption

Colors follow the entity across every figure (validated palette):
dqn=blue, ortools_fixed=aqua, nearest_neighbour=yellow,
ortools_replan=violet, random=red.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..agents.dqn import DQNAgent, DQNConfig
from ..envs.instances import INSTANCES, load
from ..experiments.rollout import rollout_policy
from ..experiments.train import GreedyDQNPolicy, make_env

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TRAINING = RESULTS / "training"
FIGURES = ROOT / "figures"

COLOR = {
    "dqn": "#2a78d6",
    "dqn_static_policy": "#8fb8e6",
    "ortools_fixed": "#1baf7a",
    "nearest_neighbour": "#eda100",
    "ortools_replan": "#4a3aa7",
    "ortools_replan_live": "#9b8ce8",
    "random": "#e34948",
}
LABEL = {
    "dqn": "DQN",
    "dqn_static_policy": "DQN (static-trained, blind)",
    "ortools_fixed": "OR-Tools (fixed plan)",
    "nearest_neighbour": "Nearest neighbour",
    "ortools_replan": "OR-Tools (re-plan)",
    "ortools_replan_live": "OR-Tools (re-plan, live-priced)",
    "random": "Random",
}
ORDER = ["random", "nearest_neighbour", "dqn", "dqn_static_policy",
         "ortools_replan_live", "ortools_replan", "ortools_fixed"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 11,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
})


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png (+pdf)")


def fig1_convergence():
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=False)
    for ax, name in zip(axes.ravel(), INSTANCES):
        inst = load(name)
        curves = []
        for seed in range(5):
            f = TRAINING / f"{name}_static_s{seed}_log.csv"
            if f.exists():
                d = pd.read_csv(f)["distance"].rolling(50, min_periods=10).mean()
                curves.append(d)
        if curves:
            arr = pd.concat(curves, axis=1)
            ax.fill_between(arr.index, arr.min(axis=1), arr.max(axis=1),
                            color=COLOR["dqn"], alpha=0.15, linewidth=0)
            ax.plot(arr.index, arr.mean(axis=1), color=COLOR["dqn"], linewidth=2)
        ax.axhline(inst.best_known_cost, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(f"{name}  (n={inst.n_customers})", fontsize=11)
        ax.tick_params(labelsize=9.5)
    axes[0, 0].text(0.97, 0.9, "best known", transform=axes[0, 0].transAxes,
                    ha="right", fontsize=9.5, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("route distance (rolling mean)")
    fig.suptitle("DQN training convergence — static instances, 5 seeds "
                 "(band = seed min–max)", y=1.02)
    save(fig, "fig1_convergence")


def fig2_box_gap(df: pd.DataFrame):
    static = df[df.delta == "static"]
    methods = [m for m in ORDER if m in static.method.unique()]
    data = [static[static.method == m].gap_pct.values for m in methods]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLOR[m])
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    ax.set_yticklabels([LABEL[m] for m in methods])
    ax.set_xlabel("optimality gap vs. best-known (%)")
    ax.set_title("Solution quality on static CVRP benchmarks (all instances, all seeds)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_gap_boxplot")


def fig3_bar_gap(df: pd.DataFrame):
    static = df[df.delta == "static"]
    methods = [m for m in ORDER if m in static.method.unique()]
    means = [static[static.method == m].gap_pct.mean() for m in methods]
    stds = [static[static.method == m].gap_pct.std() for m in methods]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(methods))
    ax.barh(y, means, xerr=stds, height=0.55,
            color=[COLOR[m] for m in methods],
            error_kw=dict(ecolor=INK2, capsize=3, linewidth=1))
    for yi, v in zip(y, means):
        ax.text(v + max(means) * 0.02 + 3, yi, f"{v:.1f}%", va="center",
                fontsize=10.5, color=INK)
    ax.set_yticks(y, [LABEL[m] for m in methods])
    ax.set_xlabel("mean optimality gap (%)  — error bars: ±1 std")
    ax.set_title("Mean optimality gap by method (static instances)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_gap_bars")


def fig4_adaptability():
    deg = pd.read_csv(RESULTS / "degradation.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = {"static": 0, "low": 1, "high": 2}
    label = dict(LABEL, dqn="DQN (trained under disruption)")
    ends = []
    for m in [m for m in ORDER if m in deg.method.unique()]:
        block = deg[deg.method == m].set_index("delta")
        pts = [(0, 0.0)] + [
            (xs[d], block.loc[d, "mean_degradation_pct"])
            for d in ("low", "high") if d in block.index
        ]
        px, py = zip(*pts)
        ax.plot(px, py, color=COLOR[m], linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5)
        ends.append([m, px[-1], py[-1], py[-1]])
    # dodge end-of-line labels that would overlap (min vertical gap in data units)
    gap = 0.9
    for prev, cur in zip(sorted(ends, key=lambda e: e[3]),
                         sorted(ends, key=lambda e: e[3])[1:]):
        if cur[3] - prev[3] < gap:
            cur[3] = prev[3] + gap
    for m, x, y, ylab in ends:
        ax.annotate(label[m], (x, ylab), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    color=COLOR[m], fontweight="bold")
    ax.set_xticks([0, 1, 2], ["static", "low dynamism", "high dynamism"])
    ax.set_ylabel("cost degradation vs. own static performance (%)")
    ax.set_title("Adaptability under traffic disruption")
    ax.set_xlim(-0.2, 2.9)
    save(fig, "fig4_adaptability")


def fig5_runtime(df: pd.DataFrame):
    rows = []
    for m in ORDER:
        sub = df[df.method == m]
        if sub.empty:
            continue
        # wall_time_s already includes any in-loop solve() time (rollout_policy's
        # timer wraps the whole episode, and replan_time_s is accumulated inside
        # policy.act() during that same window) — do not add replan_time_s again,
        # that double-counts and inflates this figure ~2x (see RESULTS_NOTES.md).
        t = sub.wall_time_s.mean()
        rows.append((m, t))
    # DQN training cost (mean over runs) as its own bar — same axis, one scale
    train_times = []
    for f in TRAINING.glob("*_log.csv"):
        train_times.append(pd.read_csv(f, usecols=["train_time_s"]).iloc[0, 0])
    if train_times:
        rows.append(("dqn_train", float(np.mean(train_times))))

    labels = [LABEL.get(m, "DQN training (one-off)") for m, _ in rows]
    vals = [t for _, t in rows]
    colors = [COLOR.get(m, COLOR["dqn"]) for m, _ in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.55, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v * 1.15, yi, f"{v:.2f}s" if v < 60 else f"{v / 60:.0f} min",
                va="center", fontsize=10.5, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlabel("wall-clock time per episode, log scale (training amortised once)")
    ax.set_title("Computational cost: decision time vs. solve time")
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_runtime")


def fig6_routes(instance_name: str = "A-n32-k5", train_seed: int = 0):
    inst = load(instance_name)

    def draw(ax, route_nodes: list[int], title: str):
        # split executed node sequence into depot-to-depot subroutes
        subroutes, cur = [], []
        for node in route_nodes[1:]:
            if node == 0:
                if cur:
                    subroutes.append(cur)
                cur = []
            else:
                cur.append(node)
        palette = list(COLOR.values())
        for i, r in enumerate(subroutes):
            path = np.array([inst.coords[j] for j in [0, *r, 0]])
            ax.plot(path[:, 0], path[:, 1], color=palette[i % len(palette)],
                    linewidth=1.6, alpha=0.9, zorder=1)
        ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1],
                   s=12 + 2.5 * inst.demands[1:], c="white",
                   edgecolors=INK2, linewidths=0.8, zorder=2)
        ax.scatter(*inst.coords[0], marker="s", s=90, c=INK, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    from ..baselines.ortools_solver import solve as ortools_solve
    from ..experiments.rollout import execute_fixed_routes

    env = make_env(inst, "high", seed=1002)
    routes, _, _ = ortools_solve(inst, time_limit_s=10)
    fixed = execute_fixed_routes(env, routes, seed=1002)

    ckpt = TRAINING / f"{instance_name}_high_s{train_seed}_final.pt"
    env2 = make_env(inst, "high", seed=1002)
    agent = DQNAgent(env2.observation_space.shape[0], env2.action_space.n,
                     DQNConfig(), seed=train_seed)
    agent.load(ckpt)
    dqn = rollout_policy(env2, GreedyDQNPolicy(agent), seed=1002)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    draw(axes[0], fixed["route"],
         f"OR-Tools fixed plan under disruption — cost {fixed['distance']:.0f}")
    draw(axes[1], dqn["route"],
         f"DQN adaptive route, same disruption — cost {dqn['distance']:.0f}")
    fig.suptitle(f"{instance_name} under high dynamism (depot = black square, "
                 "marker size = demand)", y=1.0)
    save(fig, "fig6_routes")


def main():
    df = pd.read_csv(RESULTS / "results_master.csv")
    fig1_convergence()
    fig2_box_gap(df)
    fig3_bar_gap(df)
    fig4_adaptability()
    fig5_runtime(df)
    fig6_routes()


if __name__ == "__main__":
    main()
