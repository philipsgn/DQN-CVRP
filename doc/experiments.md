# Experimental Design

## Overview

This document describes the experimental framework used to evaluate **Deep Reinforcement Learning for Dynamic Route Optimisation**.

The experiments were designed to compare a masked **Double Deep Q-Network (Double DQN)** with classical optimisation and heuristic routing approaches under both static and changing travel conditions.

The evaluation focuses on four questions:

1. Can the DQN learn feasible and effective routing policies?
2. How does its solution quality compare with established routing methods?
3. How does performance change when travel conditions become dynamic?
4. What trade-off exists between solution quality and computational decision cost?

---

# 1. Benchmark Instances

Experiments use eight instances from **CVRPLIB**, a standard benchmark collection for vehicle-routing research.

| Instance | Customers | Vehicle Capacity | Best-Known Cost |
|---|---:|---:|---:|
| P-n16-k8 | 15 | 35 | 450 |
| P-n20-k2 | 19 | 160 | 216 |
| E-n22-k4 | 21 | 6000 | 375 |
| E-n23-k3 | 22 | 4500 | 569 |
| B-n31-k5 | 30 | 100 | 672 |
| A-n32-k5 | 31 | 100 | 784 |
| A-n33-k5 | 32 | 100 | 661 |
| A-n45-k6 | 44 | 100 | 944 |

The selected instances provide variation in:

- number of customers;
- vehicle capacity;
- routing complexity; and
- reference solution cost.

Benchmark data are stored under:

```text
data/instances/
```

The repository contains both problem instances and corresponding best-known solution files used for reference checking.

---

# 2. Experimental Conditions

Three travel conditions are investigated.

## Static

The static condition assumes no change in travel cost during route execution.

```text
Travel multiplier = 1.0
```

This provides the baseline environment for comparing routing quality under stable conditions.

---

## Low Disruption

Low disruption introduces relatively small stochastic changes to travel costs.

The dynamic environment uses log-normal travel multipliers with:

```text
σ = 0.15
```

This represents moderate variation in routing conditions.

---

## High Disruption

High disruption introduces stronger stochastic variation.

The dynamic environment uses:

```text
σ = 0.40
```

This condition provides a more demanding test of policy behaviour when travel costs change substantially.

---

# 3. Dynamic Update Process

Traffic multipliers are re-sampled every three environment steps.

This means that the effective cost of travelling between nodes can change while a route is being executed.

The dynamic process is seeded.

This is important because it allows different routing approaches to experience comparable disruption realisations during evaluation.

The experimental design therefore attempts to distinguish differences caused by the routing method from differences caused purely by random traffic conditions.

---

# 4. Evaluated Routing Methods

Several methods are evaluated.

## Double DQN

The primary learned policy is a masked **Double Deep Q-Network**.

The DQN uses:

- neural Q-value approximation;
- experience replay;
- a target network;
- epsilon-greedy exploration during training;
- Double DQN targets; and
- invalid-action masking.

After training, evaluation uses greedy action selection.

---

## OR-Tools Fixed Planning

`ortools_fixed` represents a classical optimisation strategy in which a route is calculated and then executed without continuously rebuilding the plan as conditions change.

This provides a strong optimisation baseline.

---

## OR-Tools Re-planning

`ortools_replan` allows the optimisation approach to re-solve the remaining routing problem at disruption points.

This investigates whether repeated optimisation improves adaptability when routing conditions change.

---

## OR-Tools Live Re-planning

`ortools_replan_live` performs re-planning using the current live-priced travel matrix.

This represents a more responsive optimisation strategy within the experimental framework.

---

## Nearest-Neighbour

The nearest-neighbour policy provides a simple capacity-aware heuristic.

At each decision point, it selects an appropriate nearby feasible customer.

Under dynamic conditions, this provides a lightweight policy capable of responding to the current routing costs.

---

## Random Policy

The random baseline selects among valid actions without attempting to optimise the route.

Invalid actions remain masked.

The random policy therefore provides a lower performance reference for assessing whether the learned DQN has acquired useful routing behaviour.

---

# 5. DQN Training

Policies are trained across benchmark instances and experimental conditions.

The training pipeline supports multiple random seeds.

A typical sweep can be executed using:

```bash
uv run python -m src.experiments.sweep --workers 6 --seeds 5
```

The sweep coordinates repeated training across the configured routing problems.

---

# 6. Why Multiple Seeds Are Used

Deep Reinforcement Learning contains several sources of stochasticity, including:

- neural-network initialisation;
- epsilon-greedy exploration;
- replay-buffer sampling;
- optimisation updates; and
- dynamic environment realisations.

A single training run may therefore provide a misleading estimate of performance.

Using multiple seeds allows the study to examine whether observed behaviour is consistent across repeated runs.

---

# 7. Evaluation Procedure

After training, learned policies are evaluated alongside the optimisation and heuristic baselines.

The evaluation pipeline can be executed using:

```bash
uv run python -m src.experiments.run_all
```

The evaluation records metrics such as:

- total route distance;
- method;
- benchmark instance;
- dynamism condition;
- random seed;
- runtime; and
- comparison with reference solution quality.

Results are collected into structured CSV files for subsequent analysis.

---

# 8. Primary Evaluation Metrics

## Total Route Distance

Total route distance is the principal solution-quality metric.

Lower values indicate more efficient routing.

---

## Optimality Gap

Where a best-known reference solution is available, performance can be expressed relative to that reference.

Conceptually:

```text
Optimality Gap (%) =
((Method Cost - Best-Known Cost) / Best-Known Cost) × 100
```

A smaller gap indicates a solution closer to the benchmark reference.

---

## Runtime

Runtime is recorded to examine computational differences between approaches.

This is particularly important when comparing:

- repeated classical optimisation; and
- neural-network inference after training.

A learned policy may not always achieve the best absolute route distance, but rapid inference may be valuable in applications requiring frequent sequential decisions.

---

## Degradation Under Disruption

The analysis also investigates how routing performance changes when moving from static conditions to dynamic conditions.

Conceptually:

```text
Degradation (%) =
((Dynamic Cost - Static Cost) / Static Cost) × 100
```

This provides a measure of sensitivity to disruption.

---

# 9. Cross-Evaluation

The repository includes cross-evaluation experiments.

Cross-evaluation investigates how learned policies behave when evaluated under conditions that differ from those used during training.

This is particularly useful for examining whether training under dynamic conditions automatically produces more robust routing behaviour.

The implementation is available through:

```text
src/experiments/cross_eval.py
```

and outputs are stored in:

```text
results/cross_eval.csv
```

---

# 10. Statistical Analysis

The experimental evaluation does not rely solely on visual inspection or average route distance.

Statistical analysis is performed using:

- descriptive statistics;
- Wilcoxon signed-rank tests;
- rank-biserial effect sizes; and
- bootstrap confidence intervals.

The analysis can be generated using:

```bash
uv run python -m src.analysis.stats
```

The implementation is located at:

```text
src/analysis/stats.py
```

---

# 11. Wilcoxon Signed-Rank Test

The **Wilcoxon signed-rank test** is used for paired comparisons between routing methods.

This non-parametric test is appropriate for comparing paired benchmark-level results without assuming that the differences follow a normal distribution.

The analysis helps determine whether observed performance differences are sufficiently consistent across benchmark instances to provide statistical evidence of a difference between methods.

---

# 12. Effect Size

Statistical significance alone does not describe the magnitude or direction of a difference.

The analysis therefore also includes **rank-biserial effect sizes**.

Effect-size analysis provides additional information about the practical strength of the observed comparison.

---

# 13. Bootstrap Confidence Intervals

Bootstrap procedures are used to estimate confidence intervals around selected performance differences.

This provides an additional measure of uncertainty around the experimental estimates.

---

# 14. Generated Research Outputs

The experimental pipeline produces several structured outputs.

Important files include:

```text
results/results_master.csv
results/stats_summary.csv
results/wilcoxon.csv
results/degradation.csv
results/cross_eval.csv
results/baselines_static.csv
```

These files provide the underlying numerical evidence used in the analysis.

---

# 15. Visualisation

Research figures are generated from the experimental outputs.

The visualisation pipeline can be executed using:

```bash
uv run python -m src.analysis.plots
```

Important figures include:

```text
figures/fig1_convergence.png
figures/fig2_gap_boxplot.png
figures/fig3_gap_bars.png
figures/fig4_adaptability.png
figures/fig5_runtime.png
figures/fig6_routes.png
```

The figures provide visual evidence for:

- training behaviour;
- optimality gaps;
- comparison between methods;
- adaptability under disruption;
- runtime differences; and
- example routing solutions.

---

# 16. Experimental Reproducibility

Several measures are included to support reproducibility:

- benchmark instances are included in the repository;
- best-known solutions are retained;
- experiment scripts are version controlled;
- random seeds are configurable;
- numerical outputs are retained as CSV files;
- analysis scripts are included;
- figures can be regenerated;
- dependencies are recorded in `pyproject.toml`;
- exact dependency resolution is stored in `uv.lock`; and
- automated tests are included.

Large trained model checkpoints are excluded from the public repository because they can be regenerated using the supplied training pipeline.

---

# 17. Interpretation Strategy

The experimental design intentionally avoids treating the research as a simple competition between DQN and OR-Tools.

Instead, the evaluation considers different characteristics of the methods.

### Classical optimisation

Provides a strong benchmark for absolute routing solution quality.

### Deep Reinforcement Learning

Provides a learned policy capable of producing sequential decisions through neural-network inference after training.

### Heuristics

Provide lightweight reference strategies for understanding whether the DQN has learned meaningful routing behaviour.

The central research interest is therefore the **trade-off between solution quality, computational cost, and behaviour under changing conditions**.

---

# 18. Experimental Scope

The experiments should be interpreted within the scope of the implemented framework.

The evaluation does not claim that the trained DQN universally outperforms classical optimisation.

Instead, it provides controlled empirical evidence about:

- what the DQN learns;
- how it compares with simple heuristics;
- how close it comes to a strong optimisation baseline;
- how disruption affects the evaluated strategies; and
- what limitations remain for future research.

---

## Related Documentation

For the reinforcement-learning and environment design:

**[Research Methodology](methodology.md)**

For the empirical findings:

**[Results and Statistical Analysis](results.md)**

For instructions to recreate the environment and experiments:

**[Reproducibility Guide](reproducibility.md)**

For research limitations and possible extensions:

**[Limitations and Future Research](limitations.md)**

Return to:

**[Project README](../README.md)**