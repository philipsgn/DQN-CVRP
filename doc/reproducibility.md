# Reproducibility Guide

## Overview

This document explains how to set up, test, and reproduce the experiments for the **Deep Reinforcement Learning for Dynamic Route Optimisation** project.

The repository contains the source code, benchmark instances, experiment scripts, statistical analysis, result tables, figures, dependency configuration, and automated tests required to inspect and reproduce the research workflow.

The project uses **`uv`** for Python dependency and environment management.

---

# 1. Requirements

The recommended environment is:

- Python 3.11
- Git
- `uv`
- Windows, Linux, or macOS

Major Python dependencies include:

- PyTorch
- Gymnasium
- NumPy
- Pandas
- SciPy
- Matplotlib
- Google OR-Tools
- VRPLIB
- Pytest

Exact project dependency information is maintained in:

```text
pyproject.toml
uv.lock
```

---

# 2. Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/philipsgn/DQN-CVRP.git
```

Move into the repository:

```bash
cd DQN-CVRP
```

---

# 3. Install `uv`

If `uv` is not already installed, follow the official `uv` installation instructions for your operating system.

Confirm installation with:

```bash
uv --version
```

---

# 4. Create the Environment

Synchronise the environment using the project configuration:

```bash
uv sync
```

This uses:

```text
pyproject.toml
uv.lock
```

to create a reproducible Python environment.

There is normally no need to manually activate the virtual environment when using `uv`.

Commands can be executed directly using:

```bash
uv run <command>
```

This is particularly convenient on Windows systems where PowerShell execution policies may prevent manual activation of virtual-environment scripts.

---

# 5. Install the Project in Editable Mode

If required for development or testing, install the project in editable mode:

```bash
uv pip install -e .
```

This allows the source package to be imported while changes are made to the local codebase.

---

# 6. Verify the Installation

Run the automated tests:

```bash
uv run pytest
```

The tests cover key components of the implementation, including the routing environment and reinforcement-learning functionality.

The test suite is located under:

```text
tests/
```

including:

```text
tests/test_agents.py
tests/test_dynamic.py
tests/test_env.py
```

A successful test run provides a basic verification that the environment and core implementation are functioning correctly.

---

# 7. Benchmark Data

The project uses established **CVRPLIB** benchmark instances.

The benchmark files are stored under:

```text
data/instances/
```

The repository contains both:

```text
.vrp
```

problem files and corresponding:

```text
.sol
```

best-known solution files where applicable.

These allow experimental route costs to be compared against established reference solutions.

---

# 8. Run a Single Training Experiment

A single DQN training experiment can be executed using the training module.

Example:

```bash
uv run python -m src.experiments.train --instance E-n22-k4 --delta static --seed 0
```

The important parameters include:

```text
--instance
--delta
--seed
```

where `delta` controls the experimental disruption condition.

Typical conditions include:

```text
static
low
high
```

---

# 9. Run the Training Sweep

The repository includes a sweep utility for running repeated experiments across benchmark instances and random seeds.

Example:

```bash
uv run python -m src.experiments.sweep --workers 6 --seeds 5
```

This coordinates multiple training runs across the configured experimental conditions.

The number of workers can be adjusted according to the available hardware.

For systems with fewer CPU resources, use a smaller value.

For example:

```bash
uv run python -m src.experiments.sweep --workers 2 --seeds 5
```

---

# 10. Evaluate the Routing Methods

After training, the main evaluation pipeline can be executed with:

```bash
uv run python -m src.experiments.run_all
```

This evaluates the trained policies and comparison methods across the configured benchmark scenarios.

Methods represented in the experimental framework include:

- Double DQN;
- OR-Tools fixed planning;
- OR-Tools re-planning;
- OR-Tools live re-planning;
- nearest-neighbour; and
- random routing.

---

# 11. Cross-Evaluation

Cross-evaluation is used to investigate how learned policies behave under conditions different from their training environment.

The implementation is located at:

```text
src/experiments/cross_eval.py
```

The resulting output is stored in:

```text
results/cross_eval.csv
```

Cross-evaluation is particularly useful for investigating robustness under changes in disruption level.

---

# 12. Generate Statistical Analysis

After the experimental results have been generated, run:

```bash
uv run python -m src.analysis.stats
```

The statistical pipeline includes:

- descriptive statistics;
- Wilcoxon signed-rank tests;
- rank-biserial effect sizes;
- bootstrap confidence intervals; and
- disruption degradation analysis.

Generated outputs include files such as:

```text
results/stats_summary.csv
results/wilcoxon.csv
results/degradation.csv
```

---

# 13. Generate Figures

Research visualisations can be regenerated using:

```bash
uv run python -m src.analysis.plots
```

The generated figures are stored under:

```text
figures/
```

Important outputs include:

```text
fig1_convergence.png
fig2_gap_boxplot.png
fig3_gap_bars.png
fig4_adaptability.png
fig5_runtime.png
fig6_routes.png
```

These figures correspond to the visual evidence presented in the project documentation.

---

# 14. Research Workflow

The complete research workflow can be summarised as:

```text
CVRPLIB Benchmark Instances
            │
            ▼
    Environment Setup
            │
            ▼
      DQN Training
            │
            ▼
   Baseline Evaluation
            │
            ▼
  Dynamic Evaluation
            │
            ▼
     Cross-Evaluation
            │
            ▼
   Statistical Analysis
            │
            ▼
     Visualisation
            │
            ▼
 Research Interpretation
```

This separation between training, evaluation, statistical analysis, and visualisation makes the experimental pipeline easier to inspect and reproduce.

---

# 15. Random Seeds

Random seeds are used throughout the experiments because reinforcement-learning results can vary due to:

- neural-network initialisation;
- exploration;
- replay-buffer sampling;
- environment stochasticity; and
- dynamic travel-cost generation.

Using multiple seeds reduces dependence on a single favourable training run.

Where comparisons involve stochastic disruption, seeded evaluation also helps ensure that competing approaches experience comparable conditions.

---

# 16. Reproducing Published Results

The repository already contains the numerical outputs generated during the research.

Important files include:

```text
results/results_master.csv
results/stats_summary.csv
results/wilcoxon.csv
results/degradation.csv
results/cross_eval.csv
results/baselines_static.csv
```

These files allow the reported findings to be inspected without retraining every neural-network model.

This is useful because full reinforcement-learning training can require considerably more computational time than statistical analysis or figure generation.

To reproduce the analysis from existing results, the principal commands are:

```bash
uv run python -m src.analysis.stats
```

followed by:

```bash
uv run python -m src.analysis.plots
```

---

# 17. Trained Model Checkpoints

Large trained model checkpoint files are intentionally excluded from the public Git repository.

Typical checkpoint extensions such as:

```text
.pt
.pth
.ckpt
```

are excluded through `.gitignore`.

This keeps the repository lightweight and avoids storing large regenerable binary artefacts in normal Git history.

The trained models can be recreated using the provided training scripts.

---

# 18. GitHub Actions

The repository includes an automated GitHub Actions workflow under:

```text
.github/workflows/
```

The purpose of continuous integration is to automatically verify important aspects of the codebase when changes are pushed to GitHub.

Local testing should still be performed before publishing changes:

```bash
uv run pytest
```

---

# 19. Inspecting the Results Without Retraining

A researcher or supervisor who does not want to reproduce the complete training process can still inspect the research evidence directly.

Start with:

```text
results/
```

for numerical outputs.

Then inspect:

```text
figures/
```

for visualisations.

For interpretation of the findings, see:

```text
docs/results.md
```

This allows the repository to support both:

- quick research inspection; and
- deeper computational reproduction.

---

# 20. Recommended Reproduction Levels

The repository supports several levels of inspection.

## Level 1 — Inspect Existing Evidence

No model training required.

Review:

```text
README.md
docs/results.md
results/
figures/
```

This is the fastest way to understand the project.

---

## Level 2 — Verify the Software

Install the environment:

```bash
uv sync
```

Run:

```bash
uv run pytest
```

This verifies the core implementation.

---

## Level 3 — Reproduce Analysis

Using the existing experimental results:

```bash
uv run python -m src.analysis.stats
uv run python -m src.analysis.plots
```

This regenerates statistical analysis and visualisations.

---

## Level 4 — Reproduce Training

Run individual training experiments or the complete sweep.

Example:

```bash
uv run python -m src.experiments.train --instance E-n22-k4 --delta static --seed 0
```

or:

```bash
uv run python -m src.experiments.sweep --workers 6 --seeds 5
```

This is the most computationally expensive level of reproduction.

---

# 21. Reproducibility Notes

Exact numerical reproduction of reinforcement-learning experiments may depend on:

- operating system;
- CPU/GPU hardware;
- PyTorch backend;
- dependency versions;
- parallel execution; and
- sources of nondeterminism in numerical computation.

The repository therefore combines:

- locked dependencies;
- explicit random seeds;
- retained numerical results;
- benchmark data;
- source code;
- automated tests; and
- statistical outputs

to provide a practical and transparent reproducibility framework.

---

# 22. Troubleshooting

## `ModuleNotFoundError: No module named 'src'`

Install the project in editable mode:

```bash
uv pip install -e .
```

Then run:

```bash
uv run pytest
```

---

## PowerShell prevents `.venv` activation

Manual activation is not required.

Instead of:

```powershell
.venv\Scripts\activate
```

use:

```bash
uv run pytest
```

or:

```bash
uv run python -m <module>
```

This allows `uv` to execute commands using the project environment directly.

---

## Dependency Problems

Synchronise the environment again:

```bash
uv sync
```

Then verify:

```bash
uv run pytest
```

---

# 23. Reproducibility Philosophy

The purpose of this repository is not simply to publish source code.

It is intended to preserve the relationship between:

```text
Research Question
       ↓
Implementation
       ↓
Experiments
       ↓
Raw Results
       ↓
Statistical Analysis
       ↓
Figures
       ↓
Research Conclusions
```

Providing this chain of evidence makes the project easier to inspect, critique, reproduce, and extend.

---

## Related Documentation

For the research approach:

**[Research Methodology](methodology.md)**

For the benchmark and experimental setup:

**[Experimental Design](experiments.md)**

For the empirical findings:

**[Results and Statistical Analysis](results.md)**

For research limitations and future extensions:

**[Limitations and Future Research](limitations.md)**

Return to:

**[Project README](../README.md)**