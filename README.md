# Multi-Variable Conformal Prediction

This repository contains the reference implementation for the paper:

> Laura Lützow, Simone Garatti, Marco C. Campi, Lars Lindemann, Matthias Althoff.
> **Multi-Variable Conformal Prediction: Optimizing Prediction Sets using Multiple Calibration Variables.**
> Advances in Neural Information Processing Systems (NeurIPS), 2026.

It provides three families of conformal predictors that construct prediction sets with a coverage guarantee: either the expected guarantee E[η] ≥ 1 − α (exchangeable data) or the high-confidence guarantee P(η ≥ 1 − ε) ≥ 1 − β (i.i.d. data), where η is the coverage of the calibrated set.

| Class | Description |
|-------|-------------|
| **SCP** | Split Conformal Prediction — closed-form quantile calibration with a scalar nonconformity score |
| **RemMCP** | MCP with Constraint Removal — solves a convex scenario program to fit a minimal ellipsoidal set |
| **H-RelMCP** | Heuristic MCP with Constraint Relaxation — adaptive penalty-based outer loop (Alg. 1) for non-convex (union-of-ellipsoids) sets; high-confidence guarantee only, nominal (not covered by Thm. 2) |

SCP baselines with a data split (SCP-2 to SCP-5) reserve part of the calibration data to fit the set shape before applying the standard quantile step: a single ellipsoid (suffix A), a union of ellipsoids via k-means (B), or minimum-volume ellipsoids via CMA-ES (C). The paper reports SCP-1, SCP-2A/B, SCP-3A/B, RemMCP, and H-RelMCP-A/B; the remaining variants are available for additional comparisons. All methods inherit from the same `CP` abstract base class.

---

## Repository structure

```
Multi-Variable-Conformal-Prediction/
├── pyproject.toml             # Package metadata, dependencies, pytest config
├── MCP/
│   ├── CP.py                  # Abstract base class (coverage guarantee logic)
│   ├── SCP.py                 # Split Conformal Prediction
│   ├── RemMCP.py              # MCP with Constraint Removal (convex)
│   ├── HRelMCP.py             # Heuristic MCP with Constraint Relaxation (non-convex)
│   ├── shapes.py              # Ellipsoid parameterisation, score/cost functions, shape families
│   ├── fitting.py             # k-means / CMA-ES shape fits for the split-SCP baselines
│   ├── methods.py             # Registry of compared methods, coverage targets, feasibility checks
│   ├── results.py             # RunResult / MethodResult containers
│   ├── experiment.py          # Experiment runner and command-line interface
│   ├── reporting/             # Console tables, pickles, TikZ figures/tables, PNG figures
│   │   ├── axes.py             # Axis ranges of the coverage and volume plots
│   │   ├── labels.py           # Target labels and certified misconfidence
│   │   ├── regenerate.py       # Figures and tables from saved results
│   │   └── beta_distribution.py # Coverage distributions of SCP and RemMCP (Fig. 1, 4, 14)
│   └── data/
│       ├── car_env.py          # Vehicle dynamics simulator [1]
│       ├── data_generation.py  # Synthetic trajectory dataset generator [1]
│       ├── predictor.py        # Kinematic trajectory predictor [1]
│       ├── regression.py       # CASP / SCM20D loading and neural-network predictors
│       ├── regression_CASP.csv # CASP protein-structure dataset [2] (download, see below)
│       └── regression_scm20d.csv # SCM20D supply-chain dataset [3] (download, see below)
├── scripts/
│   └── run_paper_experiments.sh # Runs all experiments of the paper in parallel and combines the sweeps
├── tests/                     # pytest suite
├── tikz_figures/              # TikZ figures and tables (generated, not tracked)
├── results/                   # Pickled results and log files (generated, not tracked)
└── figures/                   # PNG figures (generated, not tracked)
```

---

## Installation

**Requirements:** Python 3.10 or later.

### Using pip and virtualenv (recommended)

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install the package and its dependencies (editable, with test extras)
pip install -e ".[test]"
```

### Using conda

```bash
conda create -n mcp python=3.11
conda activate mcp
pip install -e ".[test]"
```

The key packages are:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations throughout |
| `scipy` | Constrained optimisation (`RemMCP`, `H-RelMCP`) and binomial bounds (`CP`) |
| `pandas` | CSV loading for CASP and SCM20D datasets |
| `matplotlib` | Result plots |
| `scikit-learn` | MLP regressor (CASP/SCM20D predictor), KMeans (ellipsoid initialisation) |
| `cma` | CMA-ES solver for minimum-volume enclosing ellipsoids (SCP-2C variant) |

---

## Datasets

The vehicle-trajectory data is simulated on the fly; unlike in [1], each rollout follows one of the three driving policies drawn uniformly at random, so that the samples are i.i.d. The two regression datasets are not included in the repository:

- **CASP** [2]: download the data set and save `CASP.csv` (header `RMSD, F1, ..., F9`, 45,730 rows) as `MCP/data/regression_CASP.csv`.
- **SCM20D** [3]: download `scm20d-train.zip`, extract the ARFF file, and save its data section, i.e., all lines after the line `@data` (7,463 rows), as `MCP/data/regression_scm20d.csv`, e.g. with

  ```bash
  awk 'f; tolower($0) ~ /^@data/ {f=1}' scm20d-train.arff > MCP/data/regression_scm20d.csv
  ```

The loader detects whether a CSV file has a header row, and keeps the row order, which together with the fixed seeds determines the training split.

---

## Reproducing the results

All experiments are launched from the command line from the repository root. Without arguments, one quick run per dataset is performed:

```bash
python -m MCP.experiment
```

### Coverage targets

All methods of an experiment share one coverage target:

- **High-confidence guarantee** (`--betas`): all methods target P(η ≥ 1 − ε) ≥ 1 − β. SCP and RemMCP choose the outlier budget ρ = max{ρ′ : β̄(ρ′) ≤ β}, H-RelMCP certifies its solution a posteriori with the confidence-coverage tradeoff, sharing β among all evaluated penalty parameters. Several values of `--epsilons` and `--betas` run every combination.
- **Expected guarantee** (default, without `--betas`): SCP and RemMCP target E[η] ≥ 1 − α with α = ε. H-RelMCP, which only provides the high-confidence guarantee, targets the confidence of SCP-1.

A method that cannot certify the target with the given number of calibration points (no valid outlier budget, or for H-RelMCP not even without violations) is skipped; the log states the number of calibration points it would need. Tables mark such methods as "infeasible", and box plots leave their slot empty.

All method settings are fixed in [MCP/methods.py](MCP/methods.py) and are not tuned on the data. H-RelMCP uses the soft iteration limit i_soft = 100 of Alg. 1 and stops after at most i_max = 200 iterations; runs in which it finds no certified solution are counted in the row "No certified solution" of the console summary and the LaTeX table, separately from unexpected errors ("Errors"). The number of iterations it actually needed is reported in the console summary.

### Experiments of the paper

500 calibration runs each with 10,000 test points; `n_cal = 2000` calibration points (the default, except for SCM20D). Run k uses the seed 100 + k, so that all results are reproducible. All experiments run at once, one process per experiment and sweep value (23 in total, e.g. on a server inside `tmux`); on an AMD EPYC 9554, this takes about 11 hours:

```bash
bash scripts/run_paper_experiments.sh
```

| Experiments | Name prefix | Settings |
|---|---|---|
| Sec. 4: vehicle prediction, T = 2, 30, 50 | `mainHc_` | ε = 0.05, β = 10⁻⁶ |
| App. E.2: misconfidence sweep, vehicle T = 30 | `appHc_` | ε = 0.05, β ∈ {10⁻¹, 10⁻³, 10⁻⁶, 10⁻⁹} |
| App. E.1: miscoverage sweeps, vehicle T = 30 and CASP | `appHc_` | ε ∈ {0.02, 0.05, 0.15}, β = 10⁻⁶ |
| App. E.3: output dimensions, SCM20D, n_y = 2, 4, 6, 8 | `appHc_` | ε = 0.05, β = 10⁻⁶, `n_cal = 2000` |
| App. F.1: expected guarantee, vehicle T = 2, 30, 50 | `appExpect_` | α = ε = 0.05 |
| App. F.2: expected guarantee, miscoverage sweeps, vehicle T = 30 and CASP | `appExpect_` | α = ε ∈ {0.02, 0.05, 0.15} |

Experiment names state the data and the target, e.g. `mainHc_Vehicle30_eps5_beta1e-06` (ε in percent). Each sweep value is a separate experiment; at the end, the script combines each sweep into one experiment with one panel per value (`appHc_Vehicle30_eps5_betasweep`, `appHc_Vehicle30_epssweep_beta1e-06`, `appHc_CASP_epssweep_beta1e-06`, `appHc_SCM20Ddimsweep_eps5_beta1e-06` with the panels n_y = 2, 4, 6, 8, `appExpect_Vehicle30_epssweep`, and `appExpect_CASP_epssweep`). Shared points run only once: the vehicle point ε = 0.05, β = 10⁻⁶ of both high-confidence sweeps is the Sec. 4 experiment, and the vehicle point ε = 0.05 of the expected-guarantee sweep is `appExpect_Vehicle30_eps5`. Single experiments can also be run directly, e.g.

```bash
python -m MCP.experiment --data Vehicle --timesteps 30 --epsilons 0.05 --betas 1e-6 --n-runs 500 --name-prefix mainHc_
```

The script limits each process to one thread, writes start-up errors to `results/run_*.out`, the combination step to `results/run_combine.out`, and the commit and machine to `results/run_info.txt`, and refuses to start if `results/` already contains output. `DRY_RUN=1` only prints the commands. To stop all experiments, run `pkill -f MCP.experiment`.

Run `python -m MCP.experiment --help` for all options, e.g. `--methods` to select methods (default: the paper methods for the vehicle data, the unimodal ones for CASP and SCM20D) and `--output-dir` to write outputs elsewhere.

Per experiment, a log is written to `results/{name}_log.txt`, a compact pickle of all runs to `results/{name}_results.pkl`, PNG figures to `figures/`, and TikZ figures and tables to `tikz_figures/`. The experiment name ends in `_eps5` for ε = 0.05 (`_epssweep` for several values) and, with `--betas`, in `_beta1e-06` (`_betasweep` for several values). The tables report the computation time and the certified misconfidence β̄ of coverage ≥ 1 − ε per method (β̄(ρ) for SCP and RemMCP; mean and maximum of the a-posteriori β over the runs for H-RelMCP). Set volume is the Monte-Carlo estimate of the prediction-set volume (smaller is better while coverage is met). For CASP and SCM20D, the neural-network predictor is trained once per dataset before the calibration runs.

### Figures and tables from saved results

The figures of test coverage (violins with the quartiles, median, and mean inside) and set volume (boxplots), the LaTeX table, and the summary can be regenerated from the saved results without re-running the experiments, e.g. after copying `results/` from a server:

```bash
python -m MCP.reporting.regenerate results/*_results.pkl
```

The outputs go to `figures/` and `tikz_figures/` next to the `results/` directory (or to `--output-dir`); `--methods` selects and orders the methods. Experiments with different targets, e.g. the values of a sweep run in parallel, can be combined into one experiment with one panel per target, in the given order; the combined results are also saved as `results/{NAME}_results.pkl`:

```bash
python -m MCP.reporting.regenerate results/appHc_CASP_eps2_beta1e-06_results.pkl results/appHc_CASP_eps5_beta1e-06_results.pkl results/appHc_CASP_eps15_beta1e-06_results.pkl --combine appHc_CASP_epssweep_beta1e-06
```

Experiments with the same target, such as the SCM20D output dimensions, need one LaTeX panel label per file:

```bash
python -m MCP.reporting.regenerate results/appHc_SCM20Ddim{2,4,6,8}_eps5_beta1e-06_results.pkl --combine appHc_SCM20Ddimsweep_eps5_beta1e-06 --labels '$n_y = 2$' '$n_y = 4$' '$n_y = 6$' '$n_y = 8$'
```

The prediction-set figures need the calibration residuals, which are not saved, and are not regenerated.

### Coverage distributions

The coverage distributions of SCP and RemMCP in Fig. 1, Fig. 4, and Fig. 14, Beta(n − n_q(ρ+1) + 1, n_q(ρ+1)), are written as PGFPlots tables to `tikz_figures/fig_helpers/`:

```bash
python -m MCP.reporting.beta_distribution --epsilon 0.05 --beta 1e-6 --x-min 0.9 --n-points 400
```

This writes the tables of Fig. 4; `--beta 1e-2` writes those of Fig. 1. Without `--beta`, ρ follows the expected guarantee; with the default grid (`--x-min 0.65 --n-points 200`), this reproduces the tables of the expected-guarantee figure (Fig. 14).

To add a method, register a `MethodSpec` in [MCP/methods.py](MCP/methods.py).

---

## Running the tests

```bash
pytest                 # all tests
pytest -m "not slow"   # skip the CMA-ES and union-of-ellipsoids H-RelMCP tests
```

Some tests draw diagnostic plots. During `pytest` they go to a temporary directory; to keep them, run `MCP_TEST_PLOTS=1 pytest` or run a test file directly (`python tests/test_RemMCP.py`). Both write to `tests/plots/` (ignored by git). Toy problems shared by the tests are in `tests/toy_problems.py`.

---

## Development

Install the development tools with `pip install -e ".[dev]"`. Before committing, run:

```bash
ruff format .   # format the code
ruff check .    # lint
mypy            # type check (configured in pyproject.toml)
pytest          # tests
```

CI runs the same checks on every push.

---

## Citation

If you use this code, please cite the paper (see also [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{luetzow2026multivariable,
  title     = {Multi-Variable Conformal Prediction: Optimizing Prediction Sets using Multiple Calibration Variables},
  author    = {L{\"u}tzow, Laura and Garatti, Simone and Campi, Marco C. and Lindemann, Lars and Althoff, Matthias},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

---

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The vehicle benchmark in `MCP/data/` is adapted from [1], also under the MIT License.

---

## References

**[1] Vehicle prediction code** (MIT License, see [MCP/data/LICENSE_conformal_region_designer](MCP/data/LICENSE_conformal_region_designer)) — Renukanandan Tumu, Matthew Cleaveland, Rahul Mangharam, George J. Pappas, and Lars Lindemann. *Multi-modal conformal prediction regions by optimizing convex shape templates.* L4DC 2024. [GitHub](https://github.com/nandantumu/conformal_region_designer)

**[2] CASP dataset** — UCI ML Repository: [Physicochemical Properties of Protein Tertiary Structure](https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure)

**[3] SCM20D dataset** — MULAN: [multi-target regression datasets](https://sourceforge.net/projects/mulan/files/datasets/multi-target%20regression%20datasets/scm20d-train.zip/download)
