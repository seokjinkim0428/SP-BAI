# SP-BAI

Reference implementation for the experiments in the paper "Nearly Optimal Best Arm Identification for Semiparametric Bandits."

## Paper

Nearly Optimal Best Arm Identification for Semiparametric Bandits

Seok-Jin Kim

Accepted at AISTATS 2026.

This repository contains code for fixed-confidence best-arm identification in semiparametric bandits, together with the baselines used in the paper:

- `SP-BAI` (ours)
- `SBE` and `G-Opt/DEO` from prior semiparametric BAI work
- `RAGE` for the synthetic transductive linear-bandit comparison
- `LUCB` and `AE` for the fixed-confidence MAB comparison on Jester

## Repository Layout

```text
sp_bai/
  algorithms/    public algorithm entrypoints
  core/          design solvers, metrics, phases, rounding, utilities
  experiments/   experiment runners used by the notebooks
scripts/         small data-preparation utilities

SP-BAI-Small-gap.ipynb
SP-BAI-Uniform.ipynb
SP-BAI-Real-data.ipynb
```

The notebooks are thin wrappers around the shared `sp_bai` package. The implementation logic is intentionally centralized in the package so that the synthetic and real-data experiments use the same algorithm code paths.

## Installation

We recommend Python 3.9+.

```bash
pip install -r requirements.txt
```

For notebook use, install Jupyter if it is not already available in your environment.

## Data

The synthetic notebooks are self-contained.

The Jester experiments rely on the Jester joke-rating dataset. The original source is the UC Berkeley Jester archive:

- https://eigentaste.berkeley.edu/dataset/archive/

For convenience, a CSV-formatted Kaggle mirror is also available here:

- https://www.kaggle.com/datasets/vikashrajluhaniwal/jester-17m-jokes-ratings-dataset

The Kaggle page may be convenient for downloading CSV-formatted files, but the UC Berkeley archive remains the canonical source. For a public GitHub release, the safer practice is to avoid redistributing the raw Jester files directly in the repository and instead provide download instructions.

This repository now supports the following workflow:

1. Place the raw `jester_ratings.csv` file in the repository root.
2. Either run the preparation script manually, or let the real-data notebook generate the subset automatically.

Manual preparation:

```bash
python scripts/prepare_jester_subset.py --ratings jester_ratings.csv --output jester_subset_50699_8.csv
```

The prepared file `jester_subset_50699_8.csv` is the complete-rating subset over joke IDs `[7, 8, 13, 15, 16, 17, 18, 19]`. It contains the users who rated all selected jokes.

## Running The Experiments

### Synthetic small-gap benchmark

Open and run:

- `SP-BAI-Small-gap.ipynb`

This notebook reproduces the small-gap synthetic experiments from the paper.

### Synthetic uniform-feature benchmark

Open and run:

- `SP-BAI-Uniform.ipynb`

This notebook runs the uniform-feature synthetic benchmark.

### Jester real-data benchmark

Open and run:

- `SP-BAI-Real-data.ipynb`

If `jester_subset_50699_8.csv` is missing but `jester_ratings.csv` is available, the notebook will automatically generate the subset before loading the real-data benchmark.

This notebook contains:

- the G-Opt vs. uniform ranking comparison
- fixed-confidence BAI on Jester
- optional synthetic semiparametric stress tests used for additional analysis

## Package Entry Points

The most useful public entry points are exported from `sp_bai.experiments`:

- `run_small_gap_instance`
- `run_uniform_feature_experiment`
- `run_fixed_confidence_benchmark`
- `run_jester_toy_ranking`
- `run_ours_only_benchmark`

The shared algorithm implementations are in `sp_bai.algorithms`:

- `run_sp_bai`
- `run_sbe`
- `run_g_opt`
- `run_rage`
- `run_lucb`
- `run_ae`

## Reproducibility Notes

- The experiment runners use explicit random seeds.
- For local and Colab consistency, use the same package versions listed in `requirements.txt`.
- After changing any file inside `sp_bai/`, restart the notebook kernel or Colab runtime before rerunning experiments.
- For the Jester benchmark, the recommended public-release workflow is to keep raw data out of the repository and regenerate `jester_subset_50699_8.csv` from `jester_ratings.csv`.
- Wall-clock timing columns are machine-dependent and are not expected to match exactly across environments.

## Notes On Baselines

- `SP-BAI` is our method from "Nearly Optimal Best Arm Identification for Semiparametric Bandits" (accepted at AISTATS 2026).
- `SBE` and `G-Opt/DEO` are semiparametric bandit baselines from Kim, S.-J., Kim, G.-S., and Oh, M.-h. (2025). "Experimental Design for Semiparametric Bandits." In *Proceedings of the Thirty Eighth Conference on Learning Theory*, volume 291 of *Proceedings of Machine Learning Research*, pages 3215-3252.
- `RAGE` is the transductive linear bandit baseline from Fiez, T., Jain, L., Jamieson, K. G., and Ratliff, L. (2019). "Sequential Experimental Design for Transductive Linear Bandits." In *Advances in Neural Information Processing Systems*, volume 32.
- `LUCB` and `AE` are fixed-confidence multi-armed bandit baselines from Jamieson, K. and Nowak, R. (2014). "Best-Arm Identification Algorithms for Multi-Armed Bandits in the Fixed Confidence Setting." In *2014 48th Annual Conference on Information Sciences and Systems (CISS)*, pages 1-6.
