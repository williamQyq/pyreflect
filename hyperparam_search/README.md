# Hyperparameter Search

Uses [Optuna](https://optuna.org/) to search for the best CNN hyperparameters
(layers, dropout, batch size) for the pyreflect NR→SLD model. Minimizes
validation loss across trials.

**W&B and Modal are both optional.** The script runs locally with no external
services by default.

## Quick Start

```bash
# 1. Edit config.yml to point at your dataset and set your search space
# 2. Run:

python hyperparam_search.py --config config.yml                           # local, no logging
python hyperparam_search.py --config config.yml --use-wandb               # local + W&B
python hyperparam_search.py --config config.yml --use-modal               # Modal GPU
python hyperparam_search.py --config config.yml --use-modal --use-wandb   # Modal GPU + W&B
```

## Configuration (`config.yml`)

| Field | Description |
|-------|-------------|
| `data_dir` | Path to dataset folder (must follow `pyreflect init` structure) |
| `wandb_project` | W&B project name (used only with `--use-wandb`) |
| `n_trials` | Number of Optuna trials |
| `epochs_per_trial` | Training epochs per trial (keep low for fast search) |
| `search_space.layers` | List of layer counts to try (categorical) |
| `search_space.dropout_min/max` | Dropout sampled uniformly from this range |
| `search_space.batch_size` | List of batch sizes to try (categorical) |

The dataset folder must contain:
```
data/curves/nr_train.npy
data/curves/sld_train.npy
data/normalization_stat.npy
```

## Output

- Best parameters printed to stdout
- Saved to `<data_dir>/optuna_best_params.txt`
- Optuna study persisted to `<data_dir>/optuna_study.db` (SQLite) — reruns resume where they left off

## Prerequisites

```bash
pip install optuna pyyaml          # always required
pip install wandb                  # only for --use-wandb
pip install modal && modal setup   # only for --use-modal
```

For `--use-modal --use-wandb`: create a Modal secret named `wandb-secret`
containing `WANDB_API_KEY` (see [Modal secrets docs](https://modal.com/docs/guide/secrets)).
