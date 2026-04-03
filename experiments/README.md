# Experiments

Each subfolder is a trained model experiment. Folders follow the naming convention:

```
best_model_<dataset>_<layers>L_<dropout>D/
```

For example, `best_model_125k_6L_0.087D` = trained on 125k curves, 6 CNN layers, 0.087 dropout.

## Folder Structure

Each experiment folder is initialized with `pyreflect init` and contains:

```
<experiment>/
  settings.yml          ← training config (layers, dropout, batch_size, epochs, data paths)
  data/
    curves/
      nr_train.npy      ← NR training curves
      sld_train.npy     ← SLD training curves
    normalization_stat.npy
  trained_model.pth     ← saved model weights (after training)
```

## Setting Up an Experiment

Before training, you need an experiment directory with generated training data.

### Option A — Generate data with pyreflect (recommended)

```bash
mkdir experiments/my_experiment && cd experiments/my_experiment
pyreflect init                          # creates settings.yml
# Edit settings.yml: set layers, dropout, batch_size, epochs, num_curves
#   under nr_predict_sld.models
pyreflect run --enable-sld-prediction   # generates .npy data files (also trains
                                        # a local model — you can ignore that)
```

### Option B — Copy existing data

If you already have `.npy` files from a notebook or previous run:

```bash
mkdir -p experiments/my_experiment/data/curves
cp <your_data>/nr_train.npy  experiments/my_experiment/data/curves/
cp <your_data>/sld_train.npy experiments/my_experiment/data/curves/
cp <your_data>/normalization_stat.npy experiments/my_experiment/data/
pyreflect init --root experiments/my_experiment
# Edit experiments/my_experiment/settings.yml with your hyperparameters
```

## Local Training

Once data is in place, edit `settings.yml` with your desired hyperparameters, then:

```bash
cd <experiment-dir>
pyreflect run --enable-sld-prediction
```

This uses pyreflect's built-in training pipeline.

## Modal GPU Training

To train on a cloud GPU instead of locally:

```bash
python experiments/train_modal.py --experiment-dir experiments/my_experiment
python experiments/train_modal.py --experiment-dir experiments/my_experiment --use-wandb
```

Reads `settings.yml` from the experiment dir, uploads `data/` to Modal, trains on a T4 GPU,
and saves the resulting `trained_model.pth` back locally.

**Prerequisites:**
```bash
pip install modal && modal setup   # requires Modal account
pip install wandb                  # only for --use-wandb
```

For `--use-wandb`: create a Modal secret named `wandb-secret` containing `WANDB_API_KEY`.

## Experiments

| Folder | Dataset | Layers | Dropout | Batch | Epochs | Notes |
|--------|---------|--------|---------|-------|--------|-------|
| `best_model_30k_6L_0.107D` | 30k curves | 6 | 0.107 | 64 | 20 | Best params from 30k Optuna search |
| `best_model_125k_6L_0.087D` | 125k curves | 6 | 0.087 | 64 | 20 | Best params from 125k Optuna search |
