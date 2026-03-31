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

## Local Training

Edit `settings.yml` with your desired hyperparameters, then:

```bash
cd <experiment-dir>
pyreflect run
```

This uses pyreflect's built-in training pipeline.

## Modal GPU Training

To train on a cloud GPU instead of locally:

```bash
python experiments/train_modal.py --experiment-dir experiments/best_model_125k_6L_0.087D
python experiments/train_modal.py --experiment-dir experiments/best_model_125k_6L_0.087D --use-wandb
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
