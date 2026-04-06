"""
Optuna Hyperparameter Search for the pyreflect NR->SLD CNN model.

Searches over CNN hyperparameters (layers, dropout, batch size) to minimize
validation loss, with optional W&B logging and optional Modal GPU execution.

Usage
-----
Run locally (no external services):
    python hyperparam_search.py --config config.yml

Run locally with W&B logging:
    python hyperparam_search.py --config config.yml --use-wandb

Run on Modal GPU (no W&B):
    python hyperparam_search.py --config config.yml --use-modal

Run on Modal GPU with W&B logging:
    python hyperparam_search.py --config config.yml --use-modal --use-wandb

Prerequisites
-------------
- Core:   pip install optuna pyyaml
- W&B:    pip install wandb   (only if using --use-wandb)
- Modal:  pip install modal   (only if using --use-modal)
  - Also requires a Modal account and `modal setup`
  - If using --use-modal --use-wandb, create a Modal secret named "wandb-secret"
    containing WANDB_API_KEY (see https://modal.com/docs/guide/secrets)

Output
------
Best parameters are printed to stdout and saved to:
    <data_dir>/optuna_best_params.txt

The Optuna study is persisted to <data_dir>/optuna_study.db (SQLite) so
interrupted runs can be resumed.
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate the YAML search config."""
    path = Path(config_path)
    if not path.exists():
        sys.exit(f"Error: config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    required = ["data_dir", "n_trials", "epochs_per_trial", "search_space"]
    missing = [k for k in required if k not in config]
    if missing:
        sys.exit(f"Error: config is missing required keys: {missing}")

    ss = config["search_space"]
    ss_required = ["layers", "dropout_min", "dropout_max", "batch_size"]
    missing_ss = [k for k in ss_required if k not in ss]
    if missing_ss:
        sys.exit(f"Error: search_space is missing required keys: {missing_ss}")

    return config


# ---------------------------------------------------------------------------
# Core study runner
# All imports live inside this function so Modal can serialize and run it
# on a remote GPU without needing local modules to be importable there.
# ---------------------------------------------------------------------------

def _run_study(config_dict: dict, use_wandb: bool) -> dict:
    """
    Run the full Optuna study. Designed to work both locally and on Modal GPU.

    All heavy imports are deferred to inside this function so that Modal can
    serialize and ship it to the remote environment cleanly.

    Parameters
    ----------
    config_dict : dict
        Full config (same structure as config.yml). data_dir should already
        be set to the correct path for the environment (local path or
        /root/data on Modal).
    use_wandb : bool
        Whether to log per-epoch metrics to Weights & Biases.

    Returns
    -------
    dict with keys: success (bool), best_params (dict), best_val_loss (float)
    """
    import optuna
    import torch
    import torch.nn as nn
    from pathlib import Path
    from pyreflect.input import NRSLDDataProcessor, DataProcessor
    from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
    import pyreflect.pipelines.reflectivity_pipeline as workflow

    # Adam optimizer constants (from pyreflect's NRSLDModelTrainer)
    LEARNING_RATE = 2.15481e-05
    WEIGHT_DECAY = 2.6324e-05
    TRAIN_SPLIT = 0.8

    data_root = Path(config_dict["data_dir"])
    nr_file = str(data_root / "data/curves/nr_train.npy")
    sld_file = str(data_root / "data/curves/sld_train.npy")
    norm_file = str(data_root / "data/normalization_stat.npy")

    search_space = config_dict["search_space"]
    n_trials = config_dict["n_trials"]
    epochs_per_trial = config_dict["epochs_per_trial"]
    wandb_project = config_dict.get("wandb_project", "pyreflect-optuna")

    # Load and preprocess data once (shared across all trials for efficiency)
    print("Loading training data...")
    dproc = NRSLDDataProcessor(nr_file, sld_file).load_data()
    X_all, y_all = workflow.preprocess(dproc, norm_file)
    print(f"Data loaded: {X_all.shape[0]} curves\n")

    def train_with_epoch_logging(X, y, layers, dropout, batch_size, epochs, wandb_run):
        """
        Train the CNN for a fixed number of epochs, logging each epoch.

        Returns
        -------
        (train_losses, val_losses) : tuple of lists
        """
        trainer = NRSLDModelTrainer(
            X=X, y=y,
            layers=layers,
            batch_size=batch_size,
            epochs=epochs,
            dropout=dropout,
        )

        list_arrays = DataProcessor.split_arrays(X, y, size_split=TRAIN_SPLIT)
        tensor_arrays = DataProcessor.convert_tensors(list_arrays)
        _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
            *tensor_arrays, batch_size=batch_size
        )

        optimizer = torch.optim.Adam(
            trainer.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        loss_fn = nn.MSELoss()

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            train_loss = trainer.train_model(trainer.model, train_loader, optimizer, loss_fn)
            val_loss = trainer.validate_model(trainer.model, valid_loader, loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "overfitting_gap": val_loss - train_loss,
                })

            print(
                f"  Epoch {epoch + 1}/{epochs} — "
                f"Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            )

        return train_losses, val_losses

    def objective(trial):
        """Optuna objective: suggest hyperparams, train, return validation loss."""
        layers = trial.suggest_categorical("layers", search_space["layers"])
        dropout = trial.suggest_float(
            "dropout", search_space["dropout_min"], search_space["dropout_max"]
        )
        batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])

        print(f"\n{'='*60}")
        print(f"Trial {trial.number + 1}/{n_trials}")
        print(f"  layers={layers}, dropout={dropout:.4f}, batch_size={batch_size}")
        print(f"{'='*60}")

        wandb_run = None
        if use_wandb:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                name=f"trial_{trial.number}_L{layers}_D{dropout:.2f}_B{batch_size}",
                config={
                    "layers": layers,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "epochs": epochs_per_trial,
                    "trial_number": trial.number,
                },
                reinit=True,
            )

        try:
            train_losses, val_losses = train_with_epoch_logging(
                X_all, y_all, layers, dropout, batch_size, epochs_per_trial, wandb_run
            )

            final_val_loss = val_losses[-1]
            overfitting_gap = val_losses[-1] - train_losses[-1]

            print(f"\nTrial {trial.number + 1} complete:")
            print(f"  Final val loss:     {final_val_loss:.6f}")
            print(f"  Overfitting gap:    {overfitting_gap:.6f}")

            if wandb_run is not None:
                import wandb
                wandb.log({
                    "final_val_loss": final_val_loss,
                    "final_train_loss": train_losses[-1],
                    "overfitting_gap": overfitting_gap,
                })
                wandb.finish()

            return final_val_loss

        except Exception as e:
            print(f"Trial {trial.number + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            if wandb_run is not None:
                import wandb
                wandb.finish()
            raise optuna.TrialPruned()

    # Persist study to SQLite so interrupted runs can be resumed
    db_path = data_root / "optuna_study.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="pyreflect_optuna",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    print(f"Starting Optuna search: {n_trials} trials, {epochs_per_trial} epochs each\n")
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No trials completed successfully.")
        return {"success": False, "best_params": {}, "best_val_loss": float("inf")}

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial:     #{study.best_trial.number + 1}")
    print(f"Best val loss:  {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return {
        "success": True,
        "best_params": study.best_params,
        "best_val_loss": study.best_value,
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def _save_results(result: dict, data_dir: str) -> None:
    """Write best params to a text file in data_dir."""
    if not result["success"]:
        return

    out_path = Path(data_dir) / "optuna_best_params.txt"
    with open(out_path, "w") as f:
        f.write(f"Best validation loss: {result['best_val_loss']:.6f}\n")
        f.write("Best parameters:\n")
        for k, v in result["best_params"].items():
            f.write(f"  {k}: {v}\n")

    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------

def run_local(config: dict, use_wandb: bool) -> None:
    """Run the Optuna study on the local machine."""
    result = _run_study(config, use_wandb)
    _save_results(result, config["data_dir"])


# ---------------------------------------------------------------------------
# Modal runner
# ---------------------------------------------------------------------------

def run_modal(config: dict, config_path: str, use_wandb: bool) -> None:
    """
    Run the Optuna study on a Modal GPU.

    Modal requires functions to be decorated at module level — they cannot be
    created dynamically at runtime. To work around this while keeping a unified
    script, this function generates a minimal self-contained Modal script with
    the study runner properly defined at module level, then invokes it via
    `modal run`. The temp script is deleted after the run completes.

    Parameters
    ----------
    config : dict
        Loaded config dict.
    config_path : str
        Path to the YAML config file (unused by the temp script, kept for
        the local results path).
    use_wandb : bool
        Whether to enable W&B logging on the remote.
    """
    data_dir = Path(config["data_dir"]).resolve()
    if not data_dir.exists():
        sys.exit(f"Error: data_dir does not exist: {data_dir}")

    modal_config = {**config, "data_dir": "/root/data"}
    out_path = str(data_dir / "optuna_best_params.txt")
    wandb_pkg = '"wandb",' if use_wandb else ""
    secret_expr = '[modal.Secret.from_name("wandb-secret")]' if use_wandb else "[]"

    # Embed _run_study verbatim — it has all its imports inside so Modal can
    # serialize and ship it to the remote container without local dependencies.
    run_study_src = inspect.getsource(_run_study)

    # Build result-saving footer as plain strings to avoid f-string escaping
    # conflicts with the curly braces already present in run_study_src.
    footer_lines = [
        "",
        '@app.function(gpu="T4", image=image, secrets=secrets, timeout=12*60*60)',
        "def _remote():",
        "    return _run_study(MODAL_CONFIG, USE_WANDB)",
        "",
        "@app.local_entrypoint()",
        "def main():",
        "    result = _remote.remote()",
        "    if not result['success']:",
        "        print('No trials completed.')",
        "        return",
        "    print('Best params: ' + str(result['best_params']))",
        "    print('Best val loss: ' + str(result['best_val_loss']))",
        "    with open(" + repr(out_path) + ", 'w') as f:",
        "        f.write('Best validation loss: ' + f\"{result['best_val_loss']:.6f}\" + '\\n')",
        "        f.write('Best parameters:\\n')",
        "        for k, v in result['best_params'].items():",
        "            f.write('  ' + str(k) + ': ' + str(v) + '\\n')",
        "    print('Results saved to: ' + " + repr(out_path) + ")",
    ]

    script = "\n".join([
        "import modal",
        "",
        'app = modal.App("pyreflect-optuna")',
        "image = (",
        '    modal.Image.debian_slim(python_version="3.10")',
        "    .pip_install(",
        '        "torch==2.5.1", "numpy==2.1.0", "optuna",',
        '        "pandas", "scikit-learn", "scipy", "opencv-python",',
        '        "pyyaml", "tqdm", "refnx", "llvmlite", "numba",',
        f"        {wandb_pkg}",
        "    )",
        '    .apt_install("git")',
        '    .run_commands("pip install git+https://github.com/williamQyq/pyreflect.git")',
        f"    .add_local_dir({repr(str(data_dir))}, remote_path='/root/data', copy=True)",
        ")",
        f"secrets = {secret_expr}",
        f"MODAL_CONFIG = {json.dumps(modal_config)}",
        f"USE_WANDB = {use_wandb}",
        "",
    ]) + run_study_src + "\n".join(footer_lines)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(script)
        tmp.close()

        print(f"Launching Optuna search on Modal GPU (T4)...")
        print(f"Data: {data_dir}")
        print(f"Trials: {config['n_trials']}, Epochs/trial: {config['epochs_per_trial']}\n")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        subprocess.run(["modal", "run", tmp.name], check=True, env=env)
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the pyreflect NR->SLD CNN model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python hyperparam_search.py --config config.yml\n"
            "  python hyperparam_search.py --config config.yml --use-wandb\n"
            "  python hyperparam_search.py --config config.yml --use-modal\n"
            "  python hyperparam_search.py --config config.yml --use-modal --use-wandb\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML search config (e.g. config.yml)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log per-epoch metrics to Weights & Biases (requires wandb installed and logged in)",
    )
    parser.add_argument(
        "--use-modal",
        action="store_true",
        help="Run on Modal GPU instead of locally (requires modal installed and configured)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_modal:
        run_modal(config, config_path=args.config, use_wandb=args.use_wandb)
    else:
        run_local(config, use_wandb=args.use_wandb)



if __name__ == "__main__":
    main()
