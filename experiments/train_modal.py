"""
Train a pyreflect NR->SLD CNN model on a Modal GPU.

Reads training parameters from an experiment directory's settings.yml
(the config file created by `pyreflect init`) and trains on a Modal T4 GPU.
The trained model is saved back to the experiment directory.

For experiment setup (generating data, directory structure, prerequisites),
see experiments/README.md.

Usage
-----
    python experiments/train_modal.py --experiment-dir experiments/my_experiment
    python experiments/train_modal.py --experiment-dir experiments/my_experiment --use-wandb
"""

import argparse
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_experiment_config(experiment_dir: Path) -> dict:
    """
    Load settings.yml from the experiment directory using the same logic as
    pyreflect's load_config (searches for settings.yml / settings.yaml).
    """
    for name in ["settings.yml", "settings.yaml"]:
        cfg_path = experiment_dir / name
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f)

    sys.exit(
        f"Error: no settings.yml found in {experiment_dir}.\n"
        "Run `pyreflect init` inside the experiment directory first."
    )


def _extract_training_params(config: dict, experiment_dir: Path) -> dict:
    """
    Pull the nr_predict_sld parameters out of the pyreflect settings dict
    and resolve file paths relative to the experiment directory.
    """
    nr_cfg = config["nr_predict_sld"]
    files = nr_cfg["file"]
    models = nr_cfg["models"]

    return {
        "nr_train": str((experiment_dir / files["nr_train"]).resolve()),
        "sld_train": str((experiment_dir / files["sld_train"]).resolve()),
        "norm_stats": str((experiment_dir / models["normalization_stats"]).resolve()),
        "layers": models["layers"],
        "dropout": models["dropout"],
        "batch_size": models["batch_size"],
        "epochs": models["epochs"],
        "num_curves": models.get("num_curves", "unknown"),
    }


# ---------------------------------------------------------------------------
# Core training function
# All imports live inside this function so Modal can serialize and run it
# on a remote GPU without needing local modules to be importable there.
# ---------------------------------------------------------------------------

def _run_training(params: dict, use_wandb: bool) -> bytes:
    """
    Train the CNN with the given parameters and return the model weights as bytes.

    Designed to work both locally and on Modal GPU. All heavy imports are
    deferred inside this function so Modal can serialize it cleanly.

    Parameters
    ----------
    params : dict
        Training parameters extracted from settings.yml, with data paths
        already resolved for the execution environment (local or /root/data).
    use_wandb : bool
        Whether to log training metrics to Weights & Biases.

    Returns
    -------
    bytes
        Serialized model state dict (.pth file contents).
    """
    import torch
    from pyreflect.input import NRSLDDataProcessor
    from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
    import pyreflect.pipelines.reflectivity_pipeline as workflow

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on: {gpu_name}")
    print(f"  layers={params['layers']}, dropout={params['dropout']}, "
          f"batch_size={params['batch_size']}, epochs={params['epochs']}")
    print(f"  curves={params['num_curves']}\n")

    if use_wandb:
        import wandb
        exp_name = params.get("experiment_name", "train_modal")
        wandb.init(
            project=params.get("wandb_project", "pyreflect-training"),
            name=exp_name,
            config={
                "layers": params["layers"],
                "dropout": params["dropout"],
                "batch_size": params["batch_size"],
                "epochs": params["epochs"],
                "num_curves": params["num_curves"],
            },
        )

    dproc = NRSLDDataProcessor(params["nr_train"], params["sld_train"]).load_data()
    X, y = workflow.preprocess(dproc, params["norm_stats"])

    trainer = NRSLDModelTrainer(
        X=X, y=y,
        layers=params["layers"],
        dropout=params["dropout"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
    )
    model = trainer.train_pipeline()

    if use_wandb:
        import wandb
        wandb.finish()

    # Serialize model weights to bytes for transfer back to local machine
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Modal runner
# ---------------------------------------------------------------------------

def run_modal(experiment_dir: Path, params: dict, use_wandb: bool) -> None:
    """
    Upload experiment data to Modal, train on a GPU, and save the model locally.
    """
    try:
        import modal
    except ImportError:
        sys.exit("Error: 'modal' is not installed. Run: pip install modal")

    data_dir = experiment_dir / "data"
    if not data_dir.exists():
        sys.exit(f"Error: data directory not found: {data_dir}")

    # Build the remote params with Modal-side paths substituted in
    modal_params = {
        **params,
        "nr_train": "/root/data/curves/nr_train.npy",
        "sld_train": "/root/data/curves/sld_train.npy",
        "norm_stats": "/root/data/normalization_stat.npy",
        "experiment_name": experiment_dir.name,
        "wandb_project": "pyreflect-training",
    }

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "torch==2.5.1", "numpy==2.1.0",
            "pandas", "scikit-learn", "scipy",
            "opencv-python", "pyyaml", "tqdm", "refnx", "llvmlite", "numba",
            *(["wandb"] if use_wandb else []),
        )
        .apt_install("git")
        .run_commands("pip install git+https://github.com/williamQyq/pyreflect.git")
        .add_local_dir(str(data_dir), remote_path="/root/data", copy=True)
    )

    secrets = [modal.Secret.from_name("wandb-secret")] if use_wandb else []

    app = modal.App(f"pyreflect-train-{experiment_dir.name}")

    remote_train = app.function(
        gpu="T4",
        image=image,
        secrets=secrets,
        timeout=4 * 60 * 60,
    )(_run_training)

    print(f"Launching training on Modal GPU (T4)...")
    print(f"Experiment: {experiment_dir}")
    print(f"Params: layers={params['layers']}, dropout={params['dropout']}, "
          f"batch_size={params['batch_size']}, epochs={params['epochs']}\n")

    with app.run():
        model_bytes = remote_train.remote(modal_params, use_wandb)

    output_path = experiment_dir / "trained_model.pth"
    output_path.write_bytes(model_bytes)
    print(f"\nModel saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a pyreflect NR->SLD CNN on a Modal GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiments/train_modal.py "
            "--experiment-dir experiments/best_model_125k_6L_0.087D\n"
            "  python experiments/train_modal.py "
            "--experiment-dir experiments/best_model_125k_6L_0.087D --use-wandb\n"
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help=(
            "Path to the experiment directory containing settings.yml and data/. "
            "Create one with `pyreflect init` and edit settings.yml with your params."
        ),
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases (requires wandb and a 'wandb-secret' Modal secret)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        sys.exit(f"Error: experiment directory not found: {experiment_dir}")

    config = _load_experiment_config(experiment_dir)
    params = _extract_training_params(config, experiment_dir)

    run_modal(experiment_dir, params, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
