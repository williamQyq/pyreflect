"""
Optuna Hyperparameter Search on Modal GPU
"""

import modal

# Configuration
N_TRIALS = 20
EPOCHS_PER_TRIAL = 7
WANDB_PROJECT = "new-pyreflect-overfitting"

app = modal.App("pyreflect-optuna-new")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.0",
        "optuna",
        "wandb",
        "plotly",
        "pandas",
        "scikit-learn",
        "scipy",
        "opencv-python",
        "pyyaml",
        "tqdm",
        "refnx",
        "llvmlite",
        "numba"
    )
    .apt_install("git")
    .run_commands("pip install git+https://github.com/williamQyq/pyreflect.git")
    .add_local_dir(
        "../master_training_data",
        remote_path="/root/master_training_data",
        copy=True
    )
)


@app.function(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=12 * 60 * 60
)
def run_optuna_search():
    """Run Optuna on Modal GPU"""
    
    import optuna
    import wandb
    import numpy as np
    import torch
    import torch.nn as nn
    from dataclasses import dataclass
    from pathlib import Path
    from pyreflect.input import NRSLDDataProcessor, DataProcessor
    from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
    import pyreflect.pipelines.reflectivity_pipeline as workflow
    
    print("🔍 Starting Optuna on Modal GPU...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    DATA_ROOT = Path("/root/master_training_data")
    NR_FILE = str(DATA_ROOT / "data/curves/nr_train.npy")
    SLD_FILE = str(DATA_ROOT / "data/curves/sld_train.npy")
    NORM_STATS_FILE = str(DATA_ROOT / "data/normalization_stat.npy")
    
    @dataclass
    class SimpleTrainerParams:
        layers: int
        dropout: float
        batch_size: int
        epochs: int
    
    def train_with_logging(X_train, y_train, trainer_params, wandb_run):
        trainer = NRSLDModelTrainer(
            X=X_train, y=y_train,
            layers=trainer_params.layers,
            batch_size=trainer_params.batch_size,
            epochs=trainer_params.epochs,
            dropout=trainer_params.dropout
        )
        
        list_arrays = DataProcessor.split_arrays(X_train, y_train, size_split=0.8)
        tensor_arrays = DataProcessor.convert_tensors(list_arrays)
        _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
            *tensor_arrays, batch_size=trainer_params.batch_size)
        
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=2.15481e-05, weight_decay=2.6324e-05)
        loss_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(trainer_params.epochs):
            train_loss = trainer.train_model(trainer.model, train_loader, optimizer, loss_fn)
            val_loss = trainer.validate_model(trainer.model, valid_loader, loss_fn)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "overfitting_gap": val_loss - train_loss
            })
            
            print(f"  Epoch {epoch + 1}/{trainer_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        return train_losses, val_losses
    
    def objective(trial):
        layers = trial.suggest_int("layers", 6, 12)
        dropout = trial.suggest_float("dropout", 0.3, 0.7)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        print(f"\nTrial {trial.number + 1}/{N_TRIALS}: layers={layers}, dropout={dropout:.2f}, batch={batch_size}")
        
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial_{trial.number}_L{layers}_D{dropout:.2f}_B{batch_size}",
            config={"layers": layers, "dropout": dropout, "batch_size": batch_size,
                   "epochs": EPOCHS_PER_TRIAL, "platform": "modal", "gpu": "T4"},
            reinit=True
        )
        
        try:
            dproc = NRSLDDataProcessor(NR_FILE, SLD_FILE).load_data()
            X_train, y_train = workflow.preprocess(dproc, NORM_STATS_FILE)
            
            trainer_params = SimpleTrainerParams(layers, dropout, batch_size, EPOCHS_PER_TRIAL)
            train_losses, val_losses = train_with_logging(X_train, y_train, trainer_params, run)
            
            final_val_loss = val_losses[-1]
            
            wandb.log({
                "final_val_loss": final_val_loss,
                "final_train_loss": train_losses[-1],
                "overfitting_gap": val_losses[-1] - train_losses[-1]
            })
            
            wandb.finish()
            return final_val_loss
            
        except Exception as e:
            print(f"❌ Trial failed: {e}")
            wandb.finish()
            raise optuna.TrialPruned()
    
    study = optuna.create_study(direction="minimize", study_name="pyreflect_modal")
    
    print(f"🚀 Starting {N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs each\n")
    study.optimize(objective, n_trials=N_TRIALS)
    
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        return {"success": False}
    
    print(f"\n✅ Best trial: #{study.best_trial.number}")
    print(f"   Val loss: {study.best_value:.6f}")
    print(f"   Params: {study.best_params}")
    
    return {
        "success": True,
        "best_trial": study.best_trial.number,
        "best_val_loss": study.best_value,
        "best_params": study.best_params
    }


@app.local_entrypoint()
def main():
    print("🚀 Launching Optuna on Modal T4 GPU...")
    result = run_optuna_search.remote()
    
    if result["success"]:
        print(f"\n✅ Complete! Best params: {result['best_params']}")
        print(f"📈 View at: https://wandb.ai/raheja-k-northeastern-university/{WANDB_PROJECT}")


if __name__ == "__main__":
    main()