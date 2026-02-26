"""
Optuna Hyperparameter Search for PyReflect NR->SLD Model
Local execution with W&B logging
"""

import optuna
import wandb
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from pyreflect.input import NRSLDDataProcessor, DataProcessor
from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
import pyreflect.pipelines.reflectivity_pipeline as workflow

# Configuration
PROJECT_ROOT = Path("../master_training_data")
WANDB_PROJECT = "new-pyreflect-overfitting"  # Your new W&B project
N_TRIALS = 20
EPOCHS_PER_TRIAL = 7


def objective(trial):
    """Optuna objective - tries different hyperparameters"""
    
    layers = trial.suggest_int("layers", 6, 12)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number + 1}/{N_TRIALS}")
    print(f"Testing: layers={layers}, dropout={dropout:.2f}, batch_size={batch_size}")
    print(f"{'='*60}\n")
    
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"trial_{trial.number}_L{layers}_D{dropout:.2f}_B{batch_size}",
        config={
            "layers": layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": EPOCHS_PER_TRIAL,
            "trial_number": trial.number
        },
        reinit=True
    )
    
    try:
        nr_file = str((PROJECT_ROOT / "data/curves/nr_train.npy").resolve())
        sld_file = str((PROJECT_ROOT / "data/curves/sld_train.npy").resolve())
        norm_stats_file = str((PROJECT_ROOT / "data/normalization_stat.npy").resolve())
        
        dproc = NRSLDDataProcessor(nr_file, sld_file).load_data()
        X_train, y_train = workflow.preprocess(dproc, norm_stats_file)
        
        @dataclass
        class SimpleTrainerParams:
            layers: int
            dropout: float
            batch_size: int
            epochs: int
        
        trainer_params = SimpleTrainerParams(
            layers=layers,
            dropout=dropout,
            batch_size=batch_size,
            epochs=EPOCHS_PER_TRIAL
        )
        
        train_losses, val_losses = train_with_logging(X_train, y_train, trainer_params, run)
        
        final_val_loss = val_losses[-1]
        overfitting_gap = val_losses[-1] - train_losses[-1]
        
        wandb.log({
            "final_val_loss": final_val_loss,
            "final_train_loss": train_losses[-1],
            "overfitting_gap": overfitting_gap
        })
        
        print(f"\n✅ Trial {trial.number} complete!")
        print(f"   Final validation loss: {final_val_loss:.6f}")
        print(f"   Overfitting gap: {overfitting_gap:.6f}")
        
        wandb.finish()
        return final_val_loss
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        raise optuna.TrialPruned()


def train_with_logging(X_train, y_train, trainer_params, wandb_run):
    """Training with W&B logging"""
    
    trainer = NRSLDModelTrainer(
        X=X_train,
        y=y_train,
        layers=trainer_params.layers,
        batch_size=trainer_params.batch_size,
        epochs=trainer_params.epochs,
        dropout=trainer_params.dropout
    )
    
    list_arrays = DataProcessor.split_arrays(X_train, y_train, size_split=0.8)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=trainer_params.batch_size)
    
    import torch.nn as nn
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
        
        print(f"Epoch {epoch + 1}/{trainer_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    return train_losses, val_losses


def main():
    """Run Optuna hyperparameter search"""
    
    print("="*60)
    print("🔍 OPTUNA HYPERPARAMETER SEARCH")
    print("="*60)
    print(f"Project: {WANDB_PROJECT}")
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS_PER_TRIAL}")
    print("="*60)
    
    study = optuna.create_study(
        direction="minimize",
        study_name="pyreflect_overfitting_new",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    
    print("\n🚀 Starting optimization...\n")
    study.optimize(objective, n_trials=N_TRIALS)
    
    if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
        print("\n❌ No trials completed successfully!")
        return
    
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\n🏆 Best trial: #{study.best_trial.number}")
    print(f"   Validation loss: {study.best_value:.6f}")
    print(f"\n📊 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    results_file = PROJECT_ROOT / "optuna_best_params.txt"
    with open(results_file, "w") as f:
        f.write(f"Best validation loss: {study.best_value:.6f}\n")
        f.write(f"Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()