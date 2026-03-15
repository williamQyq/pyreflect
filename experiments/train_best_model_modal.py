"""
Train Final Model on Modal GPU with Best Optuna Parameters
Trial 12: 6 layers, 0.11 dropout, batch 64
"""

import modal

# Best parameters from Optuna Trial 12
LAYERS = 6
DROPOUT = 0.11
BATCH_SIZE = 64
EPOCHS = 20

app = modal.App("pyreflect-train-final")

# Use cached image from Optuna (fast!)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1", "numpy==2.1.0", "wandb",
        "pandas", "scikit-learn", "scipy", "opencv-python",
        "pyyaml", "tqdm", "refnx", "llvmlite", "numba"
    )
    .apt_install("git")
    .run_commands("pip install git+https://github.com/williamQyq/pyreflect.git")
    .add_local_dir("../master_training_data", remote_path="/root/data", copy=True)
)


@app.function(gpu="T4", image=image, secrets=[modal.Secret.from_name("wandb-secret")], timeout=2*60*60)
def train_final_model():
    """Train production model with optimal params"""
    
    import wandb
    import torch
    from pathlib import Path
    from pyreflect.input import NRSLDDataProcessor
    from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
    import pyreflect.pipelines.reflectivity_pipeline as workflow
    
    print(f"🚀 Training final model on GPU: {torch.cuda.get_device_name(0)}")
    print(f"Params: {LAYERS}L, {DROPOUT}D, {BATCH_SIZE}B, {EPOCHS}E\n")
    
    # W&B tracking
    wandb.init(
        project="new-pyreflect-overfitting",
        name=f"FINAL_MODEL_{LAYERS}L_{DROPOUT}D_{EPOCHS}E",
        config={"layers": LAYERS, "dropout": DROPOUT, "batch_size": BATCH_SIZE, "epochs": EPOCHS}
    )
    
    # Load data
    nr_file = "/root/data/data/curves/nr_train.npy"
    sld_file = "/root/data/data/curves/sld_train.npy"
    norm_file = "/root/data/data/normalization_stat.npy"
    
    dproc = NRSLDDataProcessor(nr_file, sld_file).load_data()
    X, y = workflow.preprocess(dproc, norm_file)
    
    # Train
    trainer = NRSLDModelTrainer(X=X, y=y, layers=LAYERS, dropout=DROPOUT, 
                                batch_size=BATCH_SIZE, epochs=EPOCHS)
    model = trainer.train_pipeline()
    
    # Save model
    torch.save(model.state_dict(), "/tmp/model.pth")
    with open("/tmp/model.pth", "rb") as f:
        model_bytes = f.read()
    
    wandb.finish()
    return model_bytes


@app.local_entrypoint()
def main():
    print("🚀 Training final model on Modal GPU...\n")
    model_bytes = train_final_model.remote()
    
    # Save locally
    from pathlib import Path
    output = Path("best_model_6L_0.11D/trained_model.pth")
    output.parent.mkdir(exist_ok=True)
    
    with open(output, "wb") as f:
        f.write(model_bytes)
    
    print(f"\n✅ Model saved: {output}")
    print("📈 View training: https://wandb.ai/raheja-k-northeastern-university/new-pyreflect-overfitting")