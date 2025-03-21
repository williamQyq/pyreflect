import numpy as np
import torch
import pandas as pd

from pyreflect.models.config import DEVICE as device

def run_model_prediction(percep, autoencoder, X:np.ndarray):
    """
    Runs the trained Autoencoder and MLP model on experimental data
    to generate latent representations and chi parameter predictions.

    Parameters:
    - percep: Trained MLP model.
    - autoencoder: Trained Autoencoder model.
    - expt_arr: Experimental SLD curve data (NumPy array).
    - sld_arr: SLD profile data (NumPy array) for interpolation.
    - num_params: Number of chi parameters to predict.
    - device: Computation device (CPU/GPU).

    Returns:
    - df_expt_labels: DataFrame containing predicted latent variables and chi parameters.
    """
    X = np.array(X)  # ensure it's a NumPy array
    if X.ndim == 3:
        # From (batch, 2, features) → (batch, 2 * features)
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 2 and X.shape[0] == 2:
        X = X.reshape(1, -1)
    else:
        raise ValueError('X must be either 3 or 2 dimensional.')

    print(f"\nX shape for model training: {X.shape}")
    # Convert to torch tensor
    expt_curve = torch.tensor(X, dtype=torch.float32).to(device)

    # Set models to evaluation mode
    autoencoder.eval()
    percep.eval()

    with torch.no_grad():
        # Encode the experimental data
        encoded_expt = autoencoder.encoder(expt_curve)
        decoded_expt = autoencoder.decoder(encoded_expt)
        out_label = percep(encoded_expt)

    # Convert tensors to NumPy for storage
    encoded_expt = encoded_expt.cpu().numpy()
    out_label = out_label.cpu().numpy()

    latent = encoded_expt[0]
    chis = out_label[0]

    # Build a dictionary of values
    row = {**{f"l{i + 1}": val for i, val in enumerate(latent)},
           **{f"chi{i + 1}": val for i, val in enumerate(chis)}}

    # Create DataFrame from a list of dicts
    df_chi = pd.DataFrame([row])

    #chi label dataframe and latent variables
    return df_chi , decoded_expt
