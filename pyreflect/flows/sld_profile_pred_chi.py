import numpy as np
import torch
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_model_prediction(percep, autoencoder, expt_arr, sld_arr, num_params):
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

    # Interpolate the experimental curve using the first SLD curve
    int_expt = np.interp(sld_arr[0][0], expt_arr[0], expt_arr[1])
    expt_arr_n = np.asarray([[expt_arr[0], expt_arr[1]]])

    # Convert experimental curve to tensor
    expt_curve = torch.from_numpy(expt_arr_n[0]).flatten().float().to(device)

    # Set models to evaluation mode
    autoencoder.eval()
    percep.eval()

    expt_labels = []

    with torch.no_grad():
        # Encode the experimental data
        encoded_expt = autoencoder.encoder(expt_curve)
        decoded_expt = autoencoder.decoder(encoded_expt)
        out_label = percep(encoded_expt)

    # Convert tensors to NumPy for storage
    encoded_expt = encoded_expt.cpu().numpy()
    out_label = out_label.cpu().numpy()

    # Store latent variables
    expt_label = {f"l{i+1}": enc for i, enc in enumerate(encoded_expt)}

    # Store predicted chi parameters
    for i in range(num_params):
        label_index = f'chi{i+1}'
        expt_label[label_index] = out_label[i]

    expt_labels.append(expt_label)

    # Convert to DataFrame
    df_expt_labels = pd.DataFrame(expt_labels)

    return df_expt_labels
