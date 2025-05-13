#!/bin/bash

# Load necessary modules
module load anaconda3/2022.05 cuda/12.1

# Create and activate the Conda environment
ENV_NAME="MY_env"
PYTHON_VERSION="3.11"

# Create the environment
conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate the environment
source activate "$ENV_NAME"

# Install required packages
conda install jupyterlab -y
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0 -y

# Export LD_LIBRARY_PATH to use CUDA libraries inside the environment
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/"

# Make the LD_LIBRARY_PATH update persistent across future environment activations
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# Upgrade pip
pip install --upgrade pip

echo "Conda environment '$ENV_NAME' created and configured successfully."
