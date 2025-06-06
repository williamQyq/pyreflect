#!/bin/bash

# === Environment variables ===
ENV_NAME="pyreflectenv"
DISPLAY_NAME="PyreflectEnvironment"

# === Load module ===
module load conda

# # === Ensure conda shell functions are available ===
eval "$(conda shell.bash hook)"

# === Check if conda environment exists ===
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "✅ Conda environment '${ENV_NAME}' already exists."
else
    echo "🔧 Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi

# === Activate the conda environment ===
conda activate "${ENV_NAME}"

conda install -c conda-forge ipykernel

# === Check if Jupyter kernel exists ===
if jupyter kernelspec list | grep -q "${ENV_NAME}"; then
    echo "✅ Jupyter kernel '${ENV_NAME}' already exists."
else
    echo "🔧 Creating Jupyter kernel '${ENV_NAME}'..."
    python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${DISPLAY_NAME}"
    echo "✅ Created Jupyter kernel '${DISPLAY_NAME}'."
fi