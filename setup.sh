#!/bin/bash
set -euo pipefail

# === Environment variables ===
ENV_NAME="pyreflectenv"
DISPLAY_NAME="PyreflectEnvironment"

# === Detect platform (linux/windows) ===
PLATFORM="unknown"
case "${OSTYPE:-}" in
    linux*) PLATFORM="linux" ;;
    msys*|cygwin*|win32*) PLATFORM="windows" ;;
    *) PLATFORM="unknown" ;;
esac

echo "🔎 Detected platform: ${PLATFORM}"

# === Load module when available (common on linux clusters) ===
if [[ "$PLATFORM" == "linux" ]] && type module >/dev/null 2>&1; then
    module load conda
elif [[ "$PLATFORM" == "linux" ]]; then
    echo "ℹ️ 'module' command not found; assuming Conda is already on PATH."
else
    echo "ℹ️ Skipping 'module load conda' for platform '${PLATFORM}'."
fi

# === Ensure conda is installed before proceeding ===
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ Conda is not installed or not on PATH. Please install Conda before running setup." >&2
    exit 1
fi

# === Ensure conda shell functions are available ===
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

conda install -y -c conda-forge ipykernel


echo "🔧 Creating Jupyter kernel '${ENV_NAME}'..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${DISPLAY_NAME}"
echo "✅ Created Jupyter kernel '${DISPLAY_NAME}'."
