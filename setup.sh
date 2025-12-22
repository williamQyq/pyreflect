#!/bin/bash
set -euo pipefail

# === Environment variables ===
ENV_NAME="pyreflectenv"
DISPLAY_NAME="PyreflectEnvironment"

# === Detect platform ===
PLATFORM="unknown"
case "${OSTYPE:-}" in
    linux*) PLATFORM="linux" ;;
    msys*|cygwin*|win32*) PLATFORM="windows" ;;
    *) PLATFORM="unknown" ;;
esac

IS_WSL=0
if [[ "$PLATFORM" == "linux" ]] && grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1
fi

echo "🔎 Detected platform: ${PLATFORM}${IS_WSL:+ (WSL)}"

# === Helper to source conda when not on PATH ===
declare -a COMMON_CONDA_PREFIXES=()
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    COMMON_CONDA_PREFIXES+=("${CONDA_PREFIX}")
fi
if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
    COMMON_CONDA_PREFIXES+=("${MAMBA_ROOT_PREFIX}")
fi
COMMON_CONDA_PREFIXES+=("$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3" "/opt/conda")

source_conda_from_known_locations() {
    for prefix in "${COMMON_CONDA_PREFIXES[@]}"; do
        local conda_init="${prefix}/etc/profile.d/conda.sh"
        if [[ -f "$conda_init" ]]; then
            echo "ℹ️ Loading Conda initialization script from '${conda_init}'."
            # shellcheck disable=SC1090
            source "$conda_init"
            return 0
        fi
    done
    return 1
}

# === Load module when available (common on linux clusters) ===
if [[ "$PLATFORM" == "linux" ]] && type module >/dev/null 2>&1; then
    module load conda
elif [[ "$PLATFORM" == "linux" ]]; then
    echo "ℹ️ 'module' command not found; attempting to locate Conda manually."
else
    echo "ℹ️ Skipping 'module load conda' for platform '${PLATFORM}'."
fi

if ! command -v conda >/dev/null 2>&1; then
    source_conda_from_known_locations || true
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
