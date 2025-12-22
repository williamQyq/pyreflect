#!/usr/bin/env bash
set -e

ENV_NAME="pyreflect"
PYTHON_VERSION="3.10"

echo "=== Checking environment ==="

# -----------------------------
# 1. Handle `module` command
# -----------------------------
if ! command -v module >/dev/null 2>&1; then
    echo "[INFO] 'module' command not found. Assuming non-HPC environment."
else
    echo "[INFO] 'module' command found."
    # Optional:
    # module load anaconda
fi

# -----------------------------
# 2. Check Conda availability
# -----------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda not found in PATH."
    exit 1
fi

echo "[OK] Conda detected: $(conda --version)"

# -----------------------------
# 3. Initialize Conda for shell
# -----------------------------
CONDA_BASE=$(conda info --base)

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[OK] Conda shell initialized."

# -----------------------------
# 4. Create environment if needed
# -----------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] Conda environment '$ENV_NAME' already exists."
else
    echo "[INFO] Creating Conda environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# -----------------------------
# 5. Activate environment
# -----------------------------
echo "[INFO] Activating environment '$ENV_NAME'"
conda activate "$ENV_NAME"

# -----------------------------
# 6. Install ipykernel
# -----------------------------
echo "[INFO] Installing ipykernel..."
conda install -y ipykernel

# -----------------------------
# 7. Register Jupyter kernel
# -----------------------------
echo "[INFO] Registering Jupyter kernel: Python ($ENV_NAME)"

python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo "=== pyreflect environment setup complete ==="
echo "You can now select 'Python ($ENV_NAME)' in Jupyter."
