#!/usr/bin/env bash
set -e

ENV_NAME="pyreflect"
PYTHON_VERSION="3.10"

echo "=== Setting up Conda environment: $ENV_NAME ==="

# -----------------------------
# 1. Detect OS (via Python)
# -----------------------------
OS_NAME=$(python - <<'EOF'
import os
print(os.name)
EOF
)

if [[ "$OS_NAME" == "nt" ]]; then
    IS_WINDOWS=true
else
    IS_WINDOWS=false
fi

echo "Is Windows: $IS_WINDOWS"

# -----------------------------
# 2. Fix Conda path (Windows)
# -----------------------------
if [[ "$IS_WINDOWS" == "true" ]]; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -d "$d" ]]; then
            export PATH="$d/Scripts:$d/condabin:$PATH"
            break
        fi
    done
fi

# -----------------------------
# 3. Check Conda
# -----------------------------
CONDA_BASE="$(conda info --base 2>/dev/null)"

if [[ -z "$CONDA_BASE" ]]; then
    echo "[ERROR] Conda not found"
    exit 1
fi

# -----------------------------
# 4. Init Conda shell
# -----------------------------
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[OK] Conda found: $(conda --version)"

# -----------------------------
# 5. Create env if missing
# -----------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] Env '$ENV_NAME' already exists"
else
    echo "[INFO] Creating env '$ENV_NAME'"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# -----------------------------
# 6. Activate env
# -----------------------------
conda activate "$ENV_NAME"

# -----------------------------
# 7. Install ipykernel
# -----------------------------
conda install -y ipykernel

# -----------------------------
# 8. Install pyreflect and dependencies
#    - If running inside the git repo (pyproject.toml present), do a
#      development/editable install: `pip install -e .`
#    - Otherwise, install the published package from PyPI.
# -----------------------------
echo "[INFO] Installing scientific dependencies via conda-forge"

conda install -y -c conda-forge \
    numpy \
    scipy \
    refnx \
    refl1d \
    numba \
    llvmlite

echo "[INFO] Installing remaining Python deps via pip"

python -m pip install --upgrade pip

if [[ -f "pyproject.toml" ]]; then
    python -m pip install -e .
else
    python -m pip install pyreflect
fi


# -----------------------------
# 9. Optional: Try to install CUDA-enabled PyTorch
#    This is a best-effort step; failures will not abort the script.
#    On HPC systems without GPU access or where CUDA/toolchain is
#    managed separately, this section can simply be ignored.
# -----------------------------
echo "[INFO] Attempting to install CUDA-enabled PyTorch (optional)"

if conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1 2>/dev/null; then
    echo "[OK] Installed CUDA-enabled PyTorch via conda (pytorch + pytorch-cuda=12.1)."
else
    echo "[WARN] Could not install CUDA-enabled PyTorch with conda. Keeping CPU-only setup."
fi

# -----------------------------
# 10. Register Jupyter kernel for this environment
# -----------------------------
python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo "=== DONE ==="
echo "Select 'Python ($ENV_NAME)' in Jupyter and you're ready to use pyreflect."
