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

# -----------------------------
# 2. Fix Conda path (Windows)
# -----------------------------
if [ "$IS_WINDOWS" = true ]; then
    CONDA_ROOT="/c/Users/qyqfi/miniconda3"
    export PATH="$CONDA_ROOT/Scripts:$PATH"
fi

# -----------------------------
# 3. Check Conda
# -----------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda not found in PATH"
    exit 1
fi

echo "[OK] Conda found: $(conda --version)"

# -----------------------------
# 4. Init Conda shell
# -----------------------------
if [ "$IS_WINDOWS" = true ]; then
    source "/c/Users/qyqfi/miniconda3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

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
# 8. Register kernel
# -----------------------------
python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo "=== DONE ==="
echo "Select 'Python ($ENV_NAME)' in Jupyter"
