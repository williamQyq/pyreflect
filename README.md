# **NR-SCFT-ML: Neutron Reflectivity SCFT Machine Learning**

A Python package for Neutron Reflectivity data analysis using Machine Learning.  
Including **Chi parameter prediction** from **SLD profile data** and **SLD profile prediction from NR curves**.

---

## **Overview**

**NR-SCFT-ML** (implemented as the `pyreflect` package) is a Python toolkit for efficient processing and analysis of Neutron Reflectivity (NR) data. It provides a streamlined pipeline for:

- Preprocessing NR datasets  
- Predicting SLD profiles using CNN-based models  
- Predicting **Chi** parameters using a combined Autoencoder and MLP architecture  

---

## **Prerequisites**

- Python >= 3.10  
- **Strongly recommended:** Conda (local machines and HPC clusters)  
- Optional: GPU with CUDA-capable PyTorch build for faster training and inference  

Large-scale synthetic data generation and model training are memory intensive. HPC resources or a workstation with large RAM and optionally a GPU are recommended.

---

## **Installation**

`pyreflect` depends on scientific Python libraries with native extensions (NumPy, SciPy, refnx, refl1d, etc.).  
For reliability across platforms, **Conda is the recommended installation method**, especially on Windows and HPC systems.

---

## Option 1. ✅ Recommended: Conda (Local or HPC)

### 1. Create and activate a Conda environment

```bash
conda create -n pyreflect python=3.10
conda activate pyreflect
```

### 2. Install scientific dependencies via conda-forge

```bash
conda install -c conda-forge \
    numpy scipy refnx refl1d numba llvmlite
```

### 3. Install pyreflect

```bash
pip install pyreflect-nr
```

After installation, you can access the CLI via:

```bash
python -m pyreflect --help
```

### GPU support (optional)

For GPU-accelerated training and inference, install a CUDA-enabled PyTorch build inside the same environment, following the official PyTorch instructions for your system or cluster.

---

## Option 2. 🛠 Automated setup (using `setup.sh`)

This repository provides a helper script [`setup.sh`](setup.sh) that:

- Detects your Conda installation  
- Creates a Conda environment named `pyreflect` (Python 3.10)  
- Installs core scientific dependencies  
- Registers a Jupyter kernel called `Python (pyreflect)`  

You can fetch and run it as follows:

```bash
curl -fsSLo setup.sh https://raw.githubusercontent.com/williamQyq/pyreflect/main/setup.sh
bash setup.sh
```

After it completes:

- In Jupyter, select the `Python (pyreflect)` kernel  
- The environment is ready for use  

---

## ⚠️ pip-only installation (not recommended on Windows)

A pure `pip` installation may work on Linux or macOS systems with a full build toolchain.  
On Windows, this may require Microsoft C++ Build Tools and is **not** recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pyreflect-nr
```

If installation fails due to native extensions, switch to the Conda workflow above.

---

## 👩‍💻 Development setup (for contributors)

Poetry is used for development and maintenance, not for end-user installation.

```bash
git clone https://github.com/williamQyq/pyreflect.git
cd pyreflect

poetry install
poetry shell
```

You can then run:

```bash
python -m pyreflect --help
```

---

## **Example Notebooks**

Example notebooks are available in the `examples` directory:

- [examples/example_notebook_generate_training_datasets.ipynb](examples/example_notebook_generate_training_datasets.ipynb) – synthetic NR/SLD dataset generation  
- [examples/example_notebook_autoencoder.ipynb](examples/example_notebook_autoencoder.ipynb) – Autoencoder + MLP workflow for Chi prediction  
- [examples/example_notebook_PCA_NR_check.ipynb](examples/example_notebook_PCA_NR_check.ipynb) – NR data exploration and PCA  
- [examples/example_reflectivity_pipeline.ipynb](examples/example_reflectivity_pipeline.ipynb) – end-to-end NR → SLD pipeline  

Tutorial video:

<a href="https://youtu.be/cc8xeLhOXDo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" width="50" height="50" />
</a>

For detailed CLI usage, configuration, and typical research workflows, see [docs/usage.md](docs/usage.md).

---

## Credits

This project builds on work by:

- Brian Qu ([NR-SLD-CNN](https://github.com/BBQ591/NR-SLD-CNN))  
- Dr. Rajeev Kumar  
- Prof. Miguel Fuentes-Cabrera ([NR-SCFT-ML](https://github.com/miguel-fc/NR-SCFT-ML))
- Shanshou Li
- Hudson Kass

## Author

Yuqing Qiao (William) – Maintainer and developer of this package
Shanshou Li - Developer
Hudson Kass - Developer


