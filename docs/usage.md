# Pyreflect Usage Guide

This document explains how to *use* the `pyreflect` CLI and pipelines once you have an environment set up. For installation and environment setup, see the root `README.md`.

---

## 1. Concepts and Components

`pyreflect` provides two main workflows for neutron reflectivity (NR) analysis:

- **NR → SLD prediction (CNN)**  
  Train a convolutional neural network (CNN) to predict scattering length density (SLD) profiles from NR curves, and then apply it to experimental NR data.

- **SLD → Chi parameter prediction (Autoencoder + MLP)**  
  Train an autoencoder + MLP model to predict interaction (Chi) parameters from SLD profiles.

Both workflows are driven by a central configuration file (`settings.yml`) and run through a Typer-based CLI (`python -m pyreflect`).

---

## 2. CLI Overview

The CLI entry point is:

```bash
python -m pyreflect --help
```

Main commands:

- `init` – scaffold a project configuration and data directories.
- `run` – execute one or both analysis pipelines using the configuration.

### 2.1 `init` – initialize a project

```bash
python -m pyreflect init [--root PATH] [--force]
```

- **`--root PATH`** (optional): project root where configuration and data folders will be created. Defaults to the current directory.
- **`--force`** (optional): overwrite an existing `settings.yml`.

What it does:

- Creates `settings.yml` in the root (from a built-in template).
- Creates `data/` and `data/curves/` under the root for storing NR/SLD data and generated curves.

After running `init`, you must **edit `settings.yml`** to point to your real data locations and tune model/training parameters.

### 2.2 `run` – run pipelines

```bash
python -m pyreflect run \
    [--root PATH] \
    [--enable-chi-prediction] \
    [--enable-sld-prediction]
```

- **`--root PATH`** (optional): project root that contains `settings.yml`. Defaults to the current directory.
- **`--enable-chi-prediction`**: run the SLD → Chi prediction pipeline.
- **`--enable-sld-prediction`**: run the NR → SLD prediction pipeline.

Important details:

- `pyreflect` always loads configuration from `settings.yml` (or `settings.yaml` / `settings.json`) in `--root`.  
  Paths to NR, SLD, and Chi data are **not** passed on the command line; they are read from the config.
- If a required file or config key is missing, you will get a clear error describing what needs to be fixed in `settings.yml`.

---

## 3. Configuration (`settings.yml`)

The CLI expects a nested configuration with two main sections:

- `nr_predict_sld`: NR → SLD prediction (CNN)
- `sld_predict_chi`: SLD → Chi prediction (Autoencoder + MLP)

Below is a **minimal example** (names and structure should be adapted to your use case; see the notebooks for more complete examples):

```yaml
nr_predict_sld:
  file:
    nr_train: data/curves/nr_train.npy          # synthetic NR curves for training
    sld_train: data/curves/sld_train.npy        # synthetic SLD curves for training
    experimental_nr_file: data/experimental/nr_experimental.npy
  models:
    model: models/nr_sld_cnn.pt                 # CNN weights
    normalization_stats: models/nr_sld_norm.npy # normalization stats (saved by training)
    num_curves: 10000                           # number of synthetic curves to generate
    num_film_layers: 3                          # model for film stack
    batch_size: 64
    epochs: 50
    layers: 12
    dropout: 0.5

sld_predict_chi:
  file:
    model_experimental_sld_profile: data/chi/sld_experimental.npy
    model_sld_file: data/chi/sld_train.npy       # SLD curves used to train the AE/MLP
    model_chi_params_file: data/chi/chi_params.npy
  models:
    latent_dim: 32
    batch_size: 64
    ae_epochs: 100
    mlp_epochs: 200
```

Notes:

- All paths are interpreted **relative to the project root** you pass via `--root` (or the current directory, if omitted).
- When you call `python -m pyreflect run`, the code internally sets `config["root"] = root` and then resolves relative paths against it.
- Files listed under `file:` sections are validated:
  - Training data (`nr_train`, `sld_train`, `model_sld_file`, `model_chi_params_file`, etc.) must exist when required.
  - Model and normalization paths are created as needed when training.

---

## 4. NR → SLD Pipeline (CNN)

This pipeline lives mainly in `pyreflect.pipelines.reflectivity_pipeline` and `pyreflect.pipelines.run_sld_prediction`.

### 4.1 High-level steps

When you run:

```bash
python -m pyreflect run --enable-sld-prediction
```

The following happens (assuming `settings.yml` is configured under `nr_predict_sld`):

1. **Load configuration** from `settings.yml` in the chosen root.
2. **Check for an existing trained CNN model** at `nr_predict_sld.models.model`.
3. If **no model is found**:
   - Initialize `NRSLDCurvesGeneratorParams` from config.
   - Use `ReflectivityDataGenerator` to generate synthetic NR/SLD pairs and save them to `nr_train` and `sld_train`.
   - Initialize `NRSLDModelTrainerParams` and load the synthetic data via `NRSLDDataProcessor`.
   - Preprocess and normalize NR/SLD curves, saving normalization stats to `normalization_stats`.
   - Train a CNN (`NRSLDModelTrainer`) that predicts SLD from NR and save its weights.
4. Whether newly trained or loaded, the CNN model is then used for **inference**:
   - `NRSLDModelInferenceParams` reads `experimental_nr_file` and `normalization_stats`.
   - The experimental NR curves are normalized using the same stats as training.
   - SLD profiles are predicted and written to the configured output paths (see notebooks and config for specific filenames).

### 4.2 Programmatic usage (optional)

You can also drive the NR → SLD workflow from Python:

```python
from pyreflect.pipelines.reflectivity_pipeline import ReflectivityPipeline
from pyreflect.models.config import NRSLDCurvesGeneratorParams, NRSLDModelTrainerParams

# Create params from your loaded YAML config
params_gen = NRSLDCurvesGeneratorParams(_config=config)
params_train = NRSLDModelTrainerParams(_config=config)

pipeline = ReflectivityPipeline(
    generator_params=params_gen,
    trainer_params=params_train,
)

nr_curves, sld_curves = pipeline.generate_synthetic_curves()
X, y = pipeline.prepare_training_data()
model = pipeline.train_model((X, y))

# Later, load model and run predictions
model = pipeline.load_model()
predicted_sld = pipeline.predict_sld(nr_data="path/to/experimental_nr.npy")
```

For concrete, runnable examples, see `examples/example_reflectivity_pipeline.ipynb`.

---

## 5. SLD → Chi Pipeline (Autoencoder + MLP)

This pipeline is implemented in `pyreflect.pipelines.run_chi_prediction`.

### 5.1 High-level steps

When you run:

```bash
python -m pyreflect run --enable-chi-prediction
```

The following happens (with `sld_predict_chi` configured in `settings.yml`):

1. **Load configuration** from `settings.yml` and inject the project root into `config["root"]`.
2. Construct `ChiPredTrainingParams` from the `sld_predict_chi` section:
   - `model_experimental_sld_profile`
   - `model_sld_file`
   - `model_chi_params_file`
   - `latent_dim`, `batch_size`, `ae_epochs`, `mlp_epochs`.
3. Load and preprocess the SLD and Chi datasets using `SLDChiDataProcessor`:
   - Normalize and prepare arrays for the autoencoder and MLP.
4. Train the autoencoder and MLP (`train_autoencoder_mlp_chi_pred.run_model_training`).
5. Run `run_model_prediction` to:
   - Reconstruct SLD profiles for the experimental data.
   - Predict Chi parameters for those profiles.
6. Print the final Chi predictions (`df_predictions`) and optionally save results (as implemented in the pipeline).

### 5.2 Typical usage pattern

1. Prepare SLD+Chi datasets from your existing analysis or simulations.
2. Set the corresponding file paths in `sld_predict_chi.file` in `settings.yml`.
3. Optionally, explore and pre-process in a notebook such as:
   - `examples/example_notebook_autoencoder.ipynb`
4. Run the CLI pipeline:

   ```bash
   python -m pyreflect run --enable-chi-prediction
   ```

---

## 6. Example Notebooks and Datasets

- **Synthetic data generation**:  
  `examples/example_notebook_generate_training_datasets.ipynb` demonstrates how to generate synthetic NR/SLD datasets suitable for training the CNN.

- **NR → SLD pipeline**:  
  `examples/example_reflectivity_pipeline.ipynb` walks through data preprocessing, model training, and inference.

- **Autoencoder + MLP for Chi**:  
  `examples/example_notebook_autoencoder.ipynb` shows the SLD → Chi workflow and how to interpret results.

- **PCA / NR data checks**:  
  `examples/example_notebook_PCA_NR_check.ipynb` provides exploratory analysis tools for NR curves.

- **Experimental datasets**:  
  The `datasets/` directory contains example processed experimental NR data, manual-fit SLD profiles, and autoencoder-denoised NR data from lab measurements. These can be used as reference inputs when configuring your own `settings.yml`.

---

## 7. Summary

- Use `python -m pyreflect init` to create `settings.yml` and the data folder structure in a chosen project root.
- Edit `settings.yml` to point to your NR, SLD, and Chi datasets and to control training parameters.
- Use `python -m pyreflect run --enable-sld-prediction` for NR → SLD, and `python -m pyreflect run --enable-chi-prediction` for SLD → Chi.
- For hands-on examples, start with the notebooks in the `examples/` folder.
