import yaml
from pathlib import Path
from typing import Annotated

import typer
from pkg_resources import require

import pyreflect.flows as workflow

import pandas as pd

from pyreflect.models.config import ChiPredTrainingParams

app = typer.Typer(help="A CLI tool for neutron reflectivity data processing.")

# Command to initialize a configuration file
@app.command("init")
def init_settings(
    directory: Annotated[
        Path,
        typer.Option(
            help="The project root directory.",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("."),
    force: bool = typer.Option(
        False, "--force", help="Force initialization even if settings.yml already exists."
    ),
):
    """Create a default settings.yml file."""
    config_path = directory / "settings.yml"
    data_folder = directory / "data"

    data_folder.mkdir(parents=True,exist_ok=True)

    if config_path.exists() and not force:
        typer.echo(f"Settings file already exists at {config_path}. Use --force to overwrite.")
        raise typer.Exit()

    # Default settings
    default_settings = {
        "mod_expt_file": str(data_folder / "mod_expt.npy"),
        "mod_sld_file": str(data_folder / "mod_sld_fp49.npy"),
        "mod_params_file": str(data_folder / "mod_params_fp49.npy"),

        #hyperparameter settings
        "latent_dim":2,
        "batch_size":16,
        "ae_epochs":200,
        "mlp_epochs":200,
    }

    with open(config_path, "w") as f:
        yaml.dump(default_settings, f)

    typer.echo(f"Initialized settings file at {config_path}.")


# Command to run the data processing and model training then saving
@app.command("run")
def run_chi_pred_model_training(
    config: Annotated[
        Path,
        typer.Option(help="Path to the settings.yml file.", exists=True, readable=True),
    ] = Path("settings.yml"),
    enable_chi_prediction: Annotated[
        bool, typer.Option(help="Run Chi prediction.")
    ]=False,
    retrain: Annotated[
        bool, typer.Option(help="Retrain the model from scratch.")
    ] = False,
):
    """Run SLD data analysis for Chi params using the specified settings."""
    if not enable_chi_prediction or not config.exists():
        typer.echo("Error: Either a valid config file must be provided or Chi prediction must be enabled.")
        raise typer.Exit()

    """Run SLD data analysis for Chi params using the specified settings."""
    # Load settings from the YAML file
    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    # # Extract file paths from the settings
    # mod_expt_file = settings.get("mod_expt_file")
    # mod_sld_file = settings.get("mod_sld_file")
    # mod_params_file = settings.get("mod_params_file")
    # batch_size = settings.get("batch_size")
    # ae_epochs = settings.get("ae_epochs")

    # IMPORTANT: Required Setting params
    required_keys = {
        "mod_expt_file",
        "mod_sld_file",
        "mod_params_file",
        "batch_size",
        "latent_dim",
        "ae_epochs",
        "mlp_epochs",
    }
    missing_keys = required_keys - settings.keys()
    if missing_keys:
        typer.echo("Invalid settings file. Missing keys: {missing_keys}")
        raise typer.Exit()

    chi_pred_params = ChiPredTrainingParams(**{key: settings[key] for key in required_keys})

    if enable_chi_prediction:
        percep, autoencoder,data_processor = workflow.run_model_training(chi_pred_params)
        df_predictions = workflow.run_model_prediction(percep, autoencoder, data_processor.expt_arr,data_processor.sld_arr,data_processor.num_params)

        print("\nFinal Chi Prediction:")
        print(pd.DataFrame(df_predictions))


@app.command("predict")
def run_chi_model_prediction():
    pass