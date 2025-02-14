import yaml
from pathlib import Path
from typing import Annotated

import typer
import pyreflect.flows as workflow

import pandas as pd

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
    ] = Path("settings.yml")
):
    """Run SLD data analysis for Chi params using the specified settings."""
    # Load settings from the YAML file
    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    # Extract file paths from the settings
    mod_expt_file = settings.get("mod_expt_file")
    mod_sld_file = settings.get("mod_sld_file")
    mod_params_file = settings.get("mod_params_file")
    batch_size = settings.get("batch_size")

    # IMPORTANT: Required Setting params
    required_keys = {"mod_expt_file", "mod_sld_file", "mod_params_file", "batch_size"}

    if not required_keys.issubset(settings):
        typer.echo("Invalid settings file. Ensure all file paths are specified.")
        raise typer.Exit()


    percep, autoencoder,data_processor = workflow.run_model_training(**{key: settings[key] for key in required_keys })
    df_predictions = workflow.run_model_prediction(percep, autoencoder, data_processor.expt_arr,data_processor.sld_arr,data_processor.num_params)

    print(pd.DataFrame(df_predictions))

@app.command("predict")
def run_chi_model_prediction():
    pass