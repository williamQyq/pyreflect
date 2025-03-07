import yaml
from pathlib import Path
from typing import Annotated

import typer
import pyreflect.flows as workflow
import pandas as pd

from pyreflect.flows import predict_sld_from_nr
from pyreflect.models.config import ChiPredTrainingParams
INVALID_METHOD_ERROR = "Invalid method"

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
    # store all data files
    data_folder = directory / "data"
    #curves folder for nr sld training
    curves_folder = directory / "data"/ "curves"

    data_folder.mkdir(parents=True,exist_ok=True)
    curves_folder.mkdir(parents=True,exist_ok=True)

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
        "ae_epochs":20,
        "mlp_epochs":20,

        # SLD Prediction settings

        "nr_sld_model_file":str(data_folder / "trained_sld_model.pth"),
    }

    yaml_content = f"""\
    # 🛠 Configuration file for NR-SLD-Chi Predictor
    # Modify these paths according to your project structure.

    # 📂 Experimental SLD profile data file (input for Chi Prediction)
    mod_expt_file: {default_settings["mod_expt_file"]}

    # 📂 SLD Profile file (input for training)
    mod_sld_file: {default_settings["mod_sld_file"]}

    # 📂 Chi Parameters file (output label for training)
    mod_params_file: {default_settings["mod_params_file"]}

    # ⚙️ SLD-Chi Model hyperparameters
    latent_dim: {default_settings["latent_dim"]}  # Dimension for latent space
    batch_size: {default_settings["batch_size"]}  # Batch size for training
    ae_epochs: {default_settings["ae_epochs"]}  # Autoencoder training epochs
    mlp_epochs: {default_settings["mlp_epochs"]}  # MLP training epochs

    # 📁 Trained NR predict SLD model storage
    expt_nr_file:
    nr_sld_curves_poly:
    sld_curves_poly:
    nr_sld_model_file: {default_settings["nr_sld_model_file"]}
    """

    with open(config_path, "w") as f:
        f.write(yaml_content)

    typer.echo(f"Initialized settings file at {config_path}.")

# Command to run the data processing and model training then saving
@app.command("run")
def _run_cli(
    config: Annotated[
        Path,
        typer.Option(help="Path to the settings.yml file.", exists=True, readable=True),
    ] = Path("settings.yml"),
    enable_chi_prediction: Annotated[
        bool, typer.Option(help="Run Chi prediction.")
    ]=False,
    enable_sld_prediction: Annotated[
        bool, typer.Option(help="Run SLD prediction.")
    ]=False,

):
    """Run SLD data analysis for Chi params using the specified settings."""
    if not config.exists():
        typer.echo("Error: Either a valid config file must be provided or Chi prediction must be enabled.")
        raise typer.Exit()

    """Run SLD data analysis for Chi params using the specified settings."""
    # Load settings from the YAML file
    with open(config, "r") as f:
        settings = yaml.safe_load(f)

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

    chi_pred_params = ChiPredTrainingParams(
        **{key: settings[key] for key in required_keys}
    )

    #run Chi Prediction
    if enable_chi_prediction:
        typer.echo("Running Chi prediction...")
        percep, autoencoder,data_processor = workflow.train_autoencoder_mlp_chi_pred.run_model_training(chi_pred_params)
        df_predictions = workflow.sld_profile_pred_chi.run_model_prediction(percep, autoencoder, data_processor.expt_arr,data_processor.sld_arr,data_processor.num_params)

        typer.echo("\nFinal Chi Prediction:")
        typer.echo(pd.DataFrame(df_predictions))

    #Run SLD Prediction
    if enable_sld_prediction:
        typer.echo("\nRunning SLD Prediction...")

        # Load NR and SLD data
        nr_file = settings["NR-SLD_CurvesPoly"]
        sld_file = settings["SLD_CurvesPoly"]
        model_path = settings["nr_sld_model_file"]

        # Load experimental NR curves for inference
        expt_nr_file = settings["expt_nr_file"]

        model = None
        # Check if a trained SLD model exists, else train one
        if Path(model_path).exists():
            model = workflow.load_nr_sld_model(model_path)
            typer.echo("Loaded existing trained SLD model.")
        else:
            typer.echo("No trained SLD model found. Training a new model...")
            model = workflow.train_nr_predict_sld_model(nr_file, sld_file, to_be_saved_model_path= model_path)

            typer.echo(f"Trained SLD model saved at {model_path}")

        if not model:
            typer.echo("Model not loaded.")
            raise typer.Exit()


        predicted_sld = workflow.predict_sld_from_nr(model,expt_nr_file)
        print(predicted_sld)
        typer.echo("SLD Prediction complete!")

