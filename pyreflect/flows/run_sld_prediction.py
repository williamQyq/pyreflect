from pathlib import Path
from .nr_predict_sld import load_nr_sld_model, train_nr_predict_sld_model, generate_nr_sld_curves, predict_sld_from_nr

from pyreflect.models.config import NRSLDCurvesGeneratorParams,NRSLDModelTrainerParams
import typer

def run_sld_prediction(root:Path,config:dict):
    config["root"] = root
    model_path = root / config["nr_predict_sld"]["models"]["model"]

    # Train Model and save if no model found
    if not model_path.exists():
        typer.echo("No trained SLD model found. Training a new model...")

        #Generate curves for model training and save generated data to folder
        try:
            typer.echo("Generating Curves for training...")
            generator_params = NRSLDCurvesGeneratorParams(_config=config)
            generate_nr_sld_curves(generator_params)

            #Init params for model training
            typer.echo("Training model using curves...")
            trainer_params = NRSLDModelTrainerParams(_config=config)
            #Train and save
            train_nr_predict_sld_model(trainer_params, auto_save=True)

        except Exception as e:
            typer.echo(e)
            raise typer.Exit()

    # Load model
    model = load_nr_sld_model(model_path)
    typer.echo("Loaded trained SLD model.")

    # Perform prediction
    experimental_nr_file = root / config["nr_predict_sld"]["file"]["experimental_nr_file"]
    if not experimental_nr_file.exists():
        msg = f"⚠️  Error: The specified experimental NR file does not exist.\n"
        msg += "Please update the configuration and provide a valid NR file for prediction."
        raise FileNotFoundError(msg)

    typer.echo("Running SLD Prediction...")
    predicted_sld = predict_sld_from_nr(model, experimental_nr_file)

    return predicted_sld
