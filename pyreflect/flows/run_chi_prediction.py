from .sld_profile_pred_chi import run_model_prediction
from .train_autoencoder_mlp_chi_pred import run_model_training
from pyreflect.models.config import ChiPredTrainingParams
import typer
import pandas as pd

def run_chi_prediction(root,config):

    # Extract required parameters else missing key error.
    try:
        config["root"] = root
        chi_pred_params = ChiPredTrainingParams(_config=config)
    except Exception as e:
        typer.echo(e)
        raise typer.Exit()

    mlp, autoencoder, data_processor = run_model_training(chi_pred_params)
    df_predictions = run_model_prediction(mlp,
                                          autoencoder,
                                          data_processor.expt_arr,
                                          data_processor.sld_arr,
                                          data_processor.num_params)

    typer.echo("\nFinal Chi Prediction:")
    typer.echo(pd.DataFrame(df_predictions))
