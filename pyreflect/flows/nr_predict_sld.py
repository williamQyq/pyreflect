from pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator, ReflectivityModel
from pyreflect.input.data_processor import NRSLDDataProcessor
from pyreflect.models.config import DEVICE, NRSLDCurvesGeneratorParams, NRSLDModelTrainerParams
from pyreflect.models.cnn import CNN
from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer

import numpy as np
import torch
import typer
from pathlib import Path
from typing import Tuple

def generate_nr_sld_curves(params:NRSLDCurvesGeneratorParams)->Tuple[np.ndarray, np.ndarray]:

    """
        Generates and saves reflectivity and SLD curve data.

        Parameters:
        num_curves (int): Number of curves to generate per layer combination.
        dir (str): Directory where the files will be saved.

    """
    m = ReflectivityDataGenerator(model=ReflectivityModel(params.num_film_layers))
    processed_nr, processed_sld_profile = m.generate_curves(params.num_curves)

    np.save(params.mod_sld_file, processed_sld_profile)
    np.save(params.mod_nr_file, processed_nr)

    typer.echo(f"NR SLD generated curves saved at: \n\
               mod sld file: {params.mod_sld_file}\n\
                mod nr file: {params.mod_nr_file}")

    return processed_nr, processed_sld_profile

def load_nr_sld_model(model_path):
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    return model

def train_nr_predict_sld_model(params:NRSLDModelTrainerParams, auto_save=True)-> None:
    """
    :param params:
    :param auto_save:
    :return:
    :raise FileNotFoundError:
    """
    data_processor = NRSLDDataProcessor(params.nr_file,params.sld_file)
    data_processor.load_data()

    # NR-SLD curves are already normalized during generation
    trainer = NRSLDModelTrainer(
        data_processor=data_processor,
        X=data_processor._nr_arr,
        y=data_processor._sld_arr,
        layers=params.layers,
        batch_size=params.batch_size,
        epochs=params.epochs,
    )

    # training
    model = trainer.train_pipeline()

    # save model
    if auto_save:
        to_be_saved_model_path = params.model_path
        torch.save(model.state_dict(), to_be_saved_model_path)
        typer.echo(f"NR predict SLD trained CNN model saved at: {to_be_saved_model_path}")

    return model

def predict_sld_from_nr(model, nr_file:str | Path)->np.ndarray:
    """
        Predicts SLD profiles from given NR curves using a trained model.

        Args:
            model (torch.nn.Module): Trained PyTorch model.
            nr_file (str): Path to the NR file to process.

        Returns:
            np.ndarray: Predicted SLD curves.
    """
    try:
        # Load data
        processor = NRSLDDataProcessor(nr_file_path=nr_file)
        processor.load_data()
    except FileNotFoundError as e:
        typer.echo(e)
        raise typer.Exit()

    # Normalization
    normalized_nr_arr = processor.normalize_nr()

    #Remove wave vector (x channel) of NR
    reshaped_nr_curves = processor.reshape_nr_to_single_channel(normalized_nr_arr)

    #Prediction
    predicted_sld_curves = _predict(model, reshaped_nr_curves)

    typer.echo(f"Predicted SLD shape: {predicted_sld_curves.shape}")

    return predicted_sld_curves


def _predict(model, X_batch:np.ndarray)->np.ndarray:
    model.eval().to(DEVICE)
    X_batch = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y = model(X_batch)

    return y.cpu().numpy()