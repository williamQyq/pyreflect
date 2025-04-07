from pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator, ReflectivityModel
from pyreflect.input.data_processor import NRSLDDataProcessor, DataProcessor
from pyreflect.models.config import DEVICE, NRSLDCurvesGeneratorParams, NRSLDModelTrainerParams
from pyreflect.models.cnn import CNN
from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer

import numpy as np
import torch
import typer
import json
from pathlib import Path
from typing import Tuple

def generate_nr_sld_curves(params:NRSLDCurvesGeneratorParams)->Tuple[np.ndarray, np.ndarray]:
    """
    Generates and saves reflectivity and SLD curve data.

    :param params: NRSLDCurvesGeneratorParams
    :return: Tuple[np.ndarray, np.ndarray]: (nr,sld)
    """

    m = ReflectivityDataGenerator(num_layers=params.num_film_layers)

    processed_nr, processed_sld_profile = m.generate(params.num_curves)

    np.save(params.mod_sld_file, processed_sld_profile)
    np.save(params.mod_nr_file, processed_nr)

    typer.echo(f"NR SLD generated curves saved at: \n\
               mod sld file: {params.mod_sld_file}\n\
                mod nr file: {params.mod_nr_file}")

    return processed_nr, processed_sld_profile

def preprocess(dproc:NRSLDDataProcessor,norm_stats_save_path: str)->Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess function normalization and remove Q from NR.
    :param dproc: DataProcessor, get nr,sld from load data
    :return: nr, sld
    """
    normalized_sld = dproc.normalize_sld()
    normalized_nr = dproc.normalize_nr()
    norm_stats = dproc.get_normalization_stats()

    with open(norm_stats_save_path, 'w') as f:
        json.dump(norm_stats, f,indent=2)

    # Only keep reflectivity, remove Q
    reshaped = dproc.reshape_nr_to_single_channel(normalized_nr)

    return reshaped,normalized_sld

def load_nr_sld_model(model_path):
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    return model
def load_normalization_stat(norm_stat_path):
    """
    Load min max for normalization used in training from file.
    :param norm_stat_path:
    :return:
    """
    try:
        with open(norm_stat_path, "r") as f:
            norm_stats = json.load(f)
    except FileNotFoundError:
        print("Normalization stats file not found.")
    except json.JSONDecodeError:
        print("File is not valid JSON.")

    return norm_stats

def train_nr_predict_sld_model(reshaped_nr_curves, normalized_sld_curves, params:NRSLDModelTrainerParams, auto_save=True)-> torch.nn.Module:
    """
    Model training
    :param reshaped_nr_curves:
    :param normalized_sld_curves:
    :param params:
    :param auto_save: boolean
    :return: torch.nn.Module
    """

    trainer = NRSLDModelTrainer(
        X=reshaped_nr_curves,
        y=normalized_sld_curves,
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

def predict_sld_from_nr(model, nr_data:np.ndarray | Path | str, norm_stats)->np.ndarray:
    """
    Predicts SLD profiles from given NR curves using a trained model.

    :param model:torch.nn.Module: trained CNN model
    :param nr_data: np.ndarray | str: NR data
    :param norm_stats: dict: normalization status
    :return: np.ndarray: SLD profiles
    """
    norm_stats_nr = norm_stats["nr"]
    norm_stats_sld = norm_stats["sld"]
    try:
        if isinstance(nr_data, (str,Path)):
            processor = NRSLDDataProcessor(nr_file_path=nr_data).load_data()
            norm_nr = processor.normalize_nr(norm_stats_nr)
        elif isinstance(nr_data, np.ndarray):
            norm_nr,_ = DataProcessor.normalize_xy_curves(nr_data,apply_log=True,min_max_stats=norm_stats_nr)
            processor = NRSLDDataProcessor()
        else:
            raise TypeError("nr_data must be a file path (str) or a NumPy array.")
    except FileNotFoundError as e:
        typer.echo(f"Error loading NR data: {e}")
        raise typer.Exit()

    # Remove wave vector (x channel) of NR
    reshaped = processor.reshape_nr_to_single_channel(norm_nr)

    y = _predict(model, reshaped)

    predicted_sld_curves = processor.denormalize(
        y,
        curve_type='sld',
        min_max_stats=norm_stats_sld)

    typer.echo(f"Predicted SLD shape: {predicted_sld_curves.shape}")

    return predicted_sld_curves

def _predict(model, X_batch:np.ndarray)->np.ndarray:
    model.eval().to(DEVICE)
    X_batch = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y = model(X_batch)

    return y.cpu().numpy()