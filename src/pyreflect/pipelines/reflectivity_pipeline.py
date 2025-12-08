from src.pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator
from src.pyreflect.input.data_processor import NRSLDDataProcessor, DataProcessor
from src.pyreflect.config.runtime import DEVICE
from src.pyreflect.models.config import NRSLDCurvesGeneratorParams, NRSLDModelTrainerParams
from src.pyreflect.models.cnn import CNN
from src.pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer

import numpy as np
import torch
import typer
from pathlib import Path
from typing import Tuple

# TODO: finish the pipeline design
class ReflectivityPipeline:
    def __init__(self, data_strategy, preprocess_strategy, trainer, predictor):
        self.data_strategy = data_strategy
        self.preprocess_strategy = preprocess_strategy
        self.trainer = trainer
        self.predictor = predictor

    def prepare_dataset(self):
        curves = self.data_strategy.generate()
        return self.preprocess_strategy.apply(*curves)

    def train(self, dataset):
        return self.trainer.train(dataset)

    def predict(self, model, nr_curves):
        processed = self.preprocess_strategy.apply_nr_only(nr_curves)
        return self.predictor.predict(model, processed)

    # def generate_training_data(self,layer_desc = None, layer_bound = None):
    #     data_generator = ReflectivityDataGenerator(
    #         num_layers=self.data_gen_params.num_film_layers,
    #         layer_desc=layer_desc,
    #         layer_bound=layer_bound,)
    #
    #     syn_nr, syn_sld = data_generator.generate(self.data_gen_params.num_curves)
    #     np.save(self.data_gen_params.mod_sld_file,syn_sld)
    #     np.save(self.data_gen_params.mod_nr_file,syn_nr)
    #
    #     typer.echo(f"Synthetic data NR SLD generated and saved at: \n\
    #                    mod sld file: {self.data_gen_params.mod_sld_file}\n\
    #                     mod nr file: {self.data_gen_params.mod_nr_file}")
    #
    #     return syn_nr, syn_sld


def preprocess(dproc:NRSLDDataProcessor,norm_stats_to_be_save_path: str)->Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess function normalization and remove Q from NR.
    :param dproc: DataProcessor, get nr,sld from load data
    :return: nr, sld
    """
    normalized_sld = dproc.normalize_sld()
    normalized_nr = dproc.normalize_nr()
    norm_stats = dproc.get_normalization_stats()

    np.save(norm_stats_to_be_save_path, norm_stats, allow_pickle=True)

    # Only keep reflectivity, remove Q
    reshaped = dproc.reshape_nr_to_single_channel(normalized_nr)

    return reshaped,normalized_sld

def load_normalization_stat(norm_stat_path)->dict:
    """
    Load min max for normalization used in training from file.
    :param norm_stat_path:
    :return: min max dict of nr and sld
    """
    return np.load(norm_stat_path, allow_pickle=True).item()

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
        dropout=params.dropout,
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