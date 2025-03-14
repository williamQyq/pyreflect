from pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator
from pyreflect.input.data_processor import NRSLDDataProcessor
from pyreflect.models.config import DEVICE, NRSLDCurvesGeneratorParams, NRSLDModelTrainerParams
from pyreflect.models.cnn import CNN
from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer

import numpy as np
import torch
import typer
from pathlib import Path

def generate_nr_sld_curves(params:NRSLDCurvesGeneratorParams)-> None:

    """
        Generates and saves reflectivity and SLD curve data.

        Parameters:
        num_curves (int): Number of curves to generate per layer combination.
        dir (str): Directory where the files will be saved.

    """

    m = ReflectivityDataGenerator()
    m.generate(params.num_curves)
    pars, train_data = m.get_preprocessed_data()

    for index in range(len(m._smooth_array)):
        min_x = min(m._smooth_array[index][0])
        for i in range(len(m._smooth_array[index][0])):
            m._smooth_array[index][0][i] -= min_x

    settingUp, SLDSet = [], []
    for i in range(len(m._smooth_array)):
        settingUp.append(np.array([m.q, m._refl_array[i]]))
        SLDSet.append(np.array(m._smooth_array[i]))

    totalStack = np.stack(settingUp) # processed NR
    totalParams = np.stack(SLDSet)  # processed SLD

    print(f"processed NR shape:{totalStack.shape}\n")
    print(f"processed SLD shape:{totalParams.shape}")

    np.save(params.mod_sld_file, totalParams)
    np.save(params.mod_nr_file, totalStack)

    typer.echo(f"NR SLD generated curves saved at: \n\
               mod sld file: {params.mod_sld_file}\n\
                mod nr file: {params.mod_nr_file}")
    return None

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

    trainer = NRSLDModelTrainer(
        data_processor=data_processor,
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
        processor = NRSLDDataProcessor(nr_file_path=nr_file)
        # load data into nr_arr
        processor.load_data()

    except FileNotFoundError as e:
        typer.echo(e)
        raise typer.Exit()

    # Normalization
    normalized_nr_arr = processor.normalize_nr()

    typer.echo(f"Processed NR shape:{normalized_nr_arr.shape}\n")

    #Remove wave vector (x channel) of NR
    reshaped_nr_curves = processor.reshape_nr_to_single_channel(normalized_nr_arr)

    # Stack all curves into a batch for efficient model inference
    reshaped_nr_curves = np.stack(reshaped_nr_curves)

    #Prediction
    predicted_sld_curves = _predict(model, reshaped_nr_curves)

    typer.echo(f"Prediction SLD shape: {predicted_sld_curves.shape}")

    return predicted_sld_curves


def _predict(model, X_batch:np.ndarray)->np.ndarray:
    model.eval().to(DEVICE)
    X_batch = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y = model(X_batch)

    return y.cpu().numpy()