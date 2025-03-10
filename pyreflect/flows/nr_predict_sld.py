from pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator
from pyreflect.input.data_processor import NRSLDDataProcessor
from pyreflect.models.config import NRSLDCurvesGeneratorParams, NRSLDModelTrainerParams
from pyreflect.models.nr_sld_predictor.config import DEVICE
from pyreflect.models.nr_sld_predictor.inference import predict_sld
from pyreflect.models.nr_sld_predictor.train import train_pipeline
from pyreflect.models.nr_sld_predictor.model import CNN

import numpy as np
import torch
import typer

def generate_nr_sld_curves(params:NRSLDCurvesGeneratorParams)-> None:

    """
        Generates and saves reflectivity and SLD curve data.

        Parameters:
        num_curves (int): Number of curves to generate per layer combination.
        dir (str): Directory where the files will be saved.

    """
    for first in range(1, 6):
        for second in range(1, 6):
            # print(first, second)
            m = ReflectivityDataGenerator(first, second)
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
    data_processor = NRSLDDataProcessor(params.nr_file,params.sld_file)
    data_processor.load_data()

    # train model
    model = train_pipeline(data_processor.nr_arr, data_processor.sld_arr)

    if auto_save:
    # save model
        to_be_saved_model_path = params.model_path
        torch.save(model.state_dict(), to_be_saved_model_path)
        typer.echo(f"NR predict SLD trained CNN model saved at: {to_be_saved_model_path}")
    return model

def predict_sld_from_nr(model, nr_file):
    processor = NRSLDDataProcessor(nr_file_path=nr_file)
    processor.load_data()
    print(f"nr curves: {processor.nr_arr}")

    processor.normalize_nr()
    reshaped_nr_curves = processor.reshape_nr_to_single_channel()

    predicted_sld_curves = [predict_sld(model, curve) for curve in reshaped_nr_curves]
    return predicted_sld_curves