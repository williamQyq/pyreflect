from pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator
from pyreflect.input.data_processor import NRSLDDataProcessor
from pyreflect.models.nr_sld_predictor.config import DEVICE
from pyreflect.models.nr_sld_predictor.inference import predict_sld
from pyreflect.models.nr_sld_predictor.train import train_pipeline
from pyreflect.models.nr_sld_predictor.model import CNN

import numpy as np
import torch
import os
from pathlib import Path

def generate_nr_sld_curves(num_curves,curves_dir):

    """
        Generates and saves reflectivity and SLD curve data.

        Parameters:
        num_curves (int): Number of curves to generate per layer combination.
        dir (str): Directory where the files will be saved.

    """
    folder = Path(curves_dir)
    if not folder.exists() or not folder.is_dir() :
        raise FileNotFoundError(f"{folder} does not exist or is not a directory")

    for first in range(1, 6):
        for second in range(1, 6):
            # print(first, second)
            m = ReflectivityDataGenerator(first, second)
            m.generate(num_curves)
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

            generated_sld_curves_file =os.path.join(curves_dir,f"SLD_CurvesPoly{first}{second}.npy")
            generated_nr_curves_file = os.path.join(curves_dir,f"NR-SLD_CurvesPoly{first}{second}.npy")
            np.save(generated_sld_curves_file, totalParams)
            np.save(generated_nr_curves_file, totalStack)

            return generated_nr_curves_file, generated_sld_curves_file

def load_nr_sld_model(model_path):
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    return model

def train_nr_predict_sld_model(nr_file,sld_file,to_be_saved_model_path):
    data_processor = NRSLDDataProcessor(nr_file,sld_file)
    data_processor.load_data()

    # train model
    model = train_pipeline(data_processor.nr_arr, data_processor.sld_arr)
    # save model
    torch.save(model.state_dict(), to_be_saved_model_path)
    print(f"NR predict SLD trained CNN model saved at: {to_be_saved_model_path}")
    return model

def predict_sld_from_nr(model, nr_file):
    processor = NRSLDDataProcessor(nr_file_path=nr_file)
    processor.load_data()
    processor.normalize_nr()
    reshaped_nr_curves = processor.reshape_nr_to_single_channel()

    predicted_sld_curves = [predict_sld(model, curve) for curve in reshaped_nr_curves]
    return predicted_sld_curves