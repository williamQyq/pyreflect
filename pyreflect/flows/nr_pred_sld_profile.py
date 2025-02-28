from pyreflect.data_processor.reflectivity_data_generator import ReflectivityDataGenerator
import numpy as np
import os

def generate_data_files(self, num_curves,dir="data/NR-SLD"):

    """
        Generates and saves reflectivity and SLD curve data.

        Parameters:
        num_curves (int): Number of curves to generate per layer combination.
        dir (str): Directory where the files will be saved.
    """
    for first in range(1, 6):
        for second in range(1, 6):
            print(first, second)
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

            totalStack = np.stack(settingUp)
            totalParams = np.stack(SLDSet)
            print(totalStack.shape, totalParams.shape)

            np.save(os.path.join(dir,f"SLD_CurvesPoly{first}{second}.npy"), totalParams)
            np.save(os.path.join(dir,f"NR-SLD_CurvesPoly{first}{second}.npy"), totalStack)
