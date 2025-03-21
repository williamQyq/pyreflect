from typing import List

import numpy as np
import os

import typer
from pyreflect.input import NRSLDDataProcessor, DataProcessor
from scipy.interpolate import CubicSpline
from refl1d.names import QProbe,Slab,SLD,Parameter,Experiment

class ReflectivityModel:
    def __init__(self,num_layers, q=None, name='polymer'):
        """
        Initialize the reflectivity physical model with a dynamic number of layers.
        :param num_layers:
        :param q: The momentum transfer range (Q-range).
        :param name:
        """
        default_q_range = np.logspace(np.log10(0.008101436040354381), np.log10(0.1975709062238298), num=308)
        self.q = q if q is not None else default_q_range
        self.num_layers = num_layers

        #Substrate base layer
        self.parameters = [
            dict(i=0, par='roughness', bounds=[1.0,2.0])    #Substrate roughness
        ]

        #Middle layers
        for i in range(1, num_layers + 1):
            self.parameters.extend([
                dict(i=i, par='sld', bounds=[1.0, 8.0]),       # Covers Graphite, Si, LFP, NMC, Metals
                dict(i=i, par='thickness', bounds=[10, 300]),  # Battery films can be thicker
                dict(i=i, par='roughness', bounds=[2, 50])     # Rougher surfaces in solid-state batteries
            ])

        # The actual model
        self.model_description = {
            "layers": [
                dict(sld=2.07, isld=0, thickness=0, roughness=2.0, name='substrate')  # Metal base (e.g., Cu, Ni, Al)
            ],
            "scale": 1,
            "background": 0
        }

        for i in range(1, num_layers + 1):
            self.model_description["layers"].append(
                dict(sld=np.random.uniform(1.0, 8.0), isld=0,
                     thickness=np.random.uniform(10, 300), roughness=np.random.uniform(2, 50),
                     name=f'layer_{i}')
            )

        # Add an air layer at the end
        self.model_description["layers"].append(dict(sld=0, isld=0, thickness=0, roughness=0, name='air'))

        self._pars_array = []
        self._refl_array = []  # NR curves
        self._smooth_array = [] #SLD profile
        self._config_name = name

    @classmethod
    def from_dict(cls, pars, q_array=None):
        """
            Create ReflectivityModels object from a dict that
            defines the reflectivity model parameters and how
            the training set should be generated.
        """
        m = cls(q_array, name=pars['name'])
        m.model_description = pars['model']
        m.parameters = pars['parameters']
        return m

    def compute_reflectivity(self,train_pars:np.ndarray):
        print("Computing reflectivity...")
        self._pars_array = self.to_model_parameters(train_pars)

        from tqdm.auto import tqdm
        for p in tqdm(self._pars_array,desc="Processing reflectivity curves", colour="green"):
            _desc = self.get_model_description(p)
            r, z, sld = self._calculate_reflectivity()
            self._refl_array.append(r)
            self._smooth_array.append([z, sld])

    def _calculate_reflectivity(self, q_resolution=0.0294855):
        zeros = np.zeros(len(self.q))
        dq = q_resolution * self.q / 2.355
        probe = QProbe(self.q, dq, data=(zeros, zeros))

        layers = self.model_description['layers']
        sample = Slab(material=SLD(name=layers[0]['name'], rho=layers[0]['sld']), interface=layers[0]['roughness'])
        for l in layers[1:]:
            sample = sample | Slab(material=SLD(name=l['name'], rho=l['sld'], irho=l['isld']), thickness=l['thickness'],
                                   interface=l['roughness'])

        probe.background = Parameter(value=self.model_description['background'], name='background')
        expt = Experiment(probe=probe, sample=sample)

        q, r = expt.reflectivity()
        z, sld, _ = expt.smooth_profile()

        #Set num data points per SLD curves
        zNew = np.linspace(z[0], z[-1], num=900)
        sldNew = CubicSpline(z, sld)(zNew)
        return self.model_description['scale'] * r, zNew, sldNew

    def get_model_description(self, pars):
        """
            Convert the parameter list to a model description that we can use
            to compute R(q).
        """
        for i, par in enumerate(self.parameters):
            self.model_description['layers'][par['i']][par['par']] = pars[i]
        return self.model_description

    def to_model_parameters(self, pars:np.ndarray):
        """
            Transform an array of parameters to a list of calculable models
        """
        pars_array = np.zeros(pars.shape)

        for i, par in enumerate(self.parameters):
            a = (par['bounds'][1] - par['bounds'][0]) / 2.
            b = (par['bounds'][1] + par['bounds'][0]) / 2.
            pars_array.T[i] = pars.T[i] * a + b

        return pars_array

    def preprocess_nr(self, noise=None):
        """
            Pre-process data
            If noise is provided, a random error will be added, taking the errors array
            as a relative uncertainty.
        """

        if noise is None:
            normalized_nr = np.log10(self._refl_array * self.q ** 2 / self.q[0] ** 2)
            return normalized_nr

        _data = self._refl_array * (1.0 + np.random.normal(size=len(noise)) * noise)
        # Catch the few cases where we generated a negative intensity and take
        # the absolute value
        _data[_data < 0] = np.fabs(_data[_data < 0])
        normalized_nr = np.log10(_data * self.q ** 2 / self.q[0] ** 2)

        return normalized_nr

    @property
    def refl_array(self):
        return self._refl_array

    @property
    def smooth_array(self):
        return self._smooth_array


class ReflectivityDataGenerator:
    def __init__(self, model:ReflectivityModel):
        #Simulated physical model
        self.model = model

        self._train_pars = [] #randomized physical properties of the layers

    def generate_curves(self, n:int):
        """
        Generate n number of NR curves and sld profile
        :param n:
        :return: processed nr, processed sld profile
        """
        self._train_pars = np.random.uniform(low=-1, high=1, size=[n, len(self.model.parameters)])
        self.model.compute_reflectivity(self._train_pars)

        if self.model.refl_array is None:
            raise ValueError("No reflectivity data generated")

        norm_nr = self.model.preprocess_nr()
        norm_sld = DataProcessor.normalize_xy_curves(self.model.smooth_array, scale=(0, 1))

        refl_arr = np.stack([np.array([self.model.q, refl]) for refl in norm_nr])
        print(f"processed NR shape:{refl_arr.shape}\n")
        print(f"processed SLD shape:{norm_sld.shape}")

        return refl_arr, norm_sld

    def save(self, output_dir=''):
        np.save(os.path.join(output_dir, f"{self._config_name}_q_values"), self.q)
        if self._train_data is not None:
            np.save(os.path.join(output_dir, f"{self._config_name}_data"), self._train_data)
            np.save(os.path.join(output_dir, f"{self._config_name}_pars"), self._train_pars)

    def load(self, data_dir=''):
        self.q = np.load(os.path.join(data_dir, f"{self._config_name}_q_values.npy"))
        self._train_data = np.load(os.path.join(data_dir, f"{self._config_name}_data.npy"))
        self._train_pars = np.load(os.path.join(data_dir, f"{self._config_name}_pars.npy"))
        return self.q, self._train_data, self._train_pars

