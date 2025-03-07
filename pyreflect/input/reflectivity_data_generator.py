import numpy as np
import os
import random
from scipy.interpolate import CubicSpline
from refl1d.names import QProbe,Slab,SLD,Parameter,Experiment

class ReflectivityDataGenerator:
    def __init__(self, first, second, q=None, name='polymer'):

        """
        Initialize the Reflectivity Data Generator.

        Parameters:
        first (int): Controls the thickness and roughness range of the bottom layer.
        second (int): Controls the thickness and roughness range of the bulk layer.
        q (array, optional): Custom q-values for the reflectivity calculation.
        name (str): Identifier for the dataset.
        """
        self.parameters = [
            dict(i=0, par='roughness', bounds=[0, 5]),
            dict(i=1, par='sld', bounds=[3.0, 3.9]),
            dict(i=1, par='thickness', bounds=[10, 25]),
            dict(i=1, par='roughness', bounds=[1, 10]),
            dict(i=2, par='sld', bounds=[0, 6.4]),
            dict(i=2, par='thickness', bounds=[first * 10, 300]),
            dict(i=2, par='roughness', bounds=[(first - 1) * 10, first * 10]),
            dict(i=3, par='sld', bounds=[0, 6.4]),
            dict(i=3, par='thickness', bounds=[10 * second, 300]),
            dict(i=3, par='roughness', bounds=[10 * (second - 1), 10 * second]),
        ]
        self.model_description = dict(
            layers=[
                dict(sld=2.07, isld=0, thickness=0, roughness=3, name='substrate'),
                dict(sld=3.3, isld=0, thickness=16.2139, roughness=7, name='siox'),
                dict(sld=1.82, isld=0, thickness=random.randint(first * 10, 300),
                     roughness=random.randint((first - 1) * 10, first * 10), name='bottom'),
                dict(sld=3.83, isld=0, thickness=random.randint(second * 10, 300),
                     roughness=random.randint((second - 1) * 10, second * 10), name='bulk'),
                dict(sld=0.0, isld=0, thickness=0, roughness=0, name='air')
            ],
            scale=1,
            background=0,
        )
        self._pars_array = []
        self._refl_array = []
        self._smooth_array = []
        self._train_pars = []
        self._train_data = None
        self._config_name = name

        self.q = q if q is not None else np.logspace(np.log10(0.0081), np.log10(0.1975), num=308)

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

    def calculate_reflectivity(self, q_resolution=0.0294855):
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
        zNew = np.linspace(z[0], z[-1], num=900)
        sldNew = CubicSpline(z, sld)(zNew)
        return self.model_description['scale'] * r, zNew, sldNew

    def generate(self, n=100):
        self._train_pars = np.random.uniform(low=-1, high=1, size=[n, len(self.parameters)])
        self.compute_reflectivity()

    def compute_reflectivity(self):
        print("Computing reflectivity")
        self._pars_array = self.to_model_parameters(self._train_pars)
        for p in self._pars_array:
            _desc = self.get_model_description(p)
            r, z, sld = self.calculate_reflectivity()
            self._refl_array.append(r)
            self._smooth_array.append([z, sld])

    def to_model_parameters(self, pars):
        pars_array = np.zeros(pars.shape)
        for i, par in enumerate(self.parameters):
            a = (par['bounds'][1] - par['bounds'][0]) / 2.
            b = (par['bounds'][1] + par['bounds'][0]) / 2.
            pars_array.T[i] = pars.T[i] * a + b
        return pars_array

    def get_model_description(self, pars):
        for i, par in enumerate(self.parameters):
            self.model_description['layers'][par['i']][par['par']] = pars[i]
        return self.model_description

    def get_preprocessed_data(self, errors=None):
        """
            Pre-process data
            If errors is provided, a random error will be added, taking the errors array
            as a relative uncertainty.
        """
        if errors is None:

            self._train_data = np.log10(self._refl_array * self.q ** 2 / self.q[0] ** 2)
        else:
            _data = self._refl_array * (1.0 + np.random.normal(size=len(errors)) * errors)
            # Catch the few cases where we generated a negative intensity and take
            # the absolute value
            _data[_data < 0] = np.fabs(_data[_data < 0])
            self._train_data = np.log10(_data * self.q ** 2 / self.q[0] ** 2)

        return self._train_pars, self._train_data

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

