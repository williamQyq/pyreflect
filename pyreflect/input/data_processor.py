from typing import List
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, seed=123):
        """
        Base class for processing data.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def split_arrays(X, y, size_split=0.8):
        """
        Splits the dataset into training, validation, and test sets.
        """
        crv_tr, crv_hld, chi_tr, chi_hld = train_test_split(X, y, train_size=size_split)
        crv_val, crv_tst, chi_val, chi_tst = train_test_split(crv_hld, chi_hld, test_size=0.5)

        return [crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst]

    @staticmethod
    def convert_tensors(list_arrays):
        ## Convert to tensors
        tensor_arrays = [torch.from_numpy(array).float() for array in list_arrays]

        return tensor_arrays

    @staticmethod
    def get_dataloaders(xtrain, ytrain, xval, yval, xtest, ytest, batch_size=32):
        """
        Converts split datasets into PyTorch dataloaders.
        """
        train_dataset = torch.utils.data.TensorDataset(xtrain,ytrain)
        valid_dataset = torch.utils.data.TensorDataset(xval,yval)
        test_dataset = torch.utils.data.TensorDataset(xtest,ytest)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return [train_dataset,valid_dataset,test_dataset,train_loader, valid_loader, test_loader]

    @staticmethod
    def normalize_xy_curves(data: List | np.ndarray, scale=(-1,1), apply_log=False)->np.stack:
        """
          Normalizes data (shape [samples,channel, features] to a given scale ([-1, 1] or [0, 1]).

          Args:
              data (np.ndarray): Input array (reflectivity R(q) or SLD).
              scale (tuple): Target normalization range (default `[-1,1]`, set `(0,1)` if needed).
              apply_log (bool): Whether to apply `log10` transformation (useful for reflectivity R(q)).

          Returns:
              np.stack: Normalized data.
        """
        # Convert to NumPy array if not already
        data = np.array(data, dtype=np.float32)

        if apply_log:
            data = np.log10(data)

        assert data.ndim == 3, f"Trying to normalize each channel independently but the data shape is {data.shape}."
        scalers = [MinMaxScaler(feature_range=scale)for _ in range(data.shape[1])]

        for I in range(data.shape[1]):
            data[:,I,:] = scalers[I].fit_transform(data[:,I,:])

        return data

class SLDChiDataProcessor(DataProcessor):
    def __init__(self, expt_sld_file_path, sld_file_path, chi_params_file_path, seed=123):
        """
        Processes SLD and Chi parameter data.
        """
        super().__init__(seed)
        self.sld_file_path = sld_file_path
        self.chi_params_file_path = chi_params_file_path
        self.expt_sld_file_path = expt_sld_file_path
        self.sld_arr = None
        self.expt_arr = None
        self.params_arr = None
        self.num_params = None

    def load_data(self):
        """Loads SLD and Chi parameter data."""
        self.sld_arr = np.load(self.sld_file_path)
        self.params_arr = np.load(self.chi_params_file_path)
        self.expt_arr = np.load(self.expt_sld_file_path)
        self.num_params = self.params_arr.shape[1]

    @classmethod
    def remove_flat_samples(cls, data_arr):
        # Remove flat data
        flat_data = []
        for i in range(data_arr.shape[0]):
            y_start = data_arr[i, 1, 0]
            if data_arr[i, 1, 1] == y_start and data_arr[i, 1, 2] == y_start:
                flat_data.append(i)

        return np.delete(data_arr,flat_data,axis=0), flat_data

    def preprocess_data(self):
        """Cleans and normalizes data by removing flat and non-impact data."""

        assert self.sld_arr is not None or self.params_arr is not None and self.expt_arr is not None, "data not loaded."

        # Remove flat data
        self.sld_arr,flat_data = self.remove_flat_samples(self.sld_arr)
        self.params_arr = np.delete(self.params_arr, flat_data, 0)

        # Remove non-impact data (chi1 range filter)
        bad_chi1 = [i for i in range(self.sld_arr.shape[0]) if not (0.07 <= self.params_arr[i, 0] <= 0.12)]
        self.sld_arr = np.delete(self.sld_arr, bad_chi1, 0)
        self.params_arr = np.delete(self.params_arr, bad_chi1, 0)

        # Batch normalize chi parameters
        for i in range(self.params_arr.shape[1]):
            min_val, max_val = self.params_arr[:, i].min(), self.params_arr[:, i].max()
            if max_val > min_val:
                self.params_arr[:, i] = ((self.params_arr[:, i] - min_val) * 2 / (max_val - min_val)) - 1

        return self.sld_arr, self.params_arr

class NRSLDDataProcessor(DataProcessor):
    def __init__(self, nr_file_path=None, sld_file_path=None, seed=123):
        """
        Processes NR and SLD curve data.
        """
        super().__init__(seed)
        self.nr_file_path = nr_file_path
        self.sld_file_path = sld_file_path
        self._nr_arr = None
        self._sld_arr = None

    def load_data(self,new_nr_file=None, new_sld_file=None):
        """

        :param new_nr_file:
        :param new_sld_file:
        :return:

        :raise
        - OSError
        """
        # Decide file paths: use new ones if provided, else use those from init
        nr_path = new_nr_file if new_nr_file is not None else self.nr_file_path
        sld_path = new_sld_file if new_sld_file is not None else self.sld_file_path

        if nr_path:
            self._nr_arr = np.load(nr_path)

        if sld_path:
            self._sld_arr = np.load(sld_path)

    def normalize_nr(self):
        """Normalizes NR curves."""
        if self._nr_arr is None:
            raise FileNotFoundError(f"NR file not loaded from path:{self.nr_file_path}")

        # Reflectivity decreases exponentially, log transformation compress large range
        return DataProcessor.normalize_xy_curves(self._nr_arr,scale=(0,1),apply_log=True)

    def normalize_sld(self):
        """Normalizes SLD curves."""
        if self._sld_arr is None:
            raise FileNotFoundError(f"SLD File not loaded from path: {self.sld_file_path}")

        return DataProcessor.normalize_xy_curves(self._sld_arr,scale=(0,1),apply_log=True)

    def reshape_nr_to_single_channel(self,nr_data:np.ndarray)->np.ndarray:
        """
        Reshapes NR data to (batch_size, 1, sequence_length) for CNN input.
        Remove the momentum q range(y-axis).

        Returns:
        - reshaped_nr (numpy.ndarray): Reshaped NR data.
        """
        if not isinstance(nr_data, np.ndarray):
            raise TypeError("nr_data must be a numpy array.")

        if nr_data.ndim != 3 or nr_data.shape[1] != 2:
            raise ValueError(f"Expected input shape (batch_size, 2, sequence_length), got {nr_data.shape}")

        # Selecting the second channel (index 1) and reshaping to (batch_size, 1, sequence_length)
        reshaped_nr = nr_data[:, 1][:, np.newaxis, :]
        return reshaped_nr