import numpy as np
from sklearn.model_selection import train_test_split
import torch

class DataProcessor:
    def __init__(self,expt_file_path, sld_file_path,chi_params_file_path, seed=123):
        self.sld_file_path = sld_file_path
        self.chi_params_file_path = chi_params_file_path
        self.expt_file_path = expt_file_path

        self.sld_arr = None
        self.params_arr = None
        self.expt_arr = None

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

    def load_data(self):
        """Loads SLD profile, experimental, and parameter data."""
        self.expt_arr = np.load(self.expt_file_path)
        self.sld_arr = np.load(self.sld_file_path)
        self.params_arr = np.load(self.chi_params_file_path)

        assert self.sld_arr.shape[1:] == (2, 72), f"Unexpected sld_arr shape: {self.sld_arr.shape}. Expected (*, 2, 72)"
        assert self.params_arr.shape[1] == 3, f"Unexpected params_arr shape: {self.params_arr.shape}. Expected (*, 3)"

    def preprocess_data(self):
        """Cleans and normalizes data by removing flat and non-impact data."""

        assert self.sld_arr is not None or self.params_arr is not None and self.expt_arr is not None, "data not loaded."

        # Remove flat data
        flat_data = []
        for i in range(self.sld_arr.shape[0]):
            y_start = self.sld_arr[i, 1, 0]
            if self.sld_arr[i, 1, 1] == y_start and self.sld_arr[i, 1, 2] == y_start:
                flat_data.append(i)

        self.sld_arr = np.delete(self.sld_arr, flat_data, 0)
        self.params_arr = np.delete(self.params_arr, flat_data, 0)

        # Remove non-impact data (chi1 range filter)
        bad_chi1 = [i for i in range(self.sld_arr.shape[0]) if not (0.07 <= self.params_arr[i, 0] <= 0.12)]
        self.sld_arr = np.delete(self.sld_arr, bad_chi1, 0)
        self.params_arr = np.delete(self.params_arr, bad_chi1, 0)

        # Normalize parameters
        for i in range(self.params_arr.shape[1]):
            min_val, max_val = self.params_arr[:, i].min(), self.params_arr[:, i].max()
            if max_val > min_val:
                self.params_arr[:, i] = ((self.params_arr[:, i] - min_val) * 2 / (max_val - min_val)) - 1

    def get_data_tensors(self):
        """Converts the data to PyTorch tensors."""
        sld_flat = self.sld_arr.reshape(self.sld_arr.shape[0], -1)
        sld_tensor = torch.from_numpy(sld_flat).float()
        params_tensor = torch.from_numpy(self.params_arr).float()
        return sld_tensor, params_tensor

    # Split combined curve (x,y) data and chi parameters data into 6 sets:
    # A training, validation, and testing for both curves and chi parameters
    # Takes in a set of curve data and a set of same-indexed chi parameter data
    # Default to 80% of data used for training
    def split_arrays(self, size_split=0.8):
        sld_tensor, params_tensor = self.get_data_tensors()
        crv_tr, crv_hld, chi_tr, chi_hld = train_test_split(sld_tensor, params_tensor, train_size=size_split, random_state=42)
        crv_val, crv_tst, chi_val, chi_tst = train_test_split(crv_hld, chi_hld, test_size=0.5, random_state=42)

        return [crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst]

    # Turn all 3 pairs of data arrays into pytorch tensors and dataloaders to feed a model
    # Also sets batch size
    def get_dataloaders(self, crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst, batch_size=32):
        """Creates PyTorch dataloaders from tensors."""

        tr_set = torch.utils.data.TensorDataset(crv_tr, chi_tr)
        tr_load = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True)

        val_set = torch.utils.data.TensorDataset(crv_val, chi_val)
        val_load = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        tst_set = torch.utils.data.TensorDataset(crv_tst, chi_tst)
        tst_load = torch.utils.data.DataLoader(tst_set, batch_size=batch_size, shuffle=True)

        return tr_set, val_set, tst_set, tr_load, val_load, tst_load