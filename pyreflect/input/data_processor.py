import numpy as np
from sklearn.model_selection import train_test_split
import torch


class DataProcessor:
    def __init__(self, seed=123):
        """
        Base class for processing data.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def split_arrays(self, X, y, size_split=0.8):
        """
        Splits the dataset into training, validation, and test sets.
        """
        crv_tr, crv_hld, chi_tr, chi_hld = train_test_split(X, y, train_size=size_split)
        crv_val, crv_tst, chi_val, chi_tst = train_test_split(crv_hld, chi_hld, test_size=0.5)

        return [crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst]

    def convert_tensors(self,list_arrays):
        ## Convert to tensors
        tensor_arrays = [torch.from_numpy(array).float() for array in list_arrays]
        return tensor_arrays

    def get_dataloaders(self, xtrain, ytrain, xval, yval, xtest, ytest, batch_size=32):
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


class NRSLDDataProcessor(DataProcessor):
    def __init__(self, nr_file_path=None, sld_file_path=None, seed=123):
        """
        Processes NR and SLD curve data.
        """
        super().__init__(seed)
        self.nr_file_path = nr_file_path
        self.sld_file_path = sld_file_path
        self.nr_arr = None
        self.sld_arr = None

    def load_data(self):
        """Loads NR and SLD data."""
        if self.nr_file_path:
            self.nr_arr = np.load(self.nr_file_path)
        if self.sld_file_path:
            self.sld_arr = np.load(self.sld_file_path)

        if self.nr_file_path is None and self.sld_file_path is None:
            raise ValueError("At least one of nr_file_path or sld_file_path must be provided.")


    def normalize(self, curves):
        """
        Normalizes curves using log transformation and min-max scaling.
        """
        assert curves is not None, "Curves data is not loaded."
        curves = np.log10(curves)

        x_points = [curve[0] for curve in curves]
        y_points = [curve[1] for curve in curves]

        min_x, max_x = np.min([np.min(x) for x in x_points]), np.max([np.max(x) for x in x_points])
        min_y, max_y = np.min([np.min(y) for y in y_points]), np.max([np.max(y) for y in y_points])

        for i in range(len(y_points)):
            x_points[i] = (x_points[i] - min_x) / (max_x - min_x)
            y_points[i] = (y_points[i] - min_y) / (max_y - min_y)

        return np.stack([x_points, y_points], axis=1)

    def normalize_nr(self):
        """Normalizes NR curves."""
        self.nr_arr = self.normalize(self.nr_arr)
        return self.nr_arr

    def normalize_sld(self):
        """Normalizes SLD curves."""
        self.sld_arr = self.normalize(self.sld_arr)
        return self.sld_arr

    def reshape_nr_to_single_channel(self):
        """
        Reshapes NR data to (batch_size, 1, sequence_length) for CNN input.

        Returns:
        - reshaped_nr (numpy.ndarray): Reshaped NR data.
        """
        if self.nr_arr is None:
            raise ValueError("NR data not loaded. Call `load_data()` first.")

        # Selecting the second channel (index 1) and reshaping to (batch_size, 1, sequence_length)
        reshaped_nr = self.nr_arr[:, 1][:, np.newaxis, :]
        return reshaped_nr