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

    def split_arrays(self, X, Y, size_split=0.8):
        """
        Splits the dataset into training, validation, and test sets.
        """
        split1 = int(len(X) * size_split)
        split2 = int(len(X) * (size_split + (1 - size_split) / 2))

        xtrain, ytrain = X[:split1], Y[:split1]
        xval, yval = X[split1:split2], Y[split1:split2]
        xtest, ytest = X[split2:], Y[split2:]

        return xtrain, ytrain, xval, yval, xtest, ytest

    def get_dataloaders(self, xtrain, ytrain, xval, yval, xtest, ytest, batch_size=32):
        """
        Converts split datasets into PyTorch dataloaders.
        """
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(xtrain, dtype=torch.float32),
                                                       torch.tensor(ytrain, dtype=torch.float32))
        valid_dataset = torch.utils.data.TensorDataset(torch.tensor(xval, dtype=torch.float32),
                                                       torch.tensor(yval, dtype=torch.float32))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(xtest, dtype=torch.float32),
                                                      torch.tensor(ytest, dtype=torch.float32))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

class SLDChiDataProcessor(DataProcessor):
    def __init__(self, sld_file_path, chi_params_file_path, seed=123):
        """
        Processes SLD and Chi parameter data.
        """
        super().__init__(seed)
        self.sld_file_path = sld_file_path
        self.chi_params_file_path = chi_params_file_path
        self.sld_arr = None
        self.params_arr = None
        self.num_params = None

    def load_data(self):
        """Loads SLD and Chi parameter data."""
        self.sld_arr = np.load(self.sld_file_path)
        self.params_arr = np.load(self.chi_params_file_path)

        self.num_params = self.params_arr.shape[1]

    def preprocess_data(self):
        """Cleans and normalizes the SLD and Chi parameter data."""
        assert self.sld_arr is not None and self.params_arr is not None, "Data not loaded."

        # Remove flat data
        flat_data = [i for i in range(self.sld_arr.shape[0])
                     if self.sld_arr[i, 1, 1] == self.sld_arr[i, 1, 0] == self.sld_arr[i, 1, 2]]
        self.sld_arr = np.delete(self.sld_arr, flat_data, 0)
        self.params_arr = np.delete(self.params_arr, flat_data, 0)

        # Normalize parameters
        for i in range(self.params_arr.shape[1]):
            min_val, max_val = self.params_arr[:, i].min(), self.params_arr[:, i].max()
            if max_val > min_val:
                self.params_arr[:, i] = ((self.params_arr[:, i] - min_val) * 2 / (max_val - min_val)) - 1


class NRSLDDataProcessor(DataProcessor):
    def __init__(self, nr_file_path, sld_file_path, seed=123):
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
        self.nr_arr = np.load(self.nr_file_path)
        self.sld_arr = np.load(self.sld_file_path)

    def normalize(self, curves):
        """
        Normalizes curves using log transformation and min-max scaling.
        """
        assert curves is not None, "Curves data not loaded."
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

    def normalize_sld(self):
        """Normalizes SLD curves."""
        self.sld_arr = self.normalize(self.sld_arr)
