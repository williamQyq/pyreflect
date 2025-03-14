from pyreflect.input import NRSLDDataProcessor

from .config import DEVICE
import torch
from .cnn import CNN
import numpy as np

# Training parameters
LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
SPLIT_RATIO = 0.8

class NRSLDModelTrainer:
    def __init__(self, data_processor:NRSLDDataProcessor,layers, batch_size, epochs):
        self.model = CNN(layers).to(DEVICE) #model
        self.X = data_processor.normalize_nr() # normalized nr curves
        self.y = data_processor.normalize_sld() # normalized sld curves
        self.data_processor = data_processor
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def train_pipeline(self):

        # remove wave vector(x channel) from nr
        R_m = self.data_processor.reshape_nr_to_single_channel(self.X)

        list_arrays = self.data_processor.split_arrays(R_m, self.y, size_split=SPLIT_RATIO)
        tensor_arrays = self.data_processor.convert_tensors(list_arrays)

        # train valid split
        train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = self.data_processor.get_dataloaders(
            *tensor_arrays, batch_size= self.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        for epoch in range(self.epochs):
            train_loss = self.train_model(self.model, train_loader, optimizer, loss_fn)
            val_loss = self.validate_model(self.model, valid_loader, loss_fn)
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        return self.model

    def train_model(self, model, train_loader, optimizer, loss_fn):
        model.train().to(DEVICE)
        total_loss = 0

        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_model(self, model, valid_loader, loss_fn):
        model.eval().to(DEVICE)
        total_loss = 0

        with torch.no_grad():
            for data, label in valid_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = model(data)
                loss = loss_fn(output, label)
                total_loss += loss.item()

        return total_loss / len(valid_loader)

