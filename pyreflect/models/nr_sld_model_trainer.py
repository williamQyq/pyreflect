from click.core import batch

from pyreflect.input import NRSLDDataProcessor

from .config import DEVICE
import torch
from .cnn import CNN

from .model_trainer import ModelTrainer

# Training parameters
LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
SPLIT_RATIO = 0.8

class NRSLDModelTrainer(ModelTrainer):
    def __init__(self, data_processor:NRSLDDataProcessor,X,y,layers, batch_size, epochs):
        super().__init__(data_processor, batch_size = batch_size, epochs = epochs)
        self.model = CNN(layers).to(DEVICE) #model
        self.X = X
        self.y = y
        self.data_processor = data_processor
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def train_pipeline(self):
        self.model.train()
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


