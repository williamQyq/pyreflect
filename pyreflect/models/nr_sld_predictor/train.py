import torch
from click.core import batch

from pyreflect.models.nr_sld_predictor.model import CNN
from pyreflect.input.data_processor import DataProcessor
from .trainer import train_model, validate_model
from .config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS
import numpy as np

def train_pipeline(curves_nr, curves_sld):
    # Data preparation
    processor = DataProcessor()
    R_m = curves_nr[:, 1][:, np.newaxis, :]
    list_arrays = processor.split_arrays(R_m, curves_sld, size_split=0.8)
    tensor_arrays = processor.convert_tensors(list_arrays)

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = processor.get_dataloaders(*tensor_arrays,batch_size=32)

    # Model initialization
    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        val_loss = validate_model(model, valid_loader, loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    return model
