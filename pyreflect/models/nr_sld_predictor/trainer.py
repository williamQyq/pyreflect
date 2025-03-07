from pyreflect.models.config import DEVICE
import torch

def train_model(model, train_loader, optimizer, loss_fn):
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

def validate_model(model, valid_loader, loss_fn):
    model.eval().to(DEVICE)
    total_loss = 0

    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            loss = loss_fn(output, label)
            total_loss += loss.item()

    return total_loss / len(valid_loader)
