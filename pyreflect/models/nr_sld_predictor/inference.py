import torch
import numpy as np
from .config import DEVICE
from .model import CNN

def predict_sld(model, nr_curve):
    model.eval().to(DEVICE)
    nr_curve = torch.tensor(nr_curve, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predicted_sld = model(nr_curve)

    return predicted_sld.cpu().numpy()
