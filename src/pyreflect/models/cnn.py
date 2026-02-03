import torch
from torch import nn
import torch
import torch.nn as nn

# Custom SpatialDropout1D for 1D CNN
# It drops entire channels (feature maps) instead of individual elements
class SpatialDropout1D(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(3)             # Reshape from (B, C, T) to (B, C, T, 1)
        x = super().forward(x)         # Apply 2D dropout across (C, T)
        x = x.squeeze(3)               # Reshape back to (B, C, T)
        return x

class CNN(nn.Module):
    def __init__(self, layers=12,dropout_prob = 0.5):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        addition = 255 / layers
        curr = 1
        spatial_every=2

        for i in range(layers - 1):
            self.layers.append(nn.Conv1d(int(curr + 0.5), int(curr + addition + 0.5), 51, padding=25))
            self.layers.append(nn.BatchNorm1d(int(curr + addition + 0.5)))
            self.layers.append(nn.ReLU(True))

            if spatial_every is not None and (i % spatial_every == 1):
                self.layers.append(SpatialDropout1D(p=dropout_prob))
            else:
                self.layers.append(nn.Dropout(dropout_prob))
            curr += addition

        self.layers.append(nn.Conv1d(int(curr + 0.5), 256, 51, padding=25))
        self.layers.append(nn.BatchNorm1d(256))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Dropout(dropout_prob))

        self.linear1 = nn.Linear(256 * 308, 900 * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = x.reshape(-1, 2, 900)
        # (x, 2, 2000)
        return torch.sigmoid(x)