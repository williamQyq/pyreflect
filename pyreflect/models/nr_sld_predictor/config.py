import torch

# General settings
SEED = 123
torch.manual_seed(SEED)

# Training parameters
LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Data split ratio
SPLIT_RATIO = 0.9
