import torch

GAMMA = 0.9
LR = 1e-4
GLOBAL_MAX_EPISODE = 5000
DEVICE = torch.device('cpu')  #torch.device("cuda" if torch.cuda.is_available() else "cpu")