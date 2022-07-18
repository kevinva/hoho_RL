import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.99
LEARNING_RATE = 5e-5
EPOCH_NUM = 10