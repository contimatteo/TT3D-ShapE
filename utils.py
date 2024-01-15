import torch

# ###


def get_cuda_device() -> torch.cuda.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cuda_is_available() -> bool:
    return torch.cuda.is_available()
