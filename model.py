import torch.nn as nn
import numpy as np


def sp_loss_for_layers(train_layer, ref_layer):
    diff = train_layer.cpu().detach().numpy() - ref_layer.cpu().detach().numpy()
    norm = np.linalg.norm(diff)
    return norm


def freeze_unchanged(train_model: nn.Sequential, ref_model: nn.Sequential):
    train_params = enumerate(train_model.parameters())
    ref_params = list(ref_model.parameters())
    for i, param in train_params:
        norm = sp_loss_for_layers(param, ref_params[i])
        print(norm)


def get_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
