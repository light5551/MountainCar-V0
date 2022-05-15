import torch.nn as nn
import numpy as np

unchanged_values_to_freeze = 3
unfrozen_iter = 1
norms = None


def check_condition(norms: list):
    eps = 10 ** -2
    for i in range(len(norms)-1):
        if np.abs(norms[i] - norms[i+1]) > eps:
            return False
    return True


def sp_loss_for_layers(train_layer, ref_layer):
    diff = train_layer.cpu().detach().numpy() - ref_layer.cpu().detach().numpy()
    norm = np.linalg.norm(diff)
    return norm


def freeze_unchanged(train_model: nn.Sequential, ref_model: nn.Sequential, step, target_update):
    if step % target_update != 0:
        return
    global norms, unfrozen_iter
    train_params = enumerate(train_model.parameters())
    ref_params = list(ref_model.parameters())
    for i, param in train_params:
        if not param.requires_grad or i >= unfrozen_iter:
            continue
        norm = sp_loss_for_layers(param, ref_params[i])
        norms[i].append(norm)
        window = norms[i][-unchanged_values_to_freeze-1:-1]
        if check_condition(window) and len(window) == unchanged_values_to_freeze:
            param.requires_grad = False
            unfrozen_iter += 1
            print('Freezing layer ', i)
    print()


def get_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
    global norms
    unfrozen_count = len(list(model.parameters()))
    norms = [list() for _ in range(unfrozen_count)]
    return model
