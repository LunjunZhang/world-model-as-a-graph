import torch
import torch.nn as nn
import torch.nn.functional as F
from rl.utils import torch_utils


def get_activ(activ_name):
    if activ_name == 'tanh':
        return nn.Tanh
    elif activ_name == 'relu':
        return nn.ReLU
    elif activ_name == 'elu':
        return nn.ELU
    elif activ_name == 'leaky_relu':
        return nn.LeakyReLU
    elif activ_name == 'sigmoid':
        return nn.Sigmoid
    elif activ_name == 'identity':
        return nn.Identity
    else:
        raise NotImplementedError


def apply_activ(x, activ_name):
    if activ_name == 'tanh':
        return F.tanh(x)
    elif activ_name == 'relu':
        return F.relu(x)
    elif activ_name == 'elu':
        return F.elu(x)
    elif activ_name == 'leaky_relu':
        return F.leaky_relu(x)
    elif activ_name == 'sigmoid':
        return F.sigmoid(x)
    elif activ_name == 'identity':
        return x
    else:
        raise NotImplementedError


def mlp(sizes, activation='tanh', output_activation='identity'):
    layers = []
    n_layer = len(sizes) - 1
    if n_layer <= 0:
        return nn.Identity()
    for j in range(n_layer):
        if j == n_layer - 1:
            activ = get_activ(output_activation)
        else:
            activ = get_activ(activation)
        layers += [
            nn.Linear(sizes[j], sizes[j + 1]),
            activ()]
    return nn.Sequential(*layers)


def set_requires_grad(net, allow_grad=True):
    for param in net.parameters():
        param.requires_grad = allow_grad


def target_soft_update(source, target, polyak=1.0):
    with torch.no_grad():
        for p, p_targ in zip(source.parameters(), target.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def copy_model_params_from_to(source, target):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def mean_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.zeros(1).mean()
    device = parameters[0].grad.device
    mean_norm = torch.mean(torch.stack([torch.norm(p.grad.detach(),
                                                   norm_type).to(device) for p in parameters]))
    return mean_norm


def total_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.zeros(1).mean()
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                    norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def convert_to_2d_tensor(x):
    x = torch_utils.to_tensor(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x
