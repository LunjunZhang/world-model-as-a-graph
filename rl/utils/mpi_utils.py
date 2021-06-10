try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import numpy as np
import torch


def use_mpi():
    return get_size() > 1


def is_root():
    return get_rank() == 0


def global_mean(x):
    if MPI is None:
        return x
    global_x = np.zeros_like(x)
    comm = MPI.COMM_WORLD
    comm.Allreduce(x, global_x, op=MPI.SUM)
    global_x /= comm.Get_size()
    return global_x


def global_sum(x):
    if MPI is None:
        return x
    global_x = np.zeros_like(x)
    comm = MPI.COMM_WORLD
    comm.Allreduce(x, global_x, op=MPI.SUM)
    return global_x


def bcast(x):
    if MPI is None:
        return x
    comm = MPI.COMM_WORLD
    comm.Bcast(x, root=0)
    return x


def get_rank():
    if MPI is not None:
        return MPI.COMM_WORLD.Get_rank()
    else:
        return 0


def get_size():
    if MPI is not None:
        return MPI.COMM_WORLD.Get_size()
    else:
        return 1


def sync_networks(network):
    if MPI is None:
        return
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network, scale_grad_by_procs=True):
    if MPI is None:
        return
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    if scale_grad_by_procs:
        global_grads /= comm.Get_size()
    _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
