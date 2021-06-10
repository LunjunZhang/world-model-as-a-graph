import threading
import numpy as np
import torch
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        
        self.lock = threading.Lock()
    
    def update(self, v):
        v = v.reshape(-1, self.size)
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]
    
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count
    
    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(
            np.maximum(np.square(self.eps),
                       (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
    
    @staticmethod
    def _mpi_average(x):
        if MPI is None:
            return x
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf
    
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        if type(v) == np.ndarray:
            return np.clip((v - self.mean) / self.std, -clip_range, clip_range)
        else:
            mean_ = torch.as_tensor(self.mean).float().to(v.device)
            std_ = torch.as_tensor(self.std).float().to(v.device)
            return torch.clamp((v - mean_) / std_, -clip_range, clip_range)
    
    def state_dict(self):
        return {'total_sum': self.total_sum, 'total_sumsq': self.total_sumsq, 'total_count': self.total_count,
                'mean': self.mean, 'std': self.std}
    
    def load_state_dict(self, state_dict):
        self.total_sum = state_dict['total_sum']
        self.total_sumsq = state_dict['total_sumsq']
        self.total_count = state_dict['total_count']
        self.mean = state_dict['mean']
        self.std = state_dict['std']
