import threading
import numpy as np
import torch
import os.path as osp

from rl.utils import mpi_utils


def sample_her_transitions(buffer, reward_func, batch_size, future_step, future_p=1.0):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)
    
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None)
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch


def sample_transitions(buffer, batch_size):
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    return batch


class Replay:
    def __init__(self, env_params, args, reward_func, name='replay'):
        self.env_params = env_params
        self.args = args
        self.reward_func = reward_func
        
        self.horizon = env_params['max_timesteps']
        self.size = args.buffer_size // self.horizon
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['action'])))
        
        self.lock = threading.Lock()
        self._save_file = str(name) + '_' + str(mpi_utils.get_rank()) + '.pt'
    
    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.reward_func, batch_size,
                                             future_step=self.args.future_step,
                                             future_p=self.args.future_p)
        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
