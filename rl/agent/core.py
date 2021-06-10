import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rl.utils import mpi_utils, net_utils
from rl.agent.normalizer import Normalizer

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class StochasticActor(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']
        
        input_dim = env_params['obs'] + env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids,
            activation=args.activ, output_activation=args.activ)
        self.mean = nn.Linear(args.hid_size, env_params['action'])
        self.logstd = nn.Linear(args.hid_size, env_params['action'])
    
    def gaussian_params(self, inputs):
        outputs = self.net(inputs)
        mean, logstd = self.mean(outputs), self.logstd(outputs)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logstd)
        return mean, std
    
    def forward(self, inputs, deterministic=False, with_logprob=True):
        mean, std = self.gaussian_params(inputs)
        pi_dist = Normal(mean, std)
        if deterministic:
            pi_action = mean
        else:
            pi_action = pi_dist.rsample()
        logp_pi = None
        if with_logprob:
            logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi


class Qfunc(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.q_func = net_utils.mlp([input_dim] + [args.hid_size] * args.n_hids + [1], activation=args.activ)
    
    def forward(self, *args):
        q_value = self.q_func(torch.cat([*args], dim=-1))
        return torch.squeeze(q_value, -1)


class DoubleQfunc(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.q1 = Qfunc(env_params, args)
        self.q2 = Qfunc(env_params, args)
    
    def forward(self, *args):
        q1 = self.q1(*args)
        q2 = self.q2(*args)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']
        
        input_dim = env_params['obs'] + env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids,
            activation=args.activ, output_activation=args.activ)
        self.mean = nn.Linear(args.hid_size, env_params['action'])
    
    def forward(self, inputs):
        outputs = self.net(inputs)
        mean = self.mean(outputs)
        pi_action = torch.tanh(mean) * self.act_limit
        return pi_action


class Critic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']
        
        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1],
            activation=args.activ)
    
    def forward(self, pi_inputs, actions):
        q_inputs = torch.cat([pi_inputs, actions / self.act_limit], dim=-1)
        q_values = self.net(q_inputs).squeeze()
        return q_values


class BaseAgent:
    def __init__(self, env_params, args, name='agent'):
        self.env_params = env_params
        self.args = args
        self._save_file = str(name) + '.pt'
    
    @staticmethod
    def to_2d(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x
    
    def to_tensor(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.args.cuda:
            x = x.cuda()
        return x
    
    @property
    def device(self):
        return torch.device("cuda" if self.args.cuda else "cpu")
    
    def get_actions(self, obs, goal):
        raise NotImplementedError
    
    def get_pis(self, obs, goal):
        raise NotImplementedError
    
    def get_qs(self, obs, goal, actions):
        raise NotImplementedError
    
    def forward(self, obs, goal, *args, **kwargs):
        """ return q_pi, pi """
        raise NotImplementedError
    
    def target_update(self):
        raise NotImplementedError
    
    def state_dict(self):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    def save(self, path):
        if mpi_utils.is_root():
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


class Agent(BaseAgent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)
        
        self.actor = Actor(env_params, args)
        self.critic = Critic(env_params, args)
        
        if mpi_utils.use_mpi():
            mpi_utils.sync_networks(self.actor)
            mpi_utils.sync_networks(self.critic)
        
        self.actor_targ = Actor(env_params, args)
        self.critic_targ = Critic(env_params, args)
        
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())
        
        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic_targ, allow_grad=False)
        
        if self.args.cuda:
            self.cuda()
        
        self.o_normalizer = Normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_normalizer = Normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
    
    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()
        self.actor_targ.cuda()
        self.critic_targ.cuda()
    
    def _clip_inputs(self, x):
        if type(x) == np.ndarray:
            return np.clip(x, -self.args.clip_obs, self.args.clip_obs)
        else:
            return torch.clamp(x, -self.args.clip_obs, self.args.clip_obs)
    
    @staticmethod
    def _concat(x, y):
        assert type(x) == type(y)
        if type(x) == np.ndarray:
            return np.concatenate([x, y], axis=-1)
        else:
            return torch.cat([x, y], dim=-1)
    
    def _preprocess_inputs(self, obs, goal):
        obs = self.to_2d(obs)
        goal = self.to_2d(goal)
        if self.args.clip_inputs:
            obs = self._clip_inputs(obs)
            goal = self._clip_inputs(goal)
        return obs, goal
    
    def _process_inputs(self, obs, goal):
        if self.args.normalize_inputs:
            obs = self.o_normalizer.normalize(obs)
            goal = self.g_normalizer.normalize(goal)
        return self.to_tensor(self._concat(obs, goal))
    
    def get_actions(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions
    
    def get_pis(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        pis = self.actor(inputs)
        return pis
    
    def get_qs(self, obs, goal, actions):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        return self.critic(inputs, actions)
    
    def forward(self, obs, goal, q_target=False, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        q_net = self.critic_targ if q_target else self.critic
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return q_net(inputs, pis), pis
    
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic, target=self.critic_targ, polyak=self.args.polyak)
    
    def normalizer_update(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        self.o_normalizer.update(obs)
        self.g_normalizer.update(goal)
        self.o_normalizer.recompute_stats()
        self.g_normalizer.recompute_stats()
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'actor_targ': self.actor_targ.state_dict(),
                'critic': self.critic.state_dict(), 'critic_targ': self.critic_targ.state_dict(),
                'o_normalizer': self.o_normalizer.state_dict(), 'g_normalizer': self.g_normalizer.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_targ.load_state_dict(state_dict['critic_targ'])
        self.o_normalizer.load_state_dict(state_dict['o_normalizer'])
        self.g_normalizer.load_state_dict(state_dict['g_normalizer'])
