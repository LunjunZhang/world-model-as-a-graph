import numpy as np

import torch
import torch.nn as nn

from rl.agent.core import Actor, Critic, BaseAgent
from rl.agent.normalizer import Normalizer
from rl.agent.latent_planner import ValueFunction, MlpAutoEncoder, Cluster

from rl.utils import mpi_utils, net_utils


class DistCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self._dist_func = Critic(env_params, args)
        self.args = args
        self.gamma = args.gamma
    
    def forward(self, pi_inputs, actions):
        dist = self._dist_func(pi_inputs, actions)
        log_gamma = np.log(self.gamma)
        return - (1 - torch.exp(dist * log_gamma)) / (1 - self.gamma)
    
    def get_dist(self, pi_inputs, actions):
        dist = self._dist_func(pi_inputs, actions)
        if self.args.q_offset:
            dist += 1.0
        return dist


class DistReverseCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self._q_func = Critic(env_params, args)
        self.args = args
        self.gamma = args.gamma
    
    def forward(self, pi_inputs, actions):
        q_value = self._q_func(pi_inputs, actions)
        return q_value
    
    def get_dist(self, pi_inputs, actions):
        q_value = self._q_func(pi_inputs, actions)
        log_gamma = np.log(self.gamma)
        dist = torch.log(1. + q_value * (1. - self.gamma)) / log_gamma
        if self.args.q_offset:
            dist += 1.0
        return dist


class Agent(BaseAgent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)
        
        self.actor = Actor(env_params, args)
        critic_func = DistCritic
        if args.use_reverse_dist_func:
            critic_func = DistReverseCritic
        self.critic = critic_func(env_params, args)
        
        if mpi_utils.use_mpi():
            mpi_utils.sync_networks(self.actor)
            mpi_utils.sync_networks(self.critic)
        
        self.actor_targ = Actor(env_params, args)
        self.critic_targ = critic_func(env_params, args)
        
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())
        
        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic_targ, allow_grad=False)
        
        self.vf = ValueFunction(env_params, args)
        self.ae = MlpAutoEncoder(env_params, args)
        self.cluster = Cluster(env_params, args)
        
        if self.args.cuda:
            self.cuda()
        
        self.o_normalizer = Normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_normalizer = Normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
    
    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()
        self.actor_targ.cuda()
        self.critic_targ.cuda()
        self.vf.cuda()
        self.ae.cuda()
        self.cluster.cuda()
    
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
        qs = self.critic(inputs, actions)
        return qs
    
    def forward(self, obs, goal, q_target=False, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        q_net = self.critic_targ if q_target else self.critic
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        qs = q_net(inputs, pis)
        return qs, pis
    
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic, target=self.critic_targ, polyak=self.args.polyak)
    
    def normalizer_update(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        self.o_normalizer.update(obs)
        self.g_normalizer.update(goal)
        self.o_normalizer.recompute_stats()
        self.g_normalizer.recompute_stats()
    
    def pairwise_value(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        pis = self.actor(inputs)
        dist = self.critic.get_dist(inputs, pis)  # this is positive
        return -dist
    
    def get_dists(self, obs, goal, actions):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        dist = self.critic.get_dist(inputs, actions)  # this is positive
        return dist
    
    def _process_vf_inputs(self, ag, bg):
        ag = self.to_2d(ag)
        bg = self.to_2d(bg)
        if self.args.clip_inputs:
            ag = self._clip_inputs(ag)
            bg = self._clip_inputs(bg)
        if self.args.normalize_inputs:
            ag = self.g_normalizer.normalize(ag)
            bg = self.g_normalizer.normalize(bg)
        return self.to_tensor(self._concat(ag, bg))
    
    def get_vf_value(self, ag, bg):
        inputs = self._process_vf_inputs(ag, bg)
        vf_value = self.vf(inputs)
        return vf_value
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'actor_targ': self.actor_targ.state_dict(),
                'critic': self.critic.state_dict(), 'critic_targ': self.critic_targ.state_dict(),
                'vf': self.vf.state_dict(), 'ae': self.ae.state_dict(), 'cluster': self.cluster.state_dict(),
                'o_normalizer': self.o_normalizer.state_dict(), 'g_normalizer': self.g_normalizer.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_targ.load_state_dict(state_dict['critic_targ'])
        self.vf.load_state_dict(state_dict['vf'])
        self.ae.load_state_dict(state_dict['ae'])
        self.cluster.load_state_dict(state_dict['cluster'])
        self.o_normalizer.load_state_dict(state_dict['o_normalizer'])
        self.g_normalizer.load_state_dict(state_dict['g_normalizer'])
