import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from rl.agent.core import Actor, Critic, BaseAgent
from rl.agent.core import LOG_STD_MIN, LOG_STD_MAX
from rl.agent.planner import DistCritic
from rl.agent.normalizer import Normalizer

from rl.utils import mpi_utils, net_utils


class ValueFunction(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        input_dim = 2 * env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1],
            activation=args.activ)
    
    def forward(self, inputs):
        v_values = self.net(inputs).squeeze()
        return v_values


class MlpEncoder(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        input_dim = env_params['goal']
        output_dim = args.embed_size
        self.net = net_utils.mlp(
            [input_dim] + [args.ae_hid_size] * args.ae_n_hids,
            activation=args.activ, output_activation=args.activ)
        self.encode = nn.Linear(args.ae_hid_size, output_dim)
    
    def forward(self, inputs):
        outputs = self.net(inputs)
        encode = self.encode(outputs)
        return encode


class MlpDecoder(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        input_dim = args.embed_size
        output_dim = env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.ae_hid_size] * args.ae_n_hids,
            activation=args.activ, output_activation=args.activ)
        self.decode = nn.Linear(args.ae_hid_size, output_dim)
    
    def forward(self, inputs):
        outputs = self.net(inputs)
        decode = self.decode(outputs)
        return decode


class MlpAutoEncoder(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.encoder = MlpEncoder(env_params, args)
        self.decoder = MlpDecoder(env_params, args)
    
    def forward(self, inputs):
        latent_code = self.encoder(inputs)
        reconstruct = self.decoder(latent_code)
        return latent_code, reconstruct


class Cluster(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.env_params = env_params
        self.n_mix = args.n_latent_landmarks
        self.z_dim = args.embed_size
        self.comp_mean = nn.Parameter(torch.randn(self.n_mix, self.z_dim) * np.sqrt(1.0 / self.n_mix))
        self.comp_logstd = nn.Parameter(torch.randn(1, self.z_dim) * 1 / np.e, requires_grad=True)
        self.mix_logit = nn.Parameter(torch.ones(self.n_mix), requires_grad=args.learned_prior)
    
    def component_log_prob(self, x):
        if x.ndim == 1:
            x = x.repeat(1, self.n_mix, 1)
        elif x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.n_mix, 1)
        assert x.ndim == 3 and x.size(1) == self.n_mix and x.size(2) == self.z_dim
        # comp_logstd = torch.sigmoid(self.comp_logstd) * (LOG_STD_MAX - LOG_STD_MIN) + LOG_STD_MIN
        comp_logstd = torch.clamp(self.comp_logstd, LOG_STD_MIN, LOG_STD_MAX)
        comp_dist = Normal(self.comp_mean, torch.exp(comp_logstd))
        comp_log_prob = comp_dist.log_prob(x).sum(dim=-1)  # (nbatch, n_mix)
        return comp_log_prob
    
    def forward(self, x, with_elbo=True):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2 and x.size(1) == self.z_dim
        log_mix_probs = torch.log_softmax(self.mix_logit, dim=-1).unsqueeze(0)  # (1, n_mix)
        assert log_mix_probs.size(0) == 1 and log_mix_probs.size(1) == self.n_mix
        
        prior_prob = torch.softmax(self.mix_logit, dim=0).unsqueeze(0)
        log_comp_probs = self.component_log_prob(x)  # (nbatch, n_mix)
        
        log_prob_x = torch.logsumexp(log_mix_probs + log_comp_probs, dim=-1, keepdim=True)  # (nbatch, 1)
        log_posterior = log_comp_probs + log_mix_probs - log_prob_x  # (nbatch, n_mix)
        posterior = torch.exp(log_posterior)
        if with_elbo:
            kl_from_prior = kl_divergence(Categorical(probs=posterior), Categorical(probs=prior_prob))
            return posterior, dict(
                comp_log_prob=log_comp_probs,
                log_data=(posterior * log_comp_probs).sum(dim=-1),
                kl_from_prior=kl_from_prior)
        else:
            return posterior
    
    def centroids(self):
        with torch.no_grad():
            return self.comp_mean.clone().detach()
    
    def circles(self):
        with torch.no_grad():
            return torch.exp(self.comp_logstd).clone().expand_as(self.comp_mean).detach()
    
    def std_mean(self):
        return torch.exp(self.comp_logstd).mean()
    
    def assign_centroids(self, x):
        self.comp_mean.data.copy_(x)


class Agent(BaseAgent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)
        
        self.actor = Actor(env_params, args)
        self.critic = DistCritic(env_params, args)
        
        if mpi_utils.use_mpi():
            mpi_utils.sync_networks(self.actor)
            mpi_utils.sync_networks(self.critic)
        
        self.actor_targ = Actor(env_params, args)
        self.critic_targ = DistCritic(env_params, args)
        
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
