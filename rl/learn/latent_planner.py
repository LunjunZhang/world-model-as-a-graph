import torch
from torch.optim import Adam

import os.path as osp
from rl.utils import mpi_utils
from rl.utils import net_utils

from rl.learn.core import dict_to_numpy


class Learner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim = Adam(agent.critic.parameters(), lr=args.lr_critic)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        self.v_optim = Adam(agent.vf.parameters(), lr=args.lr_critic)
        self.ae_optim = Adam(agent.ae.parameters(), lr=args.lr_ae)
        self.c_optim = Adam(agent.cluster.parameters(), lr=args.lr_cluster)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            q_next, _ = self.agent.forward(o2, bg, q_target=True, pi_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_bg = self.agent.get_qs(o, bg, a)
        loss_q = (q_bg - q_targ).pow(2).mean()
        
        q_ag2 = self.agent.get_qs(o, ag2, a)
        loss_ag2 = q_ag2.pow(2).mean()
        
        q_future = self.agent.get_qs(o, future_ag, a)
        
        loss_critic = loss_q
        
        self.monitor.store(
            Loss_q=loss_q.item(),
            Loss_ag2=loss_ag2.item(),
            Loss_critic=loss_critic.item(),
        )
        monitor_log = dict(
            q_targ=q_targ,
            q_bg=q_bg,
            q_ag2=q_ag2,
            q_future=q_future,
            offset=offset,
            r=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic
    
    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        
        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward(o, bg)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2
        
        pi_future = self.agent.get_pis(o, future_ag)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            Loss_actor=loss_actor.item(),
            Loss_action_l2=action_l2.item(),
            Loss_bc=loss_bc.item(),
        )
        monitor_log = dict(q_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    
    def value_loss(self, batch):
        o, a, o2, bg = batch['ob'], batch['a'], batch['o2'], batch['bg']
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            dist_bg = self.agent.get_dists(o, bg, a)
        v_bg = self.agent.get_vf_value(ag2, bg)  # important: use ag2 (rather than ag)
        loss_vf = (v_bg - dist_bg).pow(2).mean()
        
        self.monitor.store(
            Loss_vf=loss_vf.item(),
            Diff_dist_offset=(dist_bg - offset).pow(2).mean().item(),
            Diff_vf_offset=(v_bg - offset).pow(2).mean().item(),
        )
        monitor_log = dict(dist_bg=dist_bg, v_bg=v_bg)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_vf
    
    def ae_loss(self, batch):
        bg, ag = batch['bg'], batch['ag']
        
        with torch.no_grad():
            v_forward = self.agent.get_vf_value(ag, bg)
            v_backward = self.agent.get_vf_value(bg, ag)
            v_target = 0.5 * (v_forward + v_backward)
        
        ag = self.agent.to_tensor(ag)
        bg = self.agent.to_tensor(bg)
        latent_ag, recon_ag = self.agent.ae(ag)
        latent_bg, recon_bg = self.agent.ae(bg)
        zdist = (latent_ag - latent_bg).pow(2)
        if self.args.embed_op == 'mean':
            zdist = zdist.mean(dim=-1)
        elif self.args.embed_op == 'sum':
            zdist = zdist.sum(dim=-1)
        else:
            raise NotImplementedError
        assert zdist.ndim == v_target.ndim == 1
        
        loss_recon = 0.5 * (recon_ag - ag).pow(2).mean() + 0.5 * (recon_bg - bg).pow(2).mean()
        loss_zdist = (zdist - v_target).pow(2).mean()
        loss_ae = loss_recon + self.args.latent_repel * loss_zdist
        
        self.monitor.store(
            Loss_reconstruct=loss_recon.item(),
            Loss_zdist=loss_zdist.item(),
            Loss_ae=loss_ae.item(),
        )
        monitor_log = dict(zdist=zdist, )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_ae
    
    def update(self, batch, train_embed=True):
        loss_critic = self.critic_loss(batch)
        self.q_optim.zero_grad()
        loss_critic.backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm = torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic=c_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_critic=net_utils.mean_grad_norm(self.agent.critic.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic.parameters(), self.args.grad_value_clipping)
        if mpi_utils.use_mpi():
            mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.q_optim.step()
        
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        if mpi_utils.use_mpi():
            mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
        self.pi_optim.step()
        
        if train_embed:
            loss_vf = self.value_loss(batch)
            self.v_optim.zero_grad()
            loss_vf.backward()
            
            if self.args.grad_norm_clipping > 0.:
                vf_norm = torch.nn.utils.clip_grad_norm_(self.agent.vf.parameters(), self.args.grad_norm_clipping)
                self.monitor.store(gradnorm_vf=vf_norm)
            if self.args.grad_value_clipping > 0.:
                self.monitor.store(gradnorm_mean_vf=net_utils.mean_grad_norm(self.agent.vf.parameters()).item())
                torch.nn.utils.clip_grad_value_(self.agent.vf.parameters(), self.args.grad_value_clipping)
            if mpi_utils.use_mpi():
                mpi_utils.sync_grads(self.agent.vf, scale_grad_by_procs=True)
            self.v_optim.step()
            
            loss_ae = self.ae_loss(batch)
            self.ae_optim.zero_grad()
            loss_ae.backward()
            
            if self.args.grad_norm_clipping > 0.:
                ae_norm = torch.nn.utils.clip_grad_norm_(self.agent.ae.parameters(), self.args.grad_norm_clipping)
                self.monitor.store(gradnorm_ae=ae_norm)
            if self.args.grad_value_clipping > 0.:
                self.monitor.store(gradnorm_mean_ae=net_utils.mean_grad_norm(self.agent.ae.parameters()).item())
                torch.nn.utils.clip_grad_value_(self.agent.ae.parameters(), self.args.grad_value_clipping)
            if mpi_utils.use_mpi():
                mpi_utils.sync_grads(self.agent.ae, scale_grad_by_procs=True)
            self.ae_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def embed_loss(self, embedding):
        posterior, elbo = self.agent.cluster(embedding, with_elbo=True)
        log_data = elbo['log_data']
        kl_from_prior = elbo['kl_from_prior']
        if self._has_nan(log_data) or self._has_nan(kl_from_prior):
            pass
        loss_elbo = - (log_data - self.args.elbo_beta * kl_from_prior).mean()
        std_mean = self.agent.cluster.std_mean()
        loss_std = self.args.cluster_std_reg * std_mean
        loss_embed_total = loss_elbo + loss_std
        self.monitor.store(
            Loss_elbo=loss_elbo.item(),
            Loss_cluster_std=loss_std.item(),
            Loss_embed_total=loss_embed_total.item(),
        )
        monitor_log = dict(
            Cluster_log_data=log_data,
            Cluster_kl=kl_from_prior,
            Cluster_post_std=posterior.std(dim=-1),
            Cluster_std_mean=std_mean,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_embed_total
    
    def update_cluster(self, batch_ag, to_train=True):
        assert type(batch_ag) == torch.Tensor
        with torch.no_grad():
            batch_embed = self.agent.ae.encoder(batch_ag)
        self.c_optim.zero_grad()
        loss_embed = self.embed_loss(batch_embed)
        if to_train:
            loss_embed.backward()
            if self.args.grad_norm_clipping > 0.:
                cluster_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.cluster.parameters(), self.args.grad_norm_clipping)
                self.monitor.store(gradnorm_cluster=cluster_norm)
            if self.args.grad_value_clipping > 0.:
                self.monitor.store(gradnorm_mean_cluster=net_utils.mean_grad_norm(
                    self.agent.cluster.parameters()).item())
                torch.nn.utils.clip_grad_value_(self.agent.cluster.parameters(), self.args.grad_value_clipping)
            if mpi_utils.use_mpi():
                mpi_utils.sync_grads(self.agent.cluster, scale_grad_by_procs=True)
            self.c_optim.step()
        else:
            loss_embed.backward()
            if self.args.grad_norm_clipping > 0.:
                cluster_norm = net_utils.total_grad_norm(self.agent.cluster.parameters())
                self.monitor.store(gradnorm_cluster=cluster_norm.item())
            if self.args.grad_value_clipping > 0.:
                self.monitor.store(gradnorm_mean_cluster=net_utils.mean_grad_norm(
                    self.agent.cluster.parameters()).item())
            self.c_optim.zero_grad()
    
    def initialize_cluster(self, batch_ag):
        assert type(batch_ag) == torch.Tensor and batch_ag.size(0) == self.agent.cluster.n_mix
        with torch.no_grad():
            batch_embed = self.agent.ae.encoder(batch_ag)
        self.agent.cluster.assign_centroids(batch_embed)
        if mpi_utils.use_mpi():
            mpi_utils.sync_networks(self.agent.cluster)
    
    def state_dict(self):
        return dict(
            q_optim=self.q_optim.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
            v_optim=self.v_optim.state_dict(),
            ae_optim=self.ae_optim.state_dict(),
            c_optim=self.c_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim.load_state_dict(state_dict['q_optim'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
        self.v_optim.load_state_dict(state_dict['v_optim'])
        self.ae_optim.load_state_dict(state_dict['ae_optim'])
        self.c_optim.load_state_dict(state_dict['c_optim'])
    
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
