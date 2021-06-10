import torch
from torch.optim import Adam

import os.path as osp
from rl.utils import mpi_utils


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


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
            action_l2=action_l2.item(),
            Loss_bc=loss_bc.item(),
        )
        monitor_log = dict(
            q_pi=q_pi,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        
        return loss_actor
    
    def update(self, batch):
        loss_critic = self.critic_loss(batch)
        self.q_optim.zero_grad()
        loss_critic.backward()
        if mpi_utils.use_mpi():
            mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.q_optim.step()
        
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        if mpi_utils.use_mpi():
            mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    def state_dict(self):
        return dict(
            q_optim=self.q_optim.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim.load_state_dict(state_dict['q_optim'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)
