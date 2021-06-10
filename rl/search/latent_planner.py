import numpy as np
import torch

from torch.distributions import Categorical


def matrix_iter(
        matrix: torch.Tensor,
        temp=1.0,
):
    if matrix.max() > 0.0:
        import pdb; pdb.set_trace()
    dist_tensor = matrix[:, :, None] + matrix[None, :, :]  # [i, j, new] + [new, i, j] => [i, i+j, j]
    attend = torch.softmax(dist_tensor / temp, dim=1)
    return (dist_tensor * attend).sum(dim=1)


def value_iter(
        matrix: torch.Tensor,
        temp=1.0,
        n_iter=20,
):
    n_states = matrix.size(0)
    assert matrix.ndim == 2 and matrix.size(0) == matrix.size(1)
    value_matrix = matrix * (torch.ones_like(matrix).to(matrix.device) - torch.eye(n_states).to(matrix.device))
    for _ in range(n_iter):
        value_matrix = matrix_iter(value_matrix, temp=temp)
    return value_matrix


def pairwise_dists(
        states: torch.Tensor,
        goals: torch.Tensor,
        agent,
):
    with torch.no_grad():
        dists = []
        for goal in goals:
            goal_repeat = goal[None, :].repeat(states.size(0), 1)
            dists.append(
                agent.pairwise_value(states, goal_repeat)
            )
    dists = torch.stack(dists, dim=1)  # [states, goals]
    assert dists.size(0) == states.size(0) and dists.size(1) == goals.size(0)
    return dists


def v_pairwise_dists(
        start_g: torch.Tensor,
        goals: torch.Tensor,
        agent,
):
    with torch.no_grad():
        dists = []
        for goal in goals:
            goal_repeat = goal[None, :].repeat(start_g.size(0), 1)
            dists.append(
                -1. * agent.get_vf_value(start_g, goal_repeat)
            )
    dists = torch.stack(dists, dim=1)  # [start_g, goals]
    assert dists.size(0) == start_g.size(0) and dists.size(1) == goals.size(0)
    return dists


def clip_dist(dists: torch.Tensor, clip=-5.0, inf_value=1e6, ):
    dists = dists - (dists < clip).float() * inf_value
    return dists


def adaptive_clip_dist(dists: torch.Tensor, clip=-5.0, inf_value=1e6):
    assert dists.ndim == 2 and dists.size(0) == dists.size(1)
    n_states = dists.size(0)
    clip_values = clip * torch.ones(n_states).to(dists.device)
    min_dists = (dists - np.sqrt(inf_value) * torch.eye(n_states).to(dists.device)).max(dim=0)[0]
    clip_values = torch.min(min_dists, clip_values)
    dists = dists - (dists < clip_values[None, :]).float() * inf_value
    return dists


def fps_selection(
        goals_embed: torch.Tensor,
        n_select: int,
        inf_value=1e6,
        embed_epsilon=1e-3, early_stop=True,
        embed_op='mean',
):
    assert goals_embed.ndim == 2
    n_states = goals_embed.size(0)
    dists = torch.zeros(n_states).to(goals_embed.device) + inf_value
    chosen = []
    while len(chosen) < n_select:
        if dists.max() < embed_epsilon and early_stop:
            break
        idx = dists.argmax()  # farthest point idx
        idx_embed = goals_embed[idx]
        chosen.append(idx)
        # distance from the chosen point to all other pts
        diff_embed = (goals_embed - idx_embed[None, :]).pow(2)
        if embed_op == 'mean':
            new_dists = diff_embed.mean(dim=1)
        elif embed_op == 'sum':
            new_dists = diff_embed.sum(dim=1)
        elif embed_op == 'max':
            new_dists = diff_embed.max(dim=1)[0]
        else:
            raise NotImplementedError
        dists = torch.stack((dists, new_dists.float())).min(dim=0)[0]
    chosen = torch.stack(chosen)
    chosen = chosen.detach().cpu().numpy()
    return chosen


class Planner:
    def __init__(
            self,
            agent, replay_buffer,
            monitor,
            args):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.monitor = monitor
        self.args = args
        self.n_goals = None
        self.n_landmarks = None
        self.past_goal = dict()
        self.past_goal_cnt = dict()
        self.subgoal_cnt = dict()
        
        self.landmarks = None
        self.dists_to_goals = None
        self._dists_to_landmarks = None
        self._dists = None
        self._g_list = []
    
    @staticmethod
    def to_2d_array(x):
        return x.reshape(-1, x.shape[-1])
    
    def to_tensor(self, x):
        return torch.as_tensor(x).float().to(self.agent.device)
    
    @staticmethod
    def assert_compatible(x, y):
        assert x.size(0) == y.size(0) and x.size(1) == y.size(1)
    
    def reset(self):
        pass
    
    def fps_sample_batch(self, initial_sample=1000, batch_size=200):
        replay_data = self.replay_buffer.sample_regular_batch(batch_size=initial_sample)
        landmarks = replay_data['ag']
        states = replay_data['ob']
        assert landmarks.ndim == 2 and states.ndim == 2
        
        landmarks = self.to_tensor(landmarks)
        states = self.to_tensor(states)
        with torch.no_grad():
            goals_embed = self.agent.ae.encoder(landmarks)
        idx = fps_selection(
            goals_embed=goals_embed, n_select=batch_size,
            inf_value=self.args.inf_value, embed_epsilon=self.args.embed_epsilon, early_stop=False,
            embed_op=self.args.embed_op,  # 'mean' or 'sum'
        )
        landmarks = landmarks[idx]
        states = states[idx]
        return landmarks, states
    
    def update(
            self,
            goals: np.ndarray,
            test_time=False,
    ):
        if goals.ndim == 1:
            goals = self.to_2d_array(goals)
        
        with torch.no_grad():
            embeds = self.agent.cluster.centroids()
            landmarks = self.agent.ae.decoder(embeds)
        if self.args.n_extra_landmark > 0 and not test_time:
            extra_landmarks, _ = self.fps_sample_batch(
                initial_sample=self.args.initial_sample, batch_size=self.args.n_extra_landmark)
            landmarks = torch.cat([landmarks, extra_landmarks], dim=0)
        
        goals = self.to_tensor(goals)
        assert goals.size(1) == landmarks.size(1)
        n_goals = goals.size(0)  # K
        self.n_goals = n_goals
        self.past_goal = {env_id: -1 for env_id in range(self.n_goals)}
        self.past_goal_cnt = {env_id: dict() for env_id in range(self.n_goals)}
        self.subgoal_cnt = {env_id: 0 for env_id in range(self.n_goals)}
        
        n_landmarks = landmarks.size(0)
        self.n_landmarks = n_landmarks
        assert n_landmarks <= self.args.n_latent_landmarks + self.args.n_extra_landmark
        
        landmarks_only = landmarks.clone().detach()
        landmarks = torch.cat([landmarks, goals], dim=0)
        dists = v_pairwise_dists(landmarks_only, landmarks, agent=self.agent)  # n_landmark * (n_landmark + K)
        self.monitor.store(Graph_dist_raw=dists.mean().item())
        
        dists = torch.min(dists, dists * 0.)
        if dists.max() > 0.0:
            import pdb; pdb.set_trace()
        self.monitor.store(Graph_dist_min_0=dists.mean().item())
        
        # pad the last K rows in dists by -inf because nothing starts from the goal
        goals_dist_ph = torch.zeros(n_goals, n_landmarks + n_goals).to(dists.device)
        dists = torch.cat([dists, goals_dist_ph - self.args.inf_value], dim=0)
        # (n_landmark + K) * (n_landmark + K)
        
        dists = adaptive_clip_dist(
            dists, clip=self.args.dist_clip, inf_value=self.args.inf_value)
        dists = value_iter(dists, temp=self.args.temp, n_iter=self.args.vi_iter, )
        self.monitor.store(Graph_dist_vi=dists.mean().item())
        
        # from all the landmarks to goal
        dists_to_goals = dists[:, -n_goals:]  # shape: (n_landmark + K) * K
        dists_to_goals = dists_to_goals.permute(1, 0)  # K * (n_landmark + K)
        self.monitor.store(Graph_dist_to_goal=dists_to_goals.mean().item())
        
        self.landmarks = landmarks
        self.dists_to_goals = dists_to_goals
    
    def goal_repeated(self, env_id, goal_idx):
        return goal_idx in self.past_goal_cnt[env_id] and self.past_goal_cnt[env_id][goal_idx] >= 5
    
    def get_subgoals(
            self,
            obs: np.ndarray,
            goals: np.ndarray,
    ):
        obs = self.to_2d_array(obs)  # (K, obs_dim)
        goals = self.to_2d_array(goals).copy()  # (K, goal_dim)
        
        landmarks = self.landmarks
        assert landmarks.ndim == 2
        # (K, n_landmark + K, dim)
        obs = self.to_tensor(obs)[:, None, :].repeat(1, self.landmarks.size(0), 1)
        landmarks = landmarks[None, :, :].repeat(obs.size(0), 1, 1)
        self.assert_compatible(obs, landmarks)
        
        # remember, obs is batched, and thus different across the batch
        with torch.no_grad():
            dists_to_landmarks = self.agent.pairwise_value(obs, landmarks)  # (K, n_landmark + K)
        n_goals = goals.shape[0]
        dists_to_landmarks = dists_to_landmarks.reshape(n_goals, -1)
        self.monitor.store(Dist_to_landmarks=dists_to_landmarks.mean().item())
        
        dists_to_landmarks = clip_dist(
            dists_to_landmarks, clip=self.args.dist_clip, inf_value=self.args.inf_value)
        self._dists_to_landmarks = dists_to_landmarks.clone()
        
        self.assert_compatible(dists_to_landmarks, self.dists_to_goals)
        dists = dists_to_landmarks + self.dists_to_goals  # (K, n_landmark + K)
        self.monitor.store(Dist_min_path=torch.min(dists).item())
        self.monitor.store(Shortest_Path=torch.min(dists, dim=-1)[0].mean().item())
        self._dists = dists
        
        extra_steps = 1.0
        self._g_list = []
        for env_id in range(obs.size(0)):
            env_goal_idx = self.n_landmarks + env_id
            prev_idx = self.past_goal[env_id]
            if self.subgoal_cnt[env_id] > 1.0:
                goals[env_id] = self.landmarks[prev_idx].detach().cpu().numpy()
                self.subgoal_cnt[env_id] -= 1.0
                self._g_list.append(prev_idx)
                continue
            if dists_to_landmarks[env_id, env_goal_idx] < - self.args.local_horizon:
                if prev_idx != -1:
                    dists[env_id, prev_idx] -= self.args.inf_value
                try:
                    out_probs = torch.softmax(dists[env_id] / self.args.temp, dim=-1)
                    idx = Categorical(out_probs).sample()
                except:
                    idx = torch.max(dists[env_id].cpu(), dim=-1)[1]
                # idx = torch.max(dists[env_id], dim=-1)[1]
                idx = int(idx.cpu().numpy())
                steps_to_landmark = - dists_to_landmarks[env_id, idx].cpu().numpy()
                goals[env_id] = self.landmarks[idx].detach().cpu().numpy()
                env_goal_idx = idx
                self.past_goal[env_id] = env_goal_idx
                self.subgoal_cnt[env_id] = steps_to_landmark + extra_steps
            self._g_list.append(env_goal_idx)
        
        assert goals.ndim == 2
        return goals
    
    def forward_empty_step(self):
        if self.args.use_forward_empty_step:
            for env_id in self.subgoal_cnt:
                if self.subgoal_cnt[env_id] > 1.0:
                    self.subgoal_cnt[env_id] -= 1.0
