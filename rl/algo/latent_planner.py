import numpy as np
import torch
import datetime
import os
import os.path as osp
import sys

from rl import logger

from rl.utils import mpi_utils
from rl.algo.core import BaseAlgo

from rl.replay.planner import sample_her_transitions
from rl.search.latent_planner import Planner


class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        test_env,
        agent, replay, monitor, learner,
        reward_func,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            agent, replay, monitor, learner,
            reward_func,
            name=name,
        )
        self.planner = Planner(agent, replay, monitor, args)
        self.test_env = test_env
        self.fps_landmarks = None
        self._clusters_initialized = False
    
    def can_plan(self):
        replay_big_enough = self.replay.current_size > self.args.start_planning_n_traj
        return replay_big_enough
    
    def get_actions(self, ob, bg, a_max=1.0, act_randomly=False):
        act = self.agent.get_actions(ob, bg)
        if self.args.noise_eps > 0.0:
            act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
            act = np.clip(act, -a_max, a_max)
        if self.args.random_eps > 0.0:
            a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
            mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
            if self.num_envs > 1:
                mask = np.expand_dims(mask, -1)
            act += mask * (a_rand - act)
        if act_randomly:
            act = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
        return act
    
    def agent_optimize(self):
        self.timer.start('train')
        
        for n_train in range(self.args.n_batches):
            batch = self.replay.sample(batch_size=self.args.batch_size)
            self.learner.update(batch, train_embed=True)
            self.opt_steps += 1
            if self.opt_steps % self.args.target_update_freq == 0:
                self.learner.target_update()
            # cluster training
            if self.opt_steps % self.args.fps_sample_freq == 0 or self.fps_landmarks is None:
                self.fps_landmarks, _ = self.planner.fps_sample_batch(
                    initial_sample=self.args.initial_sample, batch_size=self.args.latent_batch_size)
            if self.opt_steps % self.args.cluster_update_freq == 0:
                if not self._clusters_initialized and self.agent.cluster.n_mix <= self.args.latent_batch_size:
                    self.learner.initialize_cluster(self.fps_landmarks[:self.agent.cluster.n_mix])
                self.learner.update_cluster(self.fps_landmarks, to_train=self._clusters_initialized)
        
        self.timer.end('train')
        self.monitor.store(TimePerTrainIter=self.timer.get_time('train') / self.args.n_batches)
    
    def collect_experience(self, act_randomly=False, train_agent=True):
        ob_list, ag_list, bg_list, a_list = [], [], [], []
        observation = self.env.reset()
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']
        ag_origin = ag.copy()
        a_max = self.env_params['action_max']
        self.planner.reset()
        can_plan = self.can_plan()
        if not act_randomly and can_plan:
            self.planner.update(goals=bg.copy(), test_time=False)
        
        for timestep in range(self.env_params['max_timesteps']):
            act = self.get_actions(ob, bg, a_max=a_max, act_randomly=act_randomly)
            if not act_randomly and can_plan and np.random.uniform() < self.args.plan_eps:
                sub_goals = self.planner.get_subgoals(ob, bg.copy())
                act = self.agent.get_actions(ob, sub_goals)
            else:
                self.planner.forward_empty_step()
            ob_list.append(ob.copy())
            ag_list.append(ag.copy())
            bg_list.append(bg.copy())
            a_list.append(act.copy())
            observation, _, _, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(Inner_Train_AgChangeRatio=np.mean(ag_changed))
            self.total_timesteps += self.num_envs * self.n_mpi
            for every_env_step in range(self.num_envs):
                self.env_steps += 1
                if self.env_steps % self.args.optimize_every == 0 and train_agent:
                    self.agent_optimize()
        ob_list.append(ob.copy())
        ag_list.append(ag.copy())
        
        experience = dict(ob=ob_list, ag=ag_list, bg=bg_list, a=a_list)
        experience = {k: np.array(v) for k, v in experience.items()}
        if experience['ob'].ndim == 2:
            experience = {k: np.expand_dims(v, 0) for k, v in experience.items()}
        else:
            experience = {k: np.swapaxes(v, 0, 1) for k, v in experience.items()}
        bg_achieve = self.reward_func(bg, ag, None) + 1.
        self.monitor.store(TrainSuccess=np.mean(bg_achieve))
        ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
        self.monitor.store(TrainAgChangeRatio=np.mean(ag_changed))
        self.monitor.store(Train_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
        self.replay.store(experience)
        self.update_normalizer(experience)
    
    def update_normalizer(self, buffer):
        transitions = sample_her_transitions(
            buffer=buffer, reward_func=self.reward_func,
            batch_size=self.env_params['max_timesteps'] * self.num_envs,
            future_step=self.args.future_step,
            future_p=self.args.future_p)
        self.agent.normalizer_update(obs=transitions['ob'], goal=transitions['bg'])
    
    def initialize_clusters(self):
        pts, _ = self.planner.fps_sample_batch(
            initial_sample=self.args.initial_sample, batch_size=self.agent.cluster.n_mix)
        self.learner.initialize_cluster(pts)
        self._clusters_initialized = True
    
    def run(self):
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(act_randomly=True, train_agent=False)
        
        for epoch in range(self.args.n_epochs):
            if mpi_utils.is_root():
                print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
                sys.stdout.flush()
            
            for n_iter in range(self.args.n_cycles):
                if mpi_utils.is_root():
                    print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                    sys.stdout.flush()
                self.timer.start('rollout')
                
                for n_rollout in range(self.args.num_rollouts_per_mpi):
                    self.collect_experience(train_agent=True)
                    if self.can_plan() and not self._clusters_initialized:
                        self.initialize_clusters()
                
                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout') / self.args.num_rollouts_per_mpi)
            
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(opt_steps=self.opt_steps)
            self.monitor.store(replay_size=self.replay.current_size)
            self.monitor.store(replay_fill_ratio=float(self.replay.current_size / self.replay.size))
            
            her_success = self.run_eval(use_test_env=False)
            train_env_plan_success = self.run_train_env_plan_eval() if self._clusters_initialized else 0.0
            test_env_plan_success = self.run_test_env_plan_eval() if self._clusters_initialized \
                else self.run_eval(use_test_env=True)
            if mpi_utils.is_root():
                print('Epoch %d her eval %.3f, test-env plan %.3f, train-env plan %.3f' %
                      (epoch, her_success, test_env_plan_success, train_env_plan_success))
                print('Log Path:', self.log_path)
            logger.record_tabular("Epoch", epoch)
            self.monitor.store(Test_TrainEnv_HerSuccessRate=her_success)
            self.monitor.store(Test_TrainEnv_PlanSuccessRate=train_env_plan_success)
            self.monitor.store(Test_TestEnv_PlanSuccessRate=test_env_plan_success)
            self.log_everything()
            self.save_all(self.model_path)
    
    def run_test_env_plan_eval(self):
        env = self.env
        if hasattr(self, 'test_env'):
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            ag_origin = ag.copy()
            self.planner.reset()
            self.planner.update(goals=bg.copy(), test_time=True)
            for timestep in range(env._max_episode_steps):
                sub_goals = self.planner.get_subgoals(ob, bg.copy())
                a = self.agent.get_actions(ob, sub_goals)
                observation, _, _, info = env.step(a)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
                ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
                self.monitor.store(Inner_PlanTest_AgChangeRatio=np.mean(ag_changed))
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(TestPlan_AgChangeRatio=np.mean(ag_changed))
            self.monitor.store(TestPlan_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info['is_success'] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                if info['is_success'] == 1.0:
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        if mpi_utils.use_mpi():
            success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]
        return success_rate
    
    def run_train_env_plan_eval(self):
        env = self.env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            self.planner.reset()
            self.planner.update(goals=bg.copy(), test_time=True)
            for timestep in range(env._max_episode_steps):
                sub_goals = self.planner.get_subgoals(ob, bg.copy())
                a = self.agent.get_actions(ob, sub_goals)
                observation, _, _, info = env.step(a)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
            self.monitor.store(TrainEnvTestPlan_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info['is_success'] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                if info['is_success'] == 1.0:
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        if mpi_utils.use_mpi():
            success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]
        return success_rate
    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']
