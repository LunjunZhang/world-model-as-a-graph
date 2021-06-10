import numpy as np
import torch
import datetime
import os
import os.path as osp
import sys

from rl import logger

from rl.utils import mpi_utils
from rl.utils.run_utils import Timer, log_config, merge_configs
from rl.replay.core import sample_her_transitions


class BaseAlgo:
    def __init__(
            self,
            env, env_params, args,
            agent, replay, monitor, learner,
            reward_func,
            name='algo',
    ):
        self.env = env
        self.env_params = env_params
        self.args = args
        
        self.agent = agent
        self.replay = replay
        self.monitor = monitor
        self.learner = learner
        
        self.reward_func = reward_func
        
        self.timer = Timer()
        self.start_time = self.timer.current_time
        self.total_timesteps = 0
        
        self.env_steps = 0
        self.opt_steps = 0
        
        self.num_envs = 1
        if hasattr(self.env, 'num_envs'):
            self.num_envs = getattr(self.env, 'num_envs')
        
        self.n_mpi = mpi_utils.get_size()
        self._save_file = str(name) + '.pt'
        
        if len(args.resume_ckpt) > 0:
            resume_path = osp.join(
                osp.join(self.args.save_dir, self.args.env_name),
                osp.join(args.resume_ckpt, 'state'))
            self.load_all(resume_path)
        
        self.log_path = osp.join(osp.join(self.args.save_dir, self.args.env_name), args.ckpt_name)
        self.model_path = osp.join(self.log_path, 'state')
        if mpi_utils.is_root() and not args.play:
            os.makedirs(self.model_path, exist_ok=True)
            logger.configure(dir=self.log_path, format_strs=["csv", "stdout", "tensorboard"])
            config_list = [env_params.copy(), args.__dict__.copy(), {'NUM_MPI': mpi_utils.get_size()}]
            log_config(config=merge_configs(config_list), output_dir=self.log_path)
    
    def run_eval(self, use_test_env=False):
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
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
            for timestep in range(env._max_episode_steps):
                a = self.agent.get_actions(ob, bg)
                observation, _, _, info = env.step(a)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
                ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
                self.monitor.store(Inner_Test_AgChangeRatio=np.mean(ag_changed))
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(TestAgChangeRatio=np.mean(ag_changed))
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
    
    def log_everything(self):
        for log_name in self.monitor.epoch_dict:
            log_item = self.monitor.log(log_name)
            if mpi_utils.use_mpi():
                log_item_k = log_item.keys()
                log_item_v = np.array(list(log_item.values()))
                log_item_v = mpi_utils.global_mean(log_item_v)
                log_item = dict(zip(log_item_k, log_item_v))
            logger.record_tabular(log_name, log_item['mean'])
        logger.record_tabular('TotalTimeSteps', self.total_timesteps)
        logger.record_tabular('Time', self.timer.current_time - self.start_time)
        if mpi_utils.is_root():
            logger.dump_tabular()
    
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
    
    def save_all(self, path):
        self.save(path)
        self.agent.save(path)
        self.replay.save(path)
        self.learner.save(path)
    
    def load_all(self, path):
        self.load(path)
        self.agent.load(path)
        self.replay.load(path)
        self.learner.load(path)


class Algo(BaseAlgo):
    def __init__(
            self,
            env, env_params, args,
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
            self.learner.update(batch)
            self.opt_steps += 1
            if self.opt_steps % self.args.target_update_freq == 0:
                self.learner.target_update()
        
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
        
        for timestep in range(self.env_params['max_timesteps']):
            act = self.get_actions(ob, bg, a_max=a_max, act_randomly=act_randomly)
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
        self.replay.store(experience)
        self.update_normalizer(experience)
    
    def update_normalizer(self, buffer):
        transitions = sample_her_transitions(
            buffer=buffer, reward_func=self.reward_func,
            batch_size=self.env_params['max_timesteps'] * self.num_envs,
            future_p=self.args.future_p)
        self.agent.normalizer_update(obs=transitions['ob'], goal=transitions['bg'])
    
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
                
                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout') / self.args.num_rollouts_per_mpi)
            
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(opt_steps=self.opt_steps)
            self.monitor.store(replay_size=self.replay.current_size)
            self.monitor.store(replay_fill_ratio=float(self.replay.current_size / self.replay.size))
            
            success_rate = self.run_eval()
            if mpi_utils.is_root():
                print('Epoch %d eval success rate %.3f' % (epoch, success_rate))
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular('TestSuccessRate', success_rate)
            self.log_everything()
            
            self.save_all(self.model_path)
    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']
