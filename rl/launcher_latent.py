import time
import gym
import random
import numpy as np
import torch

from functools import partial

from rl.utils import mpi_utils
from rl.utils import vec_env
from rl.utils.run_utils import Monitor

from rl.agent.latent_planner import Agent
from rl.algo.latent_planner import Algo

from rl.learn.latent_planner import Learner
from rl.replay.planner import Replay

import goal_env
from goal_env import mujoco


def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],  # env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def get_env_with_id(num_envs, env_id):
    vec_fn = vec_env.SubprocVecEnv
    return vec_fn([lambda: gym.make(env_id) for _ in range(num_envs)])


def get_env_with_fn(num_envs, env_fn, *args, **kwargs):
    vec_fn = vec_env.SubprocVecEnv
    return vec_fn([lambda: env_fn(*args, **kwargs) for _ in range(num_envs)])


def launch(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.test_env_name)
    
    rank = mpi_utils.get_rank()
    seed = args.seed + rank * args.n_workers
    
    env.seed(seed)
    test_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    assert np.all(env.action_space.high == -env.action_space.low)
    env_params = get_env_params(env)
    
    def compute_reward(state, goal, info):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        return -(dist > args.goal_eps).astype(np.float32)
    
    reward_func = compute_reward
    monitor = Monitor()
    
    if args.n_workers > 1:
        env = get_env_with_id(num_envs=args.n_workers, env_id=args.env_name)
        env.seed(seed)
        
        test_env = get_env_with_id(num_envs=args.n_workers, env_id=args.test_env_name)
        test_env.seed(seed)
    
    ckpt_name = args.ckpt_name
    if len(ckpt_name) == 0:
        data_time = time.ctime().split()[1:4]
        ckpt_name = data_time[0] + '-' + data_time[1]
        time_list = np.array([float(i) for i in data_time[2].split(':')], dtype=np.float32)
        if mpi_utils.use_mpi():
            time_list = mpi_utils.bcast(time_list)
        for time_ in time_list:
            ckpt_name += '-' + str(int(time_))
        args.ckpt_name = ckpt_name
    
    agent = Agent(env_params, args)
    replay = Replay(env_params, args, reward_func)
    learner = Learner(agent, monitor, args)
    algo = Algo(
        env=env, env_params=env_params, args=args,
        test_env=test_env,
        agent=agent, replay=replay, monitor=monitor, learner=learner,
        reward_func=reward_func,
    )
    return algo
