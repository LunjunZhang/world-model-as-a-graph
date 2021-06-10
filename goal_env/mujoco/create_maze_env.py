from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
from collections import OrderedDict
import gym
import numpy as np
import copy
from gym import Wrapper
from gym.envs.registration import EnvSpec


class GoalWrapper(Wrapper):
    def __init__(self, env, maze_size_scaling, random_start, low, high):
        super(GoalWrapper, self).__init__(env)
        ob_space = env.observation_space
        self.maze_size_scaling = maze_size_scaling
        low = np.array(low, dtype=ob_space.dtype)
        high = np.array(high, dtype=ob_space.dtype)
        maze_low = np.array(np.array([-4, -4]) / 8 * maze_size_scaling, dtype=ob_space.dtype)
        maze_high = np.array(np.array([20, 20]) / 8 * maze_size_scaling, dtype=ob_space.dtype)
        self.maze_size_scaling = maze_size_scaling
        self.goal_space = gym.spaces.Box(low=low, high=high)
        self.maze_space = gym.spaces.Box(low=maze_low, high=maze_high)
        
        self.goal_dim = low.size
        # self.distance_threshold = 5 * maze_size_scaling / 8.
        self.distance_threshold = maze_size_scaling / 8.
        
        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        }))
        self.goal = None
        self.random_start = random_start
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        out = {'observation': observation,
               'desired_goal': self.goal,
               'achieved_goal': observation[..., :self.goal_dim]}
        reward = -np.linalg.norm(observation[..., :self.goal_dim] - self.goal, axis=-1)
        info['is_success'] = (reward > -self.distance_threshold)
        reward = self.compute_rew(observation[..., :self.goal_dim], self.goal, '...')
        return out, reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        self.goal = self.goal_space.sample()
        while self.env._is_in_collision(self.goal):
            self.goal = self.goal_space.sample()
        
        # random start a position without collision
        if self.random_start:
            xy = self.maze_space.sample()
            while self.env._is_in_collision(xy):
                xy = self.maze_space.sample()
            self.env.wrapped_env.set_xy(xy)
            observation = self.env._get_obs()
        
        out = dict(observation=observation, desired_goal=self.goal, achieved_goal=observation[..., :self.goal_dim])
        return out
    
    def compute_rew(self, state, goal, info):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        return -(dist > self.distance_threshold).astype(np.float32)
    
    def render(self, *args, **kwargs):
        self.env.wrapped_env.render_callback(self.goal)
        return self.env.render(*args, **kwargs)


def create_maze_env(env_name=None, top_down_view=False, maze_size_scaling=8, random_start=True, goal_args=None):
    if goal_args is None:
        goal_args = []
    n_bins = 0
    manual_collision = False
    if env_name.startswith('Ego'):
        n_bins = 8
        env_name = env_name[3:]
    if env_name.startswith('Ant'):
        manual_collision = True
        cls = AntMazeEnv
        env_name = env_name[3:]
        maze_size_scaling = maze_size_scaling
    elif env_name.startswith('Point'):
        cls = PointMazeEnv
        manual_collision = True
        env_name = env_name[5:]
        maze_size_scaling = maze_size_scaling
    else:
        assert False, 'unknown env %s' % env_name
    
    maze_id = None
    observe_blocks = False
    put_spin_near_agent = False
    if env_name == 'Maze':
        maze_id = 'Maze'
    elif env_name == 'Maze1':
        maze_id = 'Maze1'
    elif env_name == 'Push':
        maze_id = 'Push'
    elif env_name == 'Fall':
        maze_id = 'Fall'
    elif env_name == 'Block':
        maze_id = 'Block'
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == 'BlockMaze':
        maze_id = 'BlockMaze'
        put_spin_near_agent = True
        observe_blocks = True
    else:
        raise ValueError('Unknown maze environment %s' % env_name)
    
    gym_mujoco_kwargs = {
        'maze_id': maze_id,
        'n_bins': n_bins,
        'observe_blocks': observe_blocks,
        'put_spin_near_agent': put_spin_near_agent,
        'top_down_view': top_down_view,
        'manual_collision': manual_collision,
        'maze_size_scaling': maze_size_scaling,
    }
    gym_env = cls(**gym_mujoco_kwargs)
    gym_env.reset()
    goal_args = np.array(goal_args) / 8 * maze_size_scaling
    return GoalWrapper(gym_env, maze_size_scaling, random_start, *goal_args)
