import gym
import copy
import numpy as np
import cv2
from collections import OrderedDict


class GoalPlane(gym.Env):
    def __init__(self, env_name, type='random', maze_size=16., action_size=1., distance=0.1, start=None, goals=None):
        super(GoalPlane, self).__init__()
        self.env = gym.make(env_name)
        self.maze_size = maze_size
        self.action_size = action_size
        
        self.action_space = gym.spaces.Box(
            low=-action_size, high=action_size, shape=(2,), dtype='float32')
        
        self.ob_space = gym.spaces.Box(
            low=0., high=maze_size, shape=(2,), dtype='float32')
        
        self.easy_goal_space = gym.spaces.Box(
            low=np.array([0., 0.]),
            high=np.array([self.maze_size, self.maze_size / 2]), dtype=np.float32)
        self.mid_goal_space = gym.spaces.Box(
            low=np.array([self.maze_size / 2, self.maze_size / 2]),
            high=np.array([self.maze_size, self.maze_size]), dtype=np.float32)
        self.hard_goal_space = gym.spaces.Box(
            low=np.array([0., self.maze_size * 0.65]),
            high=np.array([self.maze_size / 2, self.maze_size]), dtype=np.float32)
        self.type = type
        if self.type == 'random':
            self.goal_space = self.ob_space
        elif self.type == 'easy':
            self.goal_space = self.easy_goal_space
        elif self.type == 'mid':
            self.goal_space = self.mid_goal_space
        elif self.type == 'hard':
            self.goal_space = self.hard_goal_space
        
        self.distance = distance
        self.goals = goals
        self.start = start
        
        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': self.ob_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.ob_space,
        }))
        self.goal = None
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return reward
    
    def change_mode(self, mode='mid'):
        if mode == 'random':
            self.goal_space = self.ob_space
        elif mode == 'easy':
            self.goal_space = self.easy_goal_space
        elif mode == 'mid':
            self.goal_space = self.mid_goal_space
        elif mode == 'hard':
            self.goal_space = self.hard_goal_space
    
    def step(self, action):
        assert self.goal is not None
        observation, reward, done, info = self.env.step(np.array(action) / self.maze_size)  # normalize action
        observation = np.array(observation) * self.maze_size
        
        out = {'observation': observation,
               'desired_goal': self.goal,
               'achieved_goal': observation}
        reward = -np.linalg.norm(observation - self.goal, axis=-1)
        info['is_success'] = (reward > -self.distance)
        return out, reward, done, info
    
    def reset(self):
        if self.start is not None:
            self.env.reset()
            observation = np.array(self.start)
            self.env.restore(observation / self.maze_size)
        else:
            observation = self.env.reset()
        if self.goals is None:
            condition = True
            while condition:  # note: goal should not be in the block
                self.goal = self.goal_space.sample()
                condition = self.env.check_inside(self.goal / self.maze_size)
        else:
            self.goal = np.array(self.goals)
        out = {'observation': observation, 'desired_goal': self.goal}
        out['achieved_goal'] = observation
        return out
    
    def render(self, mode='rgb_array'):
        image = self.env.render(mode='rgb_array')
        goal_loc = copy.copy(self.goal)
        goal_loc[0] = goal_loc[0] / self.maze_size * image.shape[1]
        goal_loc[1] = goal_loc[1] / self.maze_size * image.shape[0]
        cv2.circle(image, (int(goal_loc[0]), int(goal_loc[1])), 10, (0, 255, 0), -1)
        if mode == 'human':
            cv2.imshow('image', image)
            cv2.waitKey(2)
        else:
            return image
