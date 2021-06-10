## copied from RL-Adventure2
import gym
import numpy as np
from gym import spaces


class BitFlipEnv(gym.Env):
    def __init__(self, num_bits):
        self.num_bits = num_bits
        
        self.observation_space = {
            'observation': spaces.Box(low=0, high=1, shape=(self.num_bits,)),
            'desired_goal': spaces.Box(low=0, high=1, shape=(self.num_bits,)),
            'achieved_goal': spaces.Box(low=0, high=1, shape=(self.num_bits,))
        }
        self.action_space = spaces.Discrete(self.num_bits)
    
    def get_obs(self):
        return {
            "observation": np.copy(self.state),
            "achieved_goal": np.copy(self.state),
            "desired_goal": np.copy(self.target),
        }
    
    def reset(self):
        self.done = False
        self.num_steps = 0
        self.state = np.random.randint(2, size=self.num_bits)
        self.target = np.random.randint(2, size=self.num_bits)
        return self.get_obs()
        # return self.state, self.target
    
    def step(self, action):
        self.state[action] = 1 - self.state[action]
        info = {'is_success': False}
        # print(self.state, self.target)
        if self.num_steps > self.num_bits + 1:
            self.done = True
        self.num_steps += 1
        
        if np.sum(self.state == self.target) == self.num_bits:
            self.done = True
            info = {'is_success': True}
            return self.get_obs(), 0, self.done, info
        else:
            return self.get_obs(), -1, self.done, info
    
    def compute_reward(self, state, goal, info):
        calcu = np.sum(state == goal, axis=1)
        reward = np.where(calcu == self.num_bits, 0, -1)
        return reward
    
    def get_pairwise(self, state, target):
        dist = self.num_bits - np.sum(state == target)
        return dist
