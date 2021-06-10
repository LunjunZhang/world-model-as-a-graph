"""Wrapper for creating the ant environment in gym_mujoco."""

import math
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "point.xml"
    ORI_IND = 2
    
    def __init__(self, file_path=None, expose_all_qpos=True):
        self._expose_all_qpos = expose_all_qpos
        
        mujoco_env.MujocoEnv.__init__(self, file_path, 1)
        utils.EzPickle.__init__(self)
    
    @property
    def physics(self):
        return self.model
    
    def _step(self, a):
        return self.step(a)
    
    def step(self, action):
        action[0] = 0.2 * action[0]
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -100, 100)
        qpos[1] = np.clip(qpos[1] + dy, -100, 100)
        qvel = self.data.qvel
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = 0
        done = False
        info = {}
        '''
        site_id = self.sim.model.site_name2id('target0')
        print('site_pos', self.sim.model.site_pos[site_id])
        '''
        return next_obs, reward, done, info
    
    def _get_obs(self):
        if self._expose_all_qpos:
            return np.concatenate([
                self.data.qpos.flat[:3],  # Only point-relevant coords.
                self.data.qvel.flat[:3]])
        return np.concatenate([
            self.data.qpos.flat[2:3],
            self.data.qvel.flat[:3]])
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        
        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def get_ori(self):
        return self.data.qpos[self.__class__.ORI_IND]
    
    def set_xy(self, xy):
        qpos = np.copy(self.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        
        qvel = self.data.qvel
        self.set_state(qpos, qvel)
    
    def get_xy(self):
        qpos = np.copy(self.data.qpos)
        return qpos[:2]
    
    def render_callback(self, goal):
        goal = goal.copy()
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        goal = np.concatenate([goal, sites_offset[0][-1:]])
        # self.sim.model.site_pos[site_id] = goal - sites_offset[0]
        self.sim.model.site_pos[site_id] = goal
        self.sim.forward()
