import os
from gym.utils import EzPickle
from . import fetch_env
import numpy as np


class BinEnv(fetch_env.FetchEnv, EzPickle):
    def __init__(self, action, reward_type="sparse", **kwargs):

        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        if action == "no-bin" or action == "no-init":
            pass
        elif action == "bin-aside":
            self.initial_qpos['bin:joint'] = [1.25, 0.33, 0.6, 0, 0., 0., 0.]

        self.initial_qpos['object0:joint'] = [1.25, 0.53, 0.6, 0, 0., 0., 0.]

        _kwargs = dict(
            obj_keys=("object0",) if action == "no-bin" else ('bin', 'object0'),
            obs_keys=("object0",),
            goal_key="object0",
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=0.5,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=self.initial_qpos,
            reward_type=reward_type
        )
        _kwargs.update(kwargs)
        fetch_env.FetchEnv.__init__(self, "bin_null.xml" if action == 'no-bin' else "bin.xml", **_kwargs)
        EzPickle.__init__(self)

    def _reset_sim(self):
        """
        :return: True, Read by the reset function to know this is ready.
        """
        for obj_key in self.obj_keys:
            self._reset_body(obj_key)
        if self.action == "bin-aside":
            # todo: fix the location of the bin
            original_pos = self.initial_qpos['bin:joint']
            self._reset_body("bin", original_pos[:2])
        self.sim.forward()
        return True

    def _step_callback(self):
        super()._step_callback()
        if not self.action:
            return
        # if "place" in self.action:
        #     # goal setting
        #     self.goal = self.sim.data.get_site_xpos("bin").copy()
        #     self.goal[2] = self.initial_heights['object0']
        ## todo: change to default behavior after stabilization
        if self.action == "bin-aside":
            # todo: fix the location of the bin
            original_pos = self.initial_qpos['bin:joint']
            self._reset_body("bin", original_pos[:2])

    def _sample_goal(self):
        # if self.action == "pick":
        # xpos = bin_xpos = self.sim.data.get_site_xpos("bin").copy()
        # while np.linalg.norm(xpos - bin_xpos) < 0.1:
        #     xpos = super()._sample_goal()
        # return xpos
        # elif "place" in self.action:
        #     # if np.random.uniform() < 0.1:
        #     bin_xpos = self.sim.data.get_site_xpos("bin").copy()
        #     bin_xpos[2] = self.initial_heights['object0']
        #     return bin_xpos
        return super()._sample_goal()
