from gym import utils
import numpy as np
from . import fetch_env


class BoxEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, action, reward_type='sparse'):
        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'box:joint': [1.25, 0.53, 0.4, 1, 0., 0., 0.],
            'lid:joint': [1.25, 0.53, 1, 1, 0., 0., 0.],
            # we need the height for setting the target, so no placing-aside.
            # 'lid:joint': [1.25, 0.53 if action == "open" else 0.95, 1, 1, 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, "box.xml",
            obj_keys=("box", "lid"),
            obs_keys=("lid",),
            goal_key="lid",
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=0.5, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _reset_sim(self):
        """
        :return: True, Read by the reset function to know this is ready.
        """
        self.sim.set_state(self.initial_state)

        if self.action == "open":
            box_pos = self._reset_body("box")
            self._reset_body("lid", box_pos[:2])
        elif self.action == "close":
            lid_pos = box_pos = self._reset_body("box")
            while np.linalg.norm(lid_pos[:2] - box_pos[:2]) < 0.2:
                lid_pos = self._reset_body("lid")
        else:
            raise NotImplementedError(f"Support for {self.action} is not implemented")
        self.sim.forward()
        return True

    def _step_callback(self):
        super()._step_callback()
        if self.action == "close":
            bin_xpos = self.sim.data.get_site_xpos("box").copy()
            bin_xpos[2] = self.initial_heights['lid']
            self.goal = bin_xpos

    def _sample_goal(self):
        if self.action == "open":
            xpos = box_xpos = self.sim.data.get_site_xpos("box").copy()
            while np.linalg.norm(xpos - box_xpos) < 0.2:
                xpos = super()._sample_goal()
            return xpos
        elif self.action == "close":
            box_xpos = self.sim.data.get_site_xpos("box").copy()
            box_xpos[2] = self.initial_heights['lid']
            return box_xpos
        raise NotImplementedError(f"Support for {self.action} is not implemented")
