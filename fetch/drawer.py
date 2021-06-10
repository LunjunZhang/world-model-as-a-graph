from gym import utils
import numpy as np
from . import fetch_env


class DrawerEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, action, reward_type='sparse'):
        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'drawer:slide': 0. if action == "open" else 0.2,
            'cabinet:joint': [1.45, 1.1, 0.4, 0, 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, "drawer.xml", obj_keys=("cabinet", "drawer"), goal_key="drawer",
            obs_keys=("drawer",),
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=0.5, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    handle_offset = [0, -0.16, 0.086, 0, 0, 0, 0]

    def _reset_sim(self):
        """
        :return: True, Read by the reset function to know this is ready.
        """
        self.sim.set_state(self.initial_state)

        self._reset_body("cabinet")
        if self.action == "open":
            self._reset_slide("drawer", 0)
        elif self.action == "close":
            self._reset_slide("drawer", 0.2)
        else:
            raise NotImplementedError(f"Support for {self.action} is not implemented")
        self.sim.forward()
        return True

    def _step_callback(self):
        super()._step_callback()
        bin_xpos = self.sim.data.get_site_xpos("cabinet").copy()
        bin_xpos[:2] += [0, -0.16]
        if self.action == "open":
            bin_xpos[1:2] += -0.2
        bin_xpos[2] = self.initial_heights['drawer']
        self.goal = bin_xpos

    def _sample_goal(self):
        if self.action == "open":
            xpos = bin_xpos = self.sim.data.get_site_xpos("cabinet").copy()
            while np.linalg.norm(xpos - bin_xpos) < 0.2:
                xpos = super()._sample_goal()
            bin_xpos[2] = self.initial_heights['drawer']
            return xpos
        elif self.action == "close":
            bin_xpos = self.sim.data.get_site_xpos("cabinet").copy()
            bin_xpos[2] = self.initial_heights['drawer']
            return bin_xpos
        raise NotImplementedError(f"Support for {self.action} is not implemented")
