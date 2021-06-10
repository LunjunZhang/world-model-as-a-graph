import numpy as np
from gym import utils

from fetch import fetch_env


class DrawerBlockEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, action, reward_type='sparse'):
        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'drawer:slide': 0. if action.startswith("open") else 0.2,
            'object0:joint': [1.45, 1.1, 0.4, 0, 0., 0., 0.],
            'cabinet:joint': [1.45, 1.1, 0.4, 0, 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, "drawer_block.xml", block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=0.5, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type,
            obj_keys=("cabinet", "drawer", "object0"),
            obs_keys=("drawer", "object0"),
            goal_key="object0",

        )
        utils.EzPickle.__init__(self)

    def _reset_sim(self):
        splits = self.action.split('+')
        if "mixed" in splits:
            self.sim.set_state(self.initial_state)

            self._reset_body("object0")
            if np.random.uniform() < 0.5:
                self._reset_slide("drawer", 0)
            else:
                self._reset_slide("drawer", 0.2)

            self.sim.forward()
            return True

        return super()._reset_sim()

