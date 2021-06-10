import os
from gym import utils
from .. import fetch_env


class DrawerBinEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, action, reward_type='sparse'):
        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'cabinet:joint': [1.45, 1.1, 1, 1, 0., 0., 0.],
            'drawer:slide': 0,
            'bin:joint': [1.25, 0.53, 0.4, 1, 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, "drawer_bin.xml",
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=0.5, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type,
            obj_keys=("bin", "cabinet", "drawer", "object0"),
            obs_keys=("bin", "cabinet", "object0"),
            goal_key="object0",

        )
        utils.EzPickle.__init__(self)
