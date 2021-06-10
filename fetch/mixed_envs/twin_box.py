import os
from gym import utils
from .. import fetch_env


class TwinBoxEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, action, reward_type='sparse'):
        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.73, 0.4, 1., 0., 0., 0.],
            'box0:joint': [1.05, 0.93, 0.4, 1, 0., 0., 0.],
            'box1:joint': [1.05, 0.93, 0.4, 1, 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, "twin_box.xml", block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=0.5, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=self.initial_qpos, reward_type=reward_type,
            obj_keys=("bin", "box", "lid", "object0"),
            goal_key="object0",
        )
        utils.EzPickle.__init__(self)

