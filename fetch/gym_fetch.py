from gym.utils import EzPickle
import numpy as np
from fetch import fetch_env


class GymFetchEnv(fetch_env.FetchEnv, EzPickle):
    def __init__(self, action, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        if action == "reach":
            obj_keys, goal_key = None, "robot0:grip"
        else:
            initial_qpos['object0:joint'] = [1.25, 0.53, 0.6, 0, 0., 0., 0.]
            obj_keys = "object0",
            goal_key = "object0"

        if action == "slide":
            initial_qpos['robot0:slide0'] = 0.05
            initial_qpos['object0:joint'] = [1.7, 1.1, 0.41, 1., 0., 0., 0.]
            gripper_extra_height = -0.2
            target_offset = np.array([0.4, 0.0, 0.0])
            obj_range = 0.1
            target_range = 0.3
        else:
            gripper_extra_height = 0.2
            target_offset = 0.0
            obj_range = 0.15
            target_range = 0.15

        if action == "push":
            gripper_extra_height = 0.0
        if action in ["reach", "pick-place"]:
            gripper_extra_height = 0.2
        if action != "reach":
            obj_reset = {'object0': dict(track="gripper", avoid="gripper")}

        target_in_the_air = 0.5 if action in ['reach', 'pick-place'] else False
        block_gripper = action in ['reach', 'slide']

        local_vars = locals().copy()
        del local_vars['action']
        del local_vars['self']

        fetch_env.FetchEnv.__init__(
            self, f"{action.replace('-', '_')}.xml",
            n_substeps=20,
            distance_threshold=0.05,
            **local_vars)
        EzPickle.__init__(self)


if __name__ == '__main__':
    import gym

    env = gym.make('fetch:PickPlace-v0')
    env = gym.make('fetch:Push-v0')
    # env = gym.make('fetch:Bin-no-bin-v0')
