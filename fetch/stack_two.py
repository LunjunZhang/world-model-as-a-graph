import os
from gym.utils import EzPickle
from . import fetch_env
import numpy as np


class StackTwo(fetch_env.FetchEnv, EzPickle):
    def __init__(self,
                 action,
                 reward_type="sparse",
                 n_objects=2,
                 block_gripper=False,
                 n_substeps=20,
                 gripper_extra_height=0.2,
                 target_in_the_air=0.5,
                 target_offset=0.0,
                 obj_range=0.15,
                 target_range=0.1,
                 distance_threshold=0.05,
                 ):

        self.action = action

        if n_objects == 1:
            obj_keys = "object0",
            model_path = "pick_place.xml"
        if n_objects == 2:
            obj_keys = "object0", "object1"
            model_path = f"stack_two.xml"

        initial_qpos = {'object0:joint': [1.15, 0.53, 0.4, 0, 0, 0, 0],
                        'object1:joint': [1.15, 0.53, 0.45, 0, 0, 0, 0], }
        goal_sampling = {'object0': dict(range=0, in_the_air=0),
                         'object1': dict(target="object0", offset=[0, 0., 0.04], range=0, in_the_air=0.5)}

        if action in ["fix-obj0-center", "fix-obj0-pp-goals"]:
            freeze_objects = ['object0']
            initial_qpos['object1:joint'][:2] = [1.34193226, 0.74910037]
            initial_qpos['object0:joint'][:2] = [1.34193226, 0.74910037]

            if action == "fix-obj0-pp-goals":
                target_in_the_air = 0.5
                del goal_sampling['object1']
                # goal_sampling['object1'] = dict(h=0.08, )

        obj_keys = "object0", "object1"
        obs_keys = "object0", "object1"
        goal_key = "object0", "object1"

        local_vars = locals()
        del local_vars['action']
        del local_vars['n_objects']
        del local_vars['self']

        fetch_env.FetchEnv.__init__(self, **local_vars)
        EzPickle.__init__(self)

    def _get_mode(self):
        rho_0 = {
            'table-top': 7 / 10,
            'in-hand': 1 / 5,
            'obj1-in-hand': 1 / 10
        }

        if self.action == "train":
            keys, weights = zip(*rho_0.items())
            return self.np_random.choice(keys, p=weights)
        if self.action == "test":
            return "table-top"

        return self.action

    def _reset_sim(self):
        if self.action in ["fix-obj0-center", "fix-obj0-pp-goals"]:
            return super()._reset_sim()

        # For the rest use the following
        self.sim.set_state(self.initial_state)

        mode = self._get_mode()

        if mode == "table-top":
            cache = {'object0': self._reset_body("object0")}
            self._reset_body("object1", avoid=['object0', 'gripper'], d_min=0.075, m=cache)
        elif mode == "in-hand":
            self._set_obj_in_hand('object0')
            self._reset_body("object1", avoid="gripper", d_min=0.1)
        elif mode == "obj1-in-hand":
            self._reset_body("object0", track="gripper", d_min=0, range=0)
            self._set_obj_in_hand('object1')

        self.sim.forward()
        return True
