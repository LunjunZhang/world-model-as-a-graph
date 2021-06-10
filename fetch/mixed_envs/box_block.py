from gym import utils
from fetch import fetch_env


def assign(d, *objs):
    for o in objs:
        d.update(o)
    return d


class BoxBlockEnv(fetch_env.FetchEnv, utils.EzPickle):

    def __init__(self, action,
                 block_gripper=False, n_substeps=20,
                 gripper_extra_height=0.2, target_in_the_air=0.5, target_offset=0.0,
                 obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                 reward_type='sparse',
                 obj_keys=("box", "lid", "object0"),
                 obs_keys=("box@pos", "lid", "object0"),
                 goal_key="object0",
                 initial_qpos=None,
                 obj_reset=None,
                 goal_sampling=None,
                 freeze_objects=None,
                 ):
        self.action = action
        initial_qpos = assign({'box:joint': [1.15, 0.53, 0.4, 0, 0., 0., 0.],
                               'object0:joint': [1.15, 0.53, .5, 0, 0., 0., 0.],
                               'lid:joint': [1.15, 0.53, 0.5, 0, 0., 0., 0.]}, initial_qpos or {})

        local_vars = locals()

        del local_vars['action']
        del local_vars['self']

        fetch_env.FetchEnv.__init__(self, "box_block.xml", **local_vars)
        utils.EzPickle.__init__(self)

    # noinspection PyMethodOverriding
    # def _reset_sim(self):
    #     """
    #     :return: True, Read by the reset function to know this is ready.
    #     """
    #     super()._reset_sim(forward=False)
    #
    #     box_pos = self.sim.data.get_site_xpos('box')
    #     lid_pos = self.sim.data.get_site_xpos('lid')
    #     obj_pos = self.sim.data.get_site_xpos('object0')
    #
    #     # for k, kwgs in self.obj_reset.items():
    #     # lid init
    #     if "lid" not in self.freeze_objects:
    #         while np.linalg.norm(lid_pos[:2] - box_pos[:2]) < 0.1:
    #             lid_pos = self._reset_body("lid", h=self.initial_heights['object0'])
    #     # object init
    #     if "object0" not in self.freeze_objects:
    #         while np.linalg.norm(obj_pos[:2] - box_pos[:2]) < 0.1 or np.linalg.norm(obj_pos[:2] - lid_pos[:2]) < 0.1:
    #             obj_pos = self._reset_body("object0")
    #
    #     self.sim.forward()
    #     return True

    # def _sample_open_lid_goal(self):
    #     xpos = box_xpos = self.sim.data.get_site_xpos("box").copy()
    #     while np.linalg.norm(xpos - box_xpos) < 0.2:
    #         xpos = super()._sample_single_goal("lid")  # 50% in the air.
    #     return xpos
    #
    # def _sample_closed_lid_goal(self):
    #     xpos = self.sim.data.get_site_xpos("box").copy()
    #     xpos[2] = self.initial_heights['lid']
    #     return xpos
    #
    # # not currently used
    # def _sample_table_lid_goal(self):
    #     # todo: needs attention: is the height correct, and is the range okay?
    #     xpos = box_xpos = self.sim.data.get_site_xpos("box").copy()
    #     while np.linalg.norm(xpos - box_xpos) < 0.2:
    #         xpos = super()._sample_single_goal(h=0.404)
    #     return xpos
    #
    # def _sample_place_goal(self):
    #     # Place means placing into the box.
    #     # assumes that the object is initialized inside the box.
    #     # if not, add 0.02 to the vertical position.
    #     xpos = self.sim.data.get_site_xpos("box").copy()
    #     xpos[2] = self.initial_heights['object0']  # + 0.02
    #     return xpos

    # 重要的事情说三遍！！
    # def _sample_goal(self):
    #     goal = super()._sample_goal()
    #     goal = goal if isinstance(goal, dict) else {self.goal_key: goal}
    #
    #     if "place" in self.action:
    #         goal['object0'] = self._sample_place_goal()
    #
    #     return goal[self.goal_key] if isinstance(self.goal_key, str) else goal
