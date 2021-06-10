from copy import deepcopy
from gym.envs.robotics import rotations, utils
import numpy as np
import numbers

from fetch.common import robot_env
from fetch.common.utils import goal_distance


def assign(o, *objs):
    for ob in objs:
        o.update(ob)
    return o


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            obj_keys, goal_key, obs_keys=None, obj_reset=None, goal_sampling=None, freeze_objects=None,
    ):
        """Initializes a new Fetch environment.

        On initialization, the objects are initialized according to the `init_qpos` dictionary.
        We record the height info to be reused later. We then call `_reset_sim` and `_get_goal`
        on reset.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (float): chance for the target to be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.obj_keys = obj_keys or []
        self.obj_reset = obj_reset
        self.obs_keys = obs_keys
        self.goal_key = goal_key
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        # add support for goal_tracking = object0@object1-top
        self.goal_sampling = goal_sampling or {}
        self.freeze_objects = freeze_objects or []

        initial_qpos = assign({'robot0:slide0': 0.405,
                               'robot0:slide1': 0.48,
                               'robot0:slide2': 0.0}, initial_qpos or {})

        super().__init__(model_path=model_path, n_substeps=n_substeps, n_actions=4,
                         initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)

        if self.reward_type == 'sparse':
            rs = (np.array(d) > self.distance_threshold)
            return - rs.any(axis=0).astype(np.float32) if isinstance(d, list) else - rs.astype(np.float32)
        elif self.reward_type == 'sparse-vec':
            return -(np.array(d) > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            return -np.array(d).sum(axis=0) if isinstance(d, list) else -d
        elif self.reward_type == "dense-vec":
            return -np.array(d)
        else:
            raise RuntimeError(f"`{self.reward_type}` is not a supported reward type.")

    # RobotEnv methods
    # ----------------------------
    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

        for obj_key in self.freeze_objects:
            self._reset_body(obj_key, self.initial_qpos[obj_key + ":joint"][:2])

        if self.goal_sampling:
            goals = {}
            for obj_key, options in self.goal_sampling.items():
                if options.get('track', False):
                    target = options.get('target', obj_key)
                    offset = options.get('offset', 0)
                    goals[obj_key] = self.sim.data.get_site_xpos(target) + offset

            if isinstance(self.goal_key, str):
                self.goal = goals[self.goal_key]
            else:
                self.goal = deepcopy(assign({}, self.goal, goals))

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # gripper state
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs_stack = grip_pos, gripper_vel, gripper_state, grip_velp,

        for obj_key in self.obs_keys or self.obj_keys:
            obj_key, *attrs = obj_key.split('@')
            obs_dict = dict(
                pos=self.sim.data.get_site_xpos(obj_key),
                rot=rotations.mat2euler(self.sim.data.get_site_xmat(obj_key)),
                rel_vel=self.sim.data.get_site_xvelp(obj_key) * dt - grip_velp,
                vel_rot=self.sim.data.get_site_xvelr(obj_key) * dt,
            )
            obs_dict['rel_pos'] = obs_dict['pos'] - grip_pos

            obs_stack += tuple([obs_dict[k] for k in attrs or obs_dict.keys()])

        if isinstance(self.goal_key, str):
            achieved_goal = self.sim.data.get_site_xpos(self.goal_key)
        else:
            achieved_goal = {k: self.sim.data.get_site_xpos(k) for k in self.goal_key}

        return {
            'observation': np.concatenate(obs_stack).copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _set_target(self, target_id, goal):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id(target_id)
        # second site_id was 0
        self.sim.model.site_pos[site_id] = goal - sites_offset[site_id]

    def _render_callback(self):
        # Visualize target.
        if isinstance(self.goal, dict):
            for i, obj_key in enumerate(self.goal_key):
                self._set_target(f"target{i}", self.goal[obj_key])
        else:
            self._set_target("target0", self.goal)

        self.sim.forward()

    def _reset_slide(self, obj_key=None, slide_pos=None, name=None):
        if slide_pos is None:
            ctrl_ind = self.sim.model.joint_names.index(f"{obj_key}:slide")
            low, high = self.sim.model.jnt_range[ctrl_ind]
            slide_pos = self.np_random.uniform(low, high)
        self.sim.data.set_joint_qpos(name or f'{obj_key}:slide', slide_pos)

    def _set_obj_in_hand(self, obj_key, hand_pos=None):
        pos = self.sim.data.get_joint_qpos(f'{obj_key}:joint')
        if hand_pos is None:
            pos[:3] = self.sim.data.get_site_xpos('robot0:grip')
        pos[3:] = 0
        self.sim.data.set_joint_qpos(f'{obj_key}:joint', pos)
        self._reset_slide(name="robot0:r_gripper_finger_joint", slide_pos=0.02)
        self._reset_slide(name="robot0:l_gripper_finger_joint", slide_pos=0.02)

    def _reset_body(self, obj_key, pos=None, track="gripper", avoid="gripper", offset=0,
                    range=0.15, d_min=0.1, h=None, m={}):
        current_obj_qpos = self.sim.data.get_joint_qpos(f'{obj_key}:joint').copy()
        if pos is not None:
            current_obj_qpos[:len(pos)] = pos
            self.sim.data.set_joint_qpos(f'{obj_key}:joint', current_obj_qpos)
            return current_obj_qpos

        if isinstance(range, numbers.Number):
            range = (-range, range)

        center_xpos = self.initial_gripper_xpos if track == 'gripper' else m.get(track)

        if avoid:
            if isinstance(avoid, str):
                avoid = [avoid]

            avoid_xposes = np.array(
                [self.initial_gripper_xpos if key == 'gripper' else m.get(key)
                 for key in avoid])

            xpos = avoid_xposes[0]
            while (np.linalg.norm(xpos[:2] - avoid_xposes[:, :2], axis=-1) < d_min).any():
                xpos = center_xpos + ([*offset, 0] if hasattr(offset, 'size') and offset.size == 2 else offset)
                xpos[:2] += self.np_random.uniform(*range, size=2)
        else:
            xpos = center_xpos + ([*offset, 0] if hasattr(offset, 'size') and offset.size == 2 else offset)
            xpos[:2] += self.np_random.uniform(*range, size=2)

        # if not hasattr(offset, 'size') or offset.size < 3:
        if not hasattr(offset, 'size') or offset.size < 3:
            xpos[2] = h or self.initial_heights[obj_key]

        current_obj_qpos[:3] = xpos
        # todo: add h_high argument to uniformly sample
        self.sim.data.set_joint_qpos(f'{obj_key}:joint', current_obj_qpos)
        return xpos

    def _reset_sim(self):
        """use forward=False to compose with this. super()._reset_sim(forward=True) call
        should always happen first to allow write by custom code.

        Make sure that you cal `self.sim.set_state` first if you complete overwrite this.
        It is critical.
        """
        self.sim.set_state(self.initial_state)
        # Randomize start position. Object in pick and place, gripper in reach.
        if self.obj_keys is not None:
            cache = {}
            for obj_key in self.freeze_objects:
                cache[obj_key] = self._reset_body(obj_key, self.initial_qpos[obj_key + ":joint"][:2])[:3]
            for obj_key in [k for k in self.obj_keys if k not in self.freeze_objects]:
                options = (self.obj_reset or {}).get(obj_key, {})
                cache[obj_key] = self._reset_body(obj_key, m=cache, **options)

        elif self.goal_key is None:
            pass
        elif isinstance(self.goal_key, str):
            if 'robot' not in self.goal_key:
                self._reset_body(self.goal_key)

        self.sim.forward()
        return True

    # only used by _sample_goal
    def _sample_single_goal(self, goal_key=None, h=None, high=0.45, range=None, in_the_air=None):
        if range is None:
            range = self.target_range
        if in_the_air is None:
            in_the_air = self.target_in_the_air or 0

        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-range, range, size=3)
        goal += self.target_offset
        # sets the goal to the table top.
        goal[2] = h or self.initial_heights[goal_key]
        # todo: refactor target_in_the_air to be an argument
        if not h and self.np_random.uniform() < in_the_air:
            goal[2] += self.np_random.uniform(0, high)
        return goal

    def _sample_goal(self):
        if self.goal_key == "robot0:grip":
            return self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)

        goal_keys = [self.goal_key] if isinstance(self.goal_key, str) else self.goal_key
        goal_keys = [k for k in goal_keys if k not in self.goal_sampling]
        goals = {k: self._sample_single_goal(k) for k in goal_keys}
        for obj_key, options in self.goal_sampling.items():
            assert obj_key in self.goal_key, f"{obj_key} is not part of the goal spec"
            if "target" in options:
                offset = options.get('offset', 0)
                goals[obj_key] = self.sim.data.get_site_xpos(options['target']) + offset
            else:
                goals[obj_key] = self._sample_single_goal(obj_key, **options)

        return goals[self.goal_key] if isinstance(self.goal_key, str) else goals

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)

        rs = np.array(d) < self.distance_threshold
        if isinstance(d, list):
            rs = rs.all()
        return rs.astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) \
                         + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_heights = {}
        for obj_key in self.obj_keys:
            self.initial_heights[obj_key] = self.sim.data.get_site_xpos(obj_key)[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
