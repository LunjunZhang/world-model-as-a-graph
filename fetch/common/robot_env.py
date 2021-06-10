import os
import copy
import numpy as np

import gym
from fetch.common.utils import goal_distance
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)")

DEFAULT_SIZE = 500


def obs_spec(obs):
    if isinstance(obs, dict):
        return spaces.Dict({k: obs_spec(v) for k, v in obs.items()})
    return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps,
                 view_mode='rgb', width=None, height=None):
        self.view_mode = view_mode
        self.width = width or DEFAULT_SIZE
        self.height = height or DEFAULT_SIZE
        self.initial_qpos = initial_qpos

        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), '../assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'obs_spec.modes': ['human', 'rgb_array', 'rgb', 'rgbd'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = obs_spec(obs)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        success = self._is_success(obs['achieved_goal'], self.goal)
        info = {
            'is_success': success,
            'success': success,
            'dist': goal_distance(obs['achieved_goal'], self.goal)
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        mode = mode or self.view_mode
        width = width or self.width
        height = height or self.height
        viewer = self._get_viewer(mode)

        self._render_callback()

        if mode in ['rgb', 'rgb_array']:
            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            return data[::-1, :, :].astype(np.uint8)
        elif mode == 'rgbd':
            viewer.render(width, height)
            rgb, d = viewer.read_pixels(width, height, depth=True)
            return rgb[::-1, :, :], d[::-1, :]
        elif mode == 'depth':
            viewer.render(width, height)
            _, d = viewer.read_pixels(width, height, depth=True)
            return d[::-1, :]
        elif mode == 'grey':
            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :].mean(axis=-1).astype(np.uint8)
        elif mode == "notebook":
            from IPython.display import display
            from PIL import Image

            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            display(Image.fromarray(data[::-1, :, :]))

        elif mode == 'human':
            if width and height:
                import glfw
                glfw.set_window_size(viewer.window, width, height)
            viewer.render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            else:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
