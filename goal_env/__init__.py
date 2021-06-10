import sys

sys.path.append('../')
from gym.envs.registration import register
import gym
from goal_env.bitflip import BitFlipEnv
from goal_env.fourroom import FourRoom, FourRoom2, FourRoom3, FourRoom4
from goal_env.mountaincar import MountainCarEnv
from goal_env.plane import NaivePlane, NaivePlane2, NaivePlane3, NaivePlane4, NaivePlane5
from goal_env.goal_plane_env import GoalPlane

register(
    id='Bitflip-v0',
    entry_point='goal_env.bitflip:BitFlipEnv',
    kwargs={'num_bits': 11},
    max_episode_steps=200,
    reward_threshold=100.0,
    nondeterministic=False,
)

register(
    id='FourRoom-v0',
    entry_point='goal_env.fourroom:FourRoom',
    kwargs={'goal_type': 'fix_goal'},
    max_episode_steps=200,
    reward_threshold=100.0,
    nondeterministic=False,
)

register(
    id='FourRoom-v1',
    entry_point='goal_env.fourroom:FourRoom2',
    kwargs={'goal_type': 'fix_goal'},
    max_episode_steps=200,
    reward_threshold=100.0,
    nondeterministic=False,
)

register(
    id='FourRoom-v2',
    entry_point='goal_env.fourroom:FourRoom3',
    kwargs={'goal_type': 'fix_goal'},
    max_episode_steps=200,
    reward_threshold=100.0,
    nondeterministic=False,
)

register(
    id='FourRoom-v4',
    entry_point='goal_env.fourroom:FourRoom4',
    kwargs={'goal_type': 'fix_goal'},
    max_episode_steps=200,
    reward_threshold=100.0,
    nondeterministic=False,
)

register(
    id='mcar-v0',
    entry_point='goal_env.mountaincar:MountainCarEnv',
    kwargs={'goal_dim': 1},
    max_episode_steps=200,
    reward_threshold=100.0,
    nondeterministic=False,
)

register(
    id='Plane-v0',
    entry_point='goal_env.plane:NaivePlane5',
)

register(
    id='GoalPlane-v0',
    entry_point='goal_env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneMid-v0',
    entry_point='goal_env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "type": "mid",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneHard-v0',
    entry_point='goal_env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "type": "hard",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneEasy-v0',
    entry_point='goal_env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "type": "easy",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneTest-v0',
    entry_point='goal_env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
        "goals": (2.5, 12.5)
    }
)
