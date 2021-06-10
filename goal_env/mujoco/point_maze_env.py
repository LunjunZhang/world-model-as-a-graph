from .maze_env import MazeEnv
from .point import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
