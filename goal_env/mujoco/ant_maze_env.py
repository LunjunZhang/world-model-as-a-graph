from .maze_env import MazeEnv
from .ant import AntEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv
