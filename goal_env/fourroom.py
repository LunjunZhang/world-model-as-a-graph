## importance resampling
import gym
import numpy as np
from gym import spaces


class FourRoom(gym.Env):
    def __init__(self, seed=None, goal_type='fix_goal'):
        self.n = 11
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.goal_type = goal_type
        self.goal = None
        self.init()

    def init(self):
        self.observation_space = {
            'observation': spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32),
            'desired_goal': spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32)
        }
        self.observation_space['observation'].n = self.n
        self.dx = [0, 1, 0, -1]
        self.dy = [1, 0, -1, 0]
        self.action_space = spaces.Discrete(len(self.dx))
        self.reset()

    def label2obs(self, x, y):
        a = np.zeros((self.n*self.n,))
        assert self.x < self.n and self.y < self.n
        a[x * self.n + y] = 1
        return a

    def get_obs(self):
        assert self.goal is not None
        return {
            'observation': self.label2obs(self.x, self.y),
            'desired_goal': self.label2obs(*self.goal),
            'achieved_goal': self.label2obs(self.x, self.y),
        }

    def reset(self):
        condition = True
        while condition:
            self.x = np.random.randint(1, self.n)
            self.y = np.random.randint(1, self.n)
            condition = (self.map[self.x, self.y] == 0)

        loc = np.where(self.map > 0.5)
        assert len(loc) == 2
        if self.goal_type == 'random':
            goal_idx = np.random.randint(len(loc[0]))
        elif self.goal_type == 'fix_goal':
            goal_idx = 0
        else:
            raise NotImplementedError
        self.goal = loc[0][goal_idx], loc[1][goal_idx]
        self.done = False
        return self.get_obs()

    def step(self, action):
        #assert not self.done
        nx, ny = self.x + self.dx[action], self.y + self.dy[action]
        info = {'is_success': False}
        #before = self.get_obs().argmax()
        if self.map[nx, ny]:
            self.x, self.y = nx, ny
            reward = -1
            done = False
        else:
            reward = -1
            done = False
        if nx == self.goal[0] and ny == self.goal[1]:
            reward = 0
            info = {'is_success': True}
            done = self.done = True
        return self.get_obs(), reward, done, info

    def compute_reward(self, state, goal, info):
        state_obs = state.argmax(axis=1)
        goal_obs = goal.argmax(axis=1)
        reward = np.where(state_obs == goal_obs, 0, -1)
        return reward

    def restore(self, obs):
        obs = obs.argmax()
        self.x = obs//self.n
        self.y = obs % self.n

    def bfs_dist(self, state, goal):
        #using bfs to search for shortest path
        visited = {key: False for key in range(self.n*self.n)}
        state_key = state.argmax()
        goal_key = goal.argmax()
        queue = []
        visited[state_key] = True
        queue.append(state_key)
        dist = [-np.inf] * (self.n*self.n)
        dist[state_key] = 0

        while (queue):
            par = queue.pop(0)
            if par == goal_key:
                break
            x_par, y_par = par // self.n, par % self.n
            for action in range(4):
                x_child, y_child = x_par + self.dx[action], y_par + self.dy[action]
                child = x_child*self.n + y_child
                if self.map[x_child, y_child] == 0:
                    continue
                if visited[child] == False:
                    visited[child] = True
                    queue.append(child)
                    dist[child] = dist[par] + 1

        return dist[goal_key]

    def get_pairwise(self, state, target):
        dist = self.bfs_dist(state, target)
        return dist

    def all_states(self):
        states = []
        mask = []
        for i in range(self.n):
            for j in range(self.n):
                self.x = i
                self.y = j
                states.append(self.get_obs())
                if isinstance(states[-1], dict):
                    states[-1] = states[-1]['observation']
                mask.append(self.map[self.x, self.y] > 0.5)
        return np.array(states)[mask]

    def all_edges(self):
        A = np.zeros((self.n*self.n, self.n*self.n))
        mask = []
        for i in range(self.n):
            for j in range(self.n):
                mask.append(self.map[i, j] > 0.5)
                if self.map[i][j]:
                    for a in range(4):
                        self.x = i
                        self.y = j
                        t = self.step(a)[0]
                        if isinstance(t, dict):
                            t = t['observation']
                        self.restore(t)
                        A[i*self.n+j, self.x*self.n + self.y] = 1
        return A[mask][:, mask]


class FourRoom2(FourRoom):
    def __init__(self, *args, **kwargs):
        FourRoom.__init__(self, *args, **kwargs)
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))


class FourRoom3(FourRoom):
    def __init__(self, *args, **kwargs):
        FourRoom.__init__(self, *args, **kwargs)
        self.n = 5
        self.map = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.init()


class FourRoom4(FourRoom):
    def __init__(self, seed=None, *args, **kwargs):
        FourRoom.__init__(self, *args, **kwargs)
        self.n = 16
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.init()


if __name__ == '__main__':
    a = FourRoom()