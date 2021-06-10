import gym
import numpy as np
import cv2
from gym import spaces


def line_intersection(line1, line2):
    # calculate the intersection point
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0]
    [1] - line2[1][1])  # Typo was here
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def check_cross(x0, y0, x1, y1):
    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    return np.cross(x1 - x0, y0 - x0), np.cross(y0 - x0, y1 - x0)


def check_itersection(x0, y0, x1, y1):
    EPS = 1e-10
    
    def sign(x):
        if x > EPS:
            return 1
        if x < -EPS:
            return -1
        return 0
    
    f1, f2 = check_cross(x0, y0, x1, y1)
    f3, f4 = check_cross(x1, y1, x0, y0)
    if sign(f1) == sign(f2) and sign(f3) == sign(f4) and sign(f1) != 0 and sign(f3) != 0:
        return True
    return False


class PlaneBase(gym.Env):
    def __init__(self, rects, R, is_render=False, size=512):
        self.rects = rects
        self.n = len(self.rects)
        self.size = size
        self.map = np.ones((size, size, 3), dtype=np.uint8) * 255
        self.R = R
        self.R2 = R ** 2
        self.board = np.array(
            [[0, 0],
             [1, 1]],
            dtype='float32')
        
        self.action_space = gym.spaces.Box(
            low=-R, high=R, shape=(2,), dtype='float32')
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=(2,), dtype='float32')
        
        if is_render:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            self.image_name = 'image'
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if check_itersection(self.rects[i][0], self.rects[i][1], self.rects[j][0], self.rects[j][0]):
                    raise Exception("Rectangle interaction with each other")
        
        for ((x0, y0), (x1, y1)) in rects:
            x0, y0 = int(x0 * size), int(y0 * size)
            x1, y1 = int(x1 * size), int(y1 * size)
            cv2.rectangle(self.map, (x0, y0), (x1, y1), (0, 255, 0), 1)
            
            ps = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.int32)
            cv2.fillConvexPoly(self.map, ps, (127, 127, 127))
        
        self.state = (0, 0)
        self.reset()
    
    def restore(self, obs):
        self.state = (float(obs[0]), float(obs[1]))
    
    def rect_lines(self, rect):
        (x0, y0), (x1, y1) = rect
        yield (x0, y0), (x1, y0)
        yield (x1, y0), (x1, y1)
        yield (x1, y1), (x0, y1)
        yield (x0, y1), (x0, y0)
    
    def l2dist(self, x, y):
        return ((y[0] - x[0]) ** 2) + ((y[1] - x[1]) ** 2)
    
    def check_inside(self, p):
        EPS = 1e-10
        for i in self.rects:
            if p[0] > i[0][0] + EPS and p[0] < i[1][0] - EPS and p[1] > i[0][1] + EPS and p[1] < i[1][1] - EPS:
                return True
        return False
    
    def step(self, action):
        dx, dy = action
        l = 0.0001
        p = (self.state[0] + dx * l, self.state[1] + dy * l)
        if self.check_inside(p) or p[0] > 1 or p[1] > 1 or p[0] < 0 or p[1] < 0:
            return np.array(self.state), 0, False, {}
        
        dest = (self.state[0] + dx, self.state[1] + dy)
        
        md = self.l2dist(self.state, dest)
        
        _dest = dest
        line = (self.state, dest)
        
        for i in list(self.rects) + [self.board]:
            for l in self.rect_lines(i):
                if check_itersection(self.state, dest, l[0], l[1]):
                    inter_point = line_intersection(line, l)
                    d = self.l2dist(self.state, inter_point)
                    if d < md:
                        md = d
                        _dest = inter_point
        
        self.restore(_dest)
        return np.array(self.state), -md, False, {}
    
    def render(self, mode='human'):
        image = self.map.copy()
        x, y = self.state
        x = int(x * self.size)
        y = int(y * self.size)
        cv2.circle(image, (x, y), 5, (255, 0, 255), -1)
        if mode == 'human':
            cv2.imshow('image', image)
            cv2.waitKey(2)
        else:
            return image
    
    def reset(self):
        inside_rect = True
        while inside_rect:
            a, b = np.random.random(), np.random.random()
            inside_rect = self.check_inside((a, b))
        self.state = (a, b)
        return np.array(self.state)


class NaivePlane(PlaneBase):
    def __init__(self, is_render=True, R=300, size=512):
        PlaneBase.__init__(self,
                           [
                               np.array([[128, 128], [300, 386]]) / 512,
                               np.array([[400, 400], [500, 500]]) / 512,
                           ],
                           R, is_render=is_render, size=size),


class NaivePlane2(PlaneBase):
    # two rectangle
    def __init__(self, is_render=True, R=300, size=512):
        PlaneBase.__init__(self,
                           [
                               np.array([[64, 64], [256, 256]]) / 512,
                               np.array([[300, 128], [400, 500]]) / 512,
                           ],
                           R, is_render=is_render, size=size),


class NaivePlane3(PlaneBase):
    # four rectangle
    def __init__(self, is_render=True, R=300, size=512):
        PlaneBase.__init__(self,
                           [
                               np.array([[64, 64], [192, 192]]) / 512,
                               np.array([[320, 64], [448, 192]]) / 512,
                               np.array([[320, 320], [448, 448]]) / 512,
                               np.array([[64, 320], [192, 448]]) / 512,
                           ],
                           R, is_render=is_render, size=size),


class NaivePlane4(PlaneBase):
    # four rectangle
    def __init__(self, is_render=True, R=300, size=512):
        PlaneBase.__init__(self,
                           [
                               np.array([[64, 64], [192, 512]]) / 512,
                               np.array([[320, 64], [448, 512]]) / 512,
                           ],
                           R, is_render=is_render, size=size),


class NaivePlane5(PlaneBase):
    # four rectangle
    def __init__(self, is_render=False, R=300, size=512):
        PlaneBase.__init__(self,
                           [
                               np.array([[0, 1. / 3], [2. / 3, 2. / 3]]),
                           ],
                           R, is_render=is_render, size=size),


if __name__ == '__main__':
    env = NaivePlane5()
    obs = env.reset()
    while True:
        print(obs)
        env.render()
        while True:
            try:
                print('entering the dir (x, y)')
                act = input().strip().split(' ')
                act = float(act[0]) / 512, float(act[1]) / 512
                break
            except KeyboardInterrupt as e:
                raise e
            except:
                continue
        
        obs, reward, _, _ = env.step(act)
