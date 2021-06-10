import gym

if __name__ == '__main__':

    env = gym.make("fetch:Bin-pick-v0")
    print(env.observation_space)