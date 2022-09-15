import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time

env = RandomizedEnvWrapper(gym.make('HalfCheetahRandomizedEnv-v0'), seed=123)
# env=gym.make('Ant-v2')

obs = env.reset()
for i in range(2000):
    obs, _, done, _ = env.step(np.zeros((6)))
    # env.render()
    if i % 100 == 0:
        env.randomize([-1]*5,return_env_params=True)
        env.reset()

