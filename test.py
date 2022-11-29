import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time

env_name = 'SauteRandomizeSafeDoublePendulum-v0'


env=gym.make(env_name)
# env=gym.make('HalfCheetah-v2')q
# params = env.randomize(env.randomized_default, return_env_params=True)
obs = env.reset()
step=0
for i in range(2000):

    obs, r, done, info = env.step(np.zeros(1))
    step+=1
    if step % 200==0:
        env.set_values(0.2,0.7)
    if done:

        env.reset()
    # if i % 100 == 0:
    #     params = env.randomize(env.randomized_default, return_env_params=True)
    #     env.reset()
