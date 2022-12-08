import randomizer.safe_env.pendulum
import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time
import gym
# env_name = 'RandomizeSafeFetchSlideCostEnv-v0'
env_name = 'RandomizeSafeFetchSlideCostSimpleEnv-v0'

env = gym.make(env_name)
env.reset()
for i in range(1000):
    env.render()
    s,r,d,info=env.step(env.action_space.sample())
    if d:
        env.reset()
        env.set_friction_coefficient_value([0,0,0],automatic=True)
        env.set_object_size_value([0,0],automatic=True)
        env.set_radi_danger_region_value(0,automatic=True)
        env.set_danger_offset_value(0.03,automatic=True)

