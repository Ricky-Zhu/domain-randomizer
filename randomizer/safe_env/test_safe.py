# import randomizer.safe_env.pendulum
# import randomizer
import gym
import numpy as np
# from randomizer.wrappers import RandomizedEnvWrapper
import time
import gym
env_name = 'SafePendulum-v0'
env = gym.make(env_name)
env.reset()
for i in range(1000):
    env.render()
    s,r,d,_=env.step(env.action_space.sample())
    if d:
        env.reset()