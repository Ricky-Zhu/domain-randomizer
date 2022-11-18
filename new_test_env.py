# import randomizer.safe_env.pendulum
import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time
import gym
# env_name = 'FetchPushRandomizedEnv-v0'
# env_name = 'HumanoidRandomizedEnv-v0'
# env_name = 'SafePendulum-v0'
# env_name = 'SafeFetchSlide-v0'
# env_name = 'SafeDoublePendulum-v0'
# env_name = 'RandomizeSafeDoublePendulum-v0'
# env_name = 'SafeFetchSlideWithCostFn-v0'
env_name = 'RandomizeSafeFetchSlideCostEnv-v0'

env = gym.make(env_name)
env.reset()
for i in range(1000):
    env.render()
    s,r,d,_=env.step(env.action_space.sample())
    if d:
        env.reset()
        env.set_friction_coefficient_value([0,0,0],automatic=True)
        env.set_object_size_value([0,0],automatic=True)
        env.set_radi_danger_region_value(0,automatic=True)
        env.set_danger_offset_value(0.03,automatic=True)


        print('One round completed!')