from gym import Wrapper
import numpy as np
from gym.spaces import Box


class GCPWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_high = self.observation_space['observation'].high
        obs_low = self.observation_space['observation'].low
        obs_high = np.array(np.hstack([obs_high, np.inf, np.inf, np.inf]), dtype=np.float32)
        obs_low = np.array(np.hstack([obs_low, np.inf, np.inf, np.inf]), dtype=np.float32)
        self.observation_space = Box(high=obs_high, low=obs_low)

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = np.concatenate([obs['observation'], obs['desired_goal']])
        return obs, r, done, info
