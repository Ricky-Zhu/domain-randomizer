import numpy as np
from gym.envs.mujoco import InvertedDoublePendulumEnv
from randomizer.safe_env.wrappers.safe_env import SafeEnv
import os
import xml.etree.ElementTree as et
from typing import Dict, Tuple
from gym.envs.mujoco import mujoco_env
import json
import torch
import mujoco_py
from gym import spaces
from randomizer.safe_env.utils import Array

double_pendulum_cfg = dict(
    action_dim=1,
    action_range=[
        -1,
        1],
    unsafe_reward=-200.,
    saute_discount_factor=1.0,
    max_ep_len=200,
    min_rel_budget=1.0,
    max_rel_budget=1.0,
    test_rel_budget=1.0,
    use_reward_shaping=True,
    use_state_augmentation=True

)


class DoublePendulumEnv(InvertedDoublePendulumEnv):
    """Custom double pendulum."""

    def __init__(self, mode="train"):
        assert mode == "train" or mode == "test" or mode == "deterministic", "mode can be deterministic, test or train"
        self._mode = mode
        super().__init__()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state, reward, done, info = super().step(action)
        reward /= 10.  # adjusting the reward to match the cost
        return next_state, reward, done, info


class SafeDoublePendulumEnv(SafeEnv, DoublePendulumEnv):
    """Safe double pendulum."""

    def __init__(self, **kwargs):
        self.unsafe_min = np.pi * (-25. / 180.)
        self.unsafe_max = np.pi * (75. / 180.)
        self.unsafe_middle = 0.5 * (self.unsafe_max + self.unsafe_min)
        self.max_distance = 0.5 * (self.unsafe_max - self.unsafe_min)
        super().__init__(**kwargs)

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        """Computes a linear safety cost between the current position
        (if its near the unsafe area, aka in the hazard region)
        and the centre of the unsafe region."""
        assert type(state) is np.ndarray and type(next_state) is np.ndarray and type(
            action) is np.ndarray, "Arguments must be np.ndarray"
        thetas = np.arctan2(state[..., 1], state[..., 3])
        dist_to_center = np.abs(self.unsafe_middle - thetas)
        unsafe_mask = np.float64(((self.unsafe_min) <= thetas) & (thetas <= (self.unsafe_max)))
        costs = ((self.max_distance - dist_to_center) / (self.max_distance)) * unsafe_mask
        return costs


class RandomizeSafeDoublePendulumEnv(SafeDoublePendulumEnv):
    def __init__(self, with_var=False, **kwargs):
        self.with_var = with_var
        self.reference_path = os.path.join(os.path.dirname(mujoco_env.__file__), "assets",
                                           "inverted_double_pendulum.xml")
        self.reference_xml = et.parse(self.reference_path)
        self.root = self.reference_xml.getroot()
        super().__init__(**kwargs)

    def set_with_var(self, with_var):
        self.with_var = with_var

    def set_values(self, cart_mean=0.1, pole_mean=0.6, cart_var=None, pole_var=None):
        if self.with_var and cart_var is not None and pole_var is not None:
            cart = np.random.normal(loc=cart_mean, scale=np.sqrt(cart_var))
            pole = np.random.normal(loc=pole_mean, scale=np.sqrt(pole_var))
        else:
            cart = cart_mean
            pole = pole_mean

        # modify the cart size
        cart_ref = self.root.find(".//geom[@name='cart']")
        cart_value = "{:3f} {:3f}".format(cart, cart)
        cart_ref.set('size', "{}".format(cart_value))

        # modify the pole length
        cpole_ref = self.root.find(".//geom[@name='cpole']")
        cpole_ref.set('fromto', "0 0 0 0 0 {:3f}".format(pole))

        cpole2_body_ref = self.root.find(".//body[@name='pole2']")
        cpole2_body_ref.set('pos', "0 0 {:3f}".format(pole))
        cpole2_ref = self.root.find(".//geom[@name='cpole2']")
        cpole2_ref.set("fromto", "0 0 0 0 0 {:3f}".format(pole))

        # renew the monitor site
        site_ref = self.root.find(".//site[@name='tip']")
        site_ref.set("pos", "0 0 {:3f}".format(pole))

        new_xml = et.tostring(self.root, encoding='unicode', method='xml')
        self._re_init(new_xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)


class SautedRandomizableDoublePendulumEnv(RandomizeSafeDoublePendulumEnv):
    """Sauted safe double pendulum."""

    def __init__(
            self,
            safety_budget: float = 40.0,
            saute_discount_factor: float = 0.99,
            max_ep_len: int = 200,
            min_rel_budget: float = 1.,  # minimum relative (with respect to safety_budget) budget
            max_rel_budget: float = 1.,  # maximum relative (with respect to safety_budget) budget
            test_rel_budget: float = 1.,  # test relative budget
            unsafe_reward: float = -200.,
            use_reward_shaping: bool = True,  # ablation
            use_state_augmentation: bool = True,  # ablation
            **kwargs
    ):
        assert safety_budget > 0, "Please specify a positive safety budget"
        assert saute_discount_factor > 0 and saute_discount_factor <= 1, "Please specify a discount factor in (0, 1]"
        assert min_rel_budget <= max_rel_budget, "Minimum relative budget should be smaller or equal to maximum relative budget"
        assert max_ep_len > 0

        self._safety_state = 1.
        self.use_reward_shaping = use_reward_shaping
        self.use_state_augmentation = use_state_augmentation
        self.max_ep_len = max_ep_len
        self.min_rel_budget = min_rel_budget
        self.max_rel_budget = max_rel_budget
        self.test_rel_budget = test_rel_budget
        if saute_discount_factor < 1:
            safety_budget = safety_budget * (1 - saute_discount_factor ** self.max_ep_len) / (
                    1 - saute_discount_factor) / self.max_ep_len
        self._safety_budget = np.float32(safety_budget)

        self._saute_discount_factor = saute_discount_factor
        self._unsafe_reward = unsafe_reward
        super().__init__(**kwargs)

        self.action_space = self.action_space


    @property
    def safety_budget(self):
        return self._safety_budget

    @property
    def saute_discount_factor(self):
        return self._saute_discount_factor

    @property
    def unsafe_reward(self):
        return self._unsafe_reward

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        state = super().reset()
        if self._mode == "train":
            self._safety_state = self.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)
        elif self._mode == "test" or self._mode == "deterministic":
            self._safety_state = self.test_rel_budget
        else:
            raise NotImplementedError("this error should not exist!")
        augmented_state = self._augment_state(state, self._safety_state)
        return augmented_state

    def _augment_state(self, state: np.ndarray, safety_state: np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, safety_state]) if self.use_state_augmentation else state
        return augmented_state

    def safety_step(self, cost: np.ndarray) -> np.ndarray:
        """ Update the normalized safety state z' = (z - l / d) / gamma. """
        self._safety_state -= cost / self.safety_budget
        self._safety_state /= self.saute_discount_factor
        return self._safety_state

    def step(self, action):
        """ Step through the environment. """

        next_obs, reward, done, info = super().step(action)
        next_safety_state = self.safety_step(info['cost'])
        info['true_reward'] = reward
        info['next_safety_state'] = next_safety_state
        reward = self.reshape_reward(reward, next_safety_state)
        augmented_state = self._augment_state(next_obs, next_safety_state)
        return augmented_state, reward, done, info

    def reshape_reward(self, reward: Array, next_safety_state: Array):
        """ Reshaping the reward. """
        if self.use_reward_shaping:
            reward = reward * (next_safety_state > 0) + self.unsafe_reward * (next_safety_state <= 0)
        return reward

    def reward_fn(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """ Compute rewards in a batch. """
        reward = super()._reward_fn(states, actions, next_states, is_tensor=True)
        if self.use_state_augmentation:
            # shape reward for model-based predictions
            reward = self.reshape_reward(reward, next_states[:, -1].view(-1, 1))
        return reward
