import numpy as np
from gym.envs.mujoco import InvertedDoublePendulumEnv
from randomizer.safe_env.wrappers.saute_env import saute_env
from randomizer.safe_env.wrappers.safe_env import SafeEnv
import os
import xml.etree.ElementTree as et
from typing import Dict, Tuple
from gym.envs.mujoco import mujoco_env
import json
import mujoco_py

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
    def __init__(self, **kwargs):
        self.reference_path = os.path.join(os.path.dirname(mujoco_env.__file__), "assets",
                                           "inverted_double_pendulum.xml")
        self.reference_xml = et.parse(self.reference_path)
        self.root = self.reference_xml.getroot()
        super().__init__(**kwargs)

    def set_values(self, cart=0.1, pole=0.6):
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


@saute_env
class SautedDoublePendulumEnv(SafeDoublePendulumEnv):
    """Sauted safe double pendulum."""
