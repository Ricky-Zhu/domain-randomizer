from gym import Env
import numpy as np


class SafeEnvSlide(Env):
    """Safe environment wrapper."""
    def step(self, action:np.ndarray) -> np.ndarray:
        state = self._get_state()
        dangers = self._get_dangers_positions()
        danger_size = self._get_danger_size()
        obj_radi_info = self._get_obj_radi()
        next_state, reward, done, info = super().step(action)
        info['cost'] = self._safety_cost_fn(state, action, next_state,danger_size,obj_radi_info,dangers)
        info['danger_position'] = dangers
        info['danger_size'] = danger_size
        return next_state, reward, done, info

    def _get_dangers_positions(self):
        if hasattr(self, "_get_dangers_pos"):
            return self._get_dangers_pos()
        else:
            raise NotImplementedError("Please implement _get_dangers_pos method returning the current state")


    def _get_state(self):
        """Returns current state. Uses _get_obs() method if it is implemented."""
        if hasattr(self, "_get_obs"):
            return self._get_obs()
        else:
            raise NotImplementedError("Please implement _get_obs method returning the current state")

    def _get_danger_size(self):
        if hasattr(self, "_get_dangers_size"):
            return self._get_dangers_size()
        else:
            raise NotImplementedError("Please implement _get_dangers_size method returning the danger info")

    def _get_obj_radi(self):
        if hasattr(self, "_get_obj_size"):
            return self._get_obj_size()
        else:
            raise NotImplementedError("Please implement _get_obj_size method returning the danger info")



    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        """Returns current safety cost."""
        raise NotImplementedError("Please implement _safety_cost_fn method returning the current safety cost")