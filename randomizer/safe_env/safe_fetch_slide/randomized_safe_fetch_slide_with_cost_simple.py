from randomizer.safe_env.safe_fetch_slide.safe_fetch_slide_simple import SafeFetchSlideSimpleEnv
import numpy as np
from os.path import dirname
from gym.envs.mujoco import InvertedDoublePendulumEnv
from randomizer.safe_env.wrappers.saute_env import saute_env
from randomizer.safe_env.wrappers.safe_env_slide import SafeEnvSlide
import os
import xml.etree.ElementTree as et
from typing import Dict, Tuple
from gym.envs.mujoco import mujoco_env
import json
import mujoco_py

class SafeFetchSlideWithCostSimple(SafeEnvSlide, SafeFetchSlideSimpleEnv):
    """Safe fetch_slide env with cost function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,danger_size,obj_radi_info,dangers) -> np.ndarray:
        coef = 1.0
        total_cost = 0
        object_position = state['achieved_goal'][:2]
        for i in range(len(dangers)):
            tem_name = 'danger{}'.format(i)
            danger_position = dangers[tem_name]
            distance = np.linalg.norm(object_position - danger_position)
            total_radi = danger_size + obj_radi_info
            overshoot = total_radi - distance
            modified_distance = max(overshoot,0)
            cost = modified_distance * coef
            total_cost = total_cost + cost
        return total_cost



class RandomizeSafeFetchSlideCostSimpleEnv(SafeFetchSlideWithCostSimple):
    def __init__(self, with_var=True, **kwargs):
        self.with_var = with_var
        super().__init__(**kwargs)

    def set_with_var(self, with_var):
        self.with_var = with_var

    def set_friction_coefficient_value(self,value, automatic = False, mean = [1.0, 0.005,0.0001], var = [0.01,0.000001,0.00000001]):
        if not automatic:
            target_value = value
        else:
            target_value = mean      # do not randomize the friction coefficient, the value of the sliding friction coefficient is 1.0
            # if self.with_var == True:
            #     assert var is not None
            #     target_value = []
            #     value0 = np.random.normal(loc=mean[0], scale=np.sqrt(var[0]))
            #     target_value.append(value0)
            #     value1 = np.random.normal(loc=mean[1], scale=np.sqrt(var[1]))
            #     target_value.append(value1)
            #     value2 = np.random.normal(loc=mean[2], scale=np.sqrt(var[2]))
            #     target_value.append(value2)
            # else:
            #     target_value = mean


        self.sim.model.geom_friction[-2] = target_value
        self.sim.model.geom_friction[-1] = target_value


    def set_object_size_value(self,value,automatic = False, mean = [0.025, 0.02], var = [0.000000002,0.00000001]):
        if not automatic:
            target_value = value
        else:
            if self.with_var == True:
                assert var is not None
                target_value = []
                value0 = np.random.normal(loc=mean[0], scale=np.sqrt(var[0]))
                target_value.append(value0)
                value1 = np.random.normal(loc=mean[1], scale=np.sqrt(var[1]))
                target_value.append(value1)
            else:
                target_value = mean
        if len(target_value) < 3:
            target_value.append(0)
        self.sim.model.geom_size[-1] = target_value

    def set_radi_danger_region_value(self,value,automatic = False, mean = 0.05, var = 0.000002):
        if not automatic:
            target_value = value
        else:
            if self.with_var == True:
                assert var is not None
                target_value = np.random.normal(loc=mean, scale=np.sqrt(var))
            else:
                target_value = mean
        target_value_list = [target_value,target_value,0.001]
        for order in range(1):
            self.sim.model.site_size[order] = target_value_list



    def set_danger_offset_value(self,value,automatic = False, mean = 0.1, var = 0.005):
        if not automatic:
            target_value = value
        else:
            target_value = mean
            # if self.with_var == True:
            #     assert var is not None
            #     target_value = np.random.normal(loc=mean, scale=np.sqrt(var))
            # else:
            #     target_value = mean
        self._keep_dist = target_value

