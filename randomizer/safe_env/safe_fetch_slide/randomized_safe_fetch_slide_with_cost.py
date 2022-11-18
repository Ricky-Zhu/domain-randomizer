from randomizer.safe_env.safe_fetch_slide.safe_fetch_slide import SafeFetchSlideEnv
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

class SafeFetchSlideWithCost(SafeEnvSlide, SafeFetchSlideEnv):
    """Safe fetch_slide env with cost function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,danger_size,obj_radi_info,dangers) -> np.ndarray:
        # assert type(state) is np.ndarray and type(next_state) is np.ndarray and type(
        #     action) is np.ndarray, "Arguments must be np.ndarray"
        coef = 1.0
        total_cost = 0
        object_position = state['achieved_goal'][:2]
        # print('object position')
        # print(object_position)
        for i in range(len(dangers)):
            tem_name = 'danger{}'.format(i)
            danger_position = dangers[tem_name]
            distance = np.linalg.norm(object_position - danger_position)
            total_radi = danger_size + obj_radi_info
            overshoot = total_radi - distance
            modified_distance = max(overshoot,0)
            # modified_distance = overshoot
            cost = modified_distance * coef
            total_cost = total_cost + cost
        print(total_cost)
        return total_cost



class RandomizeSafeFetchSlideCostEnv(SafeFetchSlideWithCost):
    def __init__(self, with_var=True, **kwargs):
        self.with_var = with_var
        self.reference_path_randomize = dirname(dirname(os.path.abspath(__file__))) + '/assets/robotics/fetch/slide.xml'
        # self.reference_path = os.path.join(os.path.dirname(mujoco_env.__file__), "assets",
        #                                    "inverted_double_pendulum.xml")
        # self.reference_path_randomize = "/home/liqun/domain-randomizer-master/randomizer/safe_env/assets/robotics/fetch/slide.xml"
        self.reference_xml_randomize = et.parse(self.reference_path_randomize)
        self.root_randomize = self.reference_xml_randomize.getroot()
        super().__init__(**kwargs)

    def set_with_var(self, with_var):
        self.with_var = with_var

    def set_friction_coefficient_value(self,value, automatic = False, mean = [0.1, 0.005,0.0001], var = [0.01,0.000001,0.00000001]):
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
                value2 = np.random.normal(loc=mean[2], scale=np.sqrt(var[2]))
                target_value.append(value2)
            else:
                target_value = mean
        # # change the friction parameter for table
        # table_body = self.root_randomize.find(".//body[@name='table0']/geom")
        # target_value_str = "{:3f} {:3f} {:3f}".format(target_value[0], target_value[1], target_value[2])
        # table_body.set('friction', '{}'.format(target_value_str))
        #
        # # change the friction parameter for object
        # object_body = self.root_randomize.find(".//body[@name='object0']/geom")
        # object_body.set('friction', '{}'.format(target_value_str))
        #
        # new_xml_randomize = et.tostring(self.root_randomize, encoding='unicode', method='xml')
        # self._re_init_randomize(new_xml_randomize)
        #
        # table_body = self.root_randomize.find(".//body[@name='table0']/geom")
        # table_fric = table_body.get('friction')
        # print('table_fric')
        # print(table_fric)
        #
        # object_body = self.root_randomize.find(".//body[@name='object0']/geom")
        # obj_fric = object_body.get('friction')
        # print('obj_fric')
        # print(obj_fric)


        self.sim.model.geom_friction[-2] = target_value
        self.sim.model.geom_friction[-1] = target_value

        print('Now the frictions are')
        for i in range(len(self.sim.model.geom_friction)):
            print(self.sim.model.geom_friction[i])


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
        # print('geom_size')
        # print(self.sim.model.geom_size)
        # print('object0')
        # print(self.sim.model.geom_size[-1])
        # print('target_value')
        # print(target_value)
        self.sim.model.geom_size[-1] = target_value
        print('object0 size')
        print(self.sim.model.geom_size[-1])
        print('object radi')
        print(self.sim.model.geom_size[-1][0])

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
        for order in range(4):
            self.sim.model.site_size[order] = target_value_list
        print('site_size')
        print(self.sim.model.site_size)
        print('danger radi')
        print(self.sim.model.site_size[0][0])



    def set_danger_offset_value(self,value,automatic = False, mean = 0.1, var = 0.005):
        if not automatic:
            target_value = value
        else:
            if self.with_var == True:
                assert var is not None
                target_value = np.random.normal(loc=mean, scale=np.sqrt(var))
            else:
                target_value = mean
        # SafeFetchSlideEnv.set_offset(target_value)
        # FetchEnv.set_offset(target_value)
        self.keep_dist = target_value



    # def _re_init_randomize(self, xml):
    #     self.model = mujoco_py.load_model_from_xml(xml)
    #     self.sim = mujoco_py.MjSim(self.model)
    #     self.data = self.sim.data
    #     self.init_qpos = self.data.qpos.ravel().copy()
    #     self.init_qvel = self.data.qvel.ravel().copy()
    #     observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
    #     assert not done
    #     if self.viewer:
    #         self.viewer.update_sim(self.sim)