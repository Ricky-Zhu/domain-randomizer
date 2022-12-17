import os
import numpy as np
from os.path import dirname
from gym import utils
from randomizer.safe_env.safe_fetch_slide.fetch_simple_env import FetchSimpleEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = dirname(dirname(os.path.abspath(__file__))) + '/assets/robotics/fetch/slide_simple.xml'
print(MODEL_XML_PATH)


class SafeFetchSlideSimpleEnv(FetchSimpleEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.25,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.41, 1., 0., 0., 0.],
        }
        FetchSimpleEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.1, 0.0, 0.0]),
            obj_range=0.05, target_range=0.15, distance_threshold=0.015,
            initial_qpos=initial_qpos, danger_regions_num=1, reward_type=reward_type)

        utils.EzPickle.__init__(self)
