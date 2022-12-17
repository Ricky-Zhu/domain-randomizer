from randomizer.safe_env.safe_fetch_slide.safe_fetch_slide_simple import SafeFetchSlideSimpleEnv
import numpy as np
from randomizer.safe_env.wrappers.saute_env import saute_env
from randomizer.safe_env.wrappers.safe_env_slide import SafeEnvSlide


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SafeFetchSlideWithCostSimple(SafeEnvSlide, SafeFetchSlideSimpleEnv):
    """Safe fetch_slide env with cost function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            object_pos = self.sim.data.get_site_xpos('object0')
            dist_obj_grip = np.linalg.norm(grip_pos - object_pos)
            rew = -d + ((-dist_obj_grip) * 10)
            return rew

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, danger_size, obj_radi_info,
                        dangers) -> np.ndarray:
        coef = 0.1
        total_cost = 0
        object_position = state['achieved_goal'][:2]
        for i in range(len(dangers)):
            tem_name = 'danger{}'.format(i)
            danger_position = dangers[tem_name]
            max_dis = np.sqrt(0.85)
            distance = np.linalg.norm(object_position - danger_position)
            total_radi = danger_size + obj_radi_info
            overshoot = total_radi - distance
            if overshoot >= 0:
                dis_cost = (max_dis - distance) * coef
                modi_dis_cost = max(0,dis_cost)
                cost = modi_dis_cost + 5.0
            else:
                dis_cost = (max_dis - distance) * coef
                cost = max(0,dis_cost)
            # modified_distance = max(overshoot, 0)
            # cost = modified_distance * coef
            total_cost = total_cost + cost
        return total_cost


class RandomizeSafeFetchSlideCostSimpleEnv(SafeFetchSlideWithCostSimple):
    def __init__(self, with_var=True, mode='train', augment_hazard_info_in_state=False, **kwargs):
        self.with_var = with_var
        self._mode = mode
        self.augment_hazard_info_in_state = augment_hazard_info_in_state
        super().__init__(**kwargs)

    def set_with_var(self, with_var):
        self.with_var = with_var

    def set_friction_coefficient_value(self, value, automatic=False, mean=[0.3, 0.005, 0.0001],
                                       var=[0.01, 0.000001, 0.00000001]):
        if not automatic:
            target_value = value
        else:
            target_value = mean  # do not randomize the friction coefficient, the value of the sliding friction coefficient is 1.0
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

    def set_object_size_value(self, value, automatic=False, mean=[0.025, 0.02], var=[0.000000002, 0.00000001]):
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

    def set_radi_danger_region_value(self, value, automatic=False, mean=0.05, var=0.000002):
        if not automatic:
            target_value = value
        else:
            if self.with_var == True:
                assert var is not None
                target_value = np.random.normal(loc=mean, scale=np.sqrt(var))
            else:
                target_value = mean
        target_value_list = [target_value, target_value, 0.001]
        for order in range(1):
            self.sim.model.site_size[order] = target_value_list

    def set_danger_offset_value(self, value, automatic=False, mean=0.1, var=0.005):
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

    def _get_obs(self):

        obs_to_return = super()._get_obs()

        if self.augment_hazard_info_in_state:

            hazard_regions = []
            for danger_region_index, danger_region_value in self.danger_regions.items():
                hazard_regions.append(danger_region_value)
            hazard_regions = np.concatenate(hazard_regions)
            obs_to_return['observation'] = np.concatenate([obs_to_return['observation'], hazard_regions])

        return obs_to_return


@saute_env
class SauteRandomizableFetchSlideSimple(RandomizeSafeFetchSlideCostSimpleEnv):
    "saute env"


if __name__ == "__main__":
    # test the saute env
    test_env = SauteRandomizableFetchSlideSimple(augment_hazard_info_in_state=True)
    s = test_env.reset()
    for i in range(100):
        s, r, d, info = test_env.step(test_env.action_space.sample())
