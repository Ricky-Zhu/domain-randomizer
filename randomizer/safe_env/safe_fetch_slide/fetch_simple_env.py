import numpy as np

from gym.envs.robotics import rotations, utils
from randomizer.safe_env.safe_fetch_slide import robot_simple_env
import xml.etree.ElementTree as et


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchSimpleEnv(robot_simple_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, danger_regions_num, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self._keep_dist = 0.1
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.danger_region_num = danger_regions_num

        super(FetchSimpleEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    @property
    def keep_dist(self):
        return self._keep_dist

    def _get_dangers_pos(self):
        return self.danger_regions

    def _get_dangers_size(self):
        return self.sim.model.site_size[0][0]

    def _get_obj_size(self):
        return self.sim.model.geom_size[-1][0]


    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 165.
        self.viewer.cam.elevation = -40.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

        height = self.goal[2]

        # visualize the danger regions
        for i in range(self.danger_region_num):
            temp_name = 'danger{}'.format(i)
            site_danger_id = self.sim.model.site_name2id(temp_name)
            danger_pos = self.danger_regions[temp_name]
            danger_pos = np.append(danger_pos, height)
            self.sim.model.site_pos[site_danger_id] = danger_pos - sites_offset[0]

        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.05:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            x_min, x_max, y_min, y_max = self.danger_region_sample_range
            goal_xyz = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal_xyz = goal_xyz + self.target_offset
            goal_xyz[2] = self.height_offset
            while True:
                if np.linalg.norm(object_xpos - goal_xyz[:2]) >= 0.25:
                    break
                goal_xyz = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                                  size=3)
                goal_xyz = goal_xyz + self.target_offset
                goal_xyz[2] = self.height_offset

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

            self.goal_in_fetch = goal_xyz
            if self.target_in_the_air and self.np_random.uniform() < 0.3:
                self.goal_in_fetch[2] += self.np_random.uniform(0, 0.45)

        self.sim.forward()
        return True

    def _sample_goal(self):
        return self.goal_in_fetch.copy()


    def _sample_goal_init(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()


    def _sample_danger_regions_init(self):
        x_min, x_max, y_min, y_max = self.danger_region_sample_range

        def valid_layout(xy, danger_regions):
            valid = True
            for other_region, other_region_pos in danger_regions.items():
                dist = np.linalg.norm(xy - other_region_pos)
                if dist < self.keep_dist:
                    valid = False
                    return valid
            return valid

        danger_regions = {}
        danger_regions['goal'] = self.goal.copy()[:2]
        danger_regions['object'] = self.sim.data.get_joint_qpos('object0:joint')[:2]
        for i in range(self.danger_region_num):
            while True:
                xy = np.random.uniform([x_min, y_min], [x_max, y_max])
                if valid_layout(xy, danger_regions):
                    danger_regions['danger{}'.format(i)] = xy
                    break


        danger_regions.pop('goal')
        danger_regions.pop('object')

        return danger_regions

    def _sample_danger_regions(self):

        danger_regions = {}
        # ratio = np.random.uniform(0, 1)
        ratio = 0.5
        goal_xy_for_dangers = self.goal_in_fetch[:2]
        object_xy_for_dangers = self.sim.data.get_joint_qpos('object0:joint')[:2]
        xy = ratio * goal_xy_for_dangers + (1 - ratio) * object_xy_for_dangers
        time_steps_danger = 0
        while True:
            if ((np.linalg.norm(goal_xy_for_dangers - xy) >= self.keep_dist) and (np.linalg.norm(object_xy_for_dangers - xy) >= self.keep_dist)) or (time_steps_danger > 50000):
                danger_regions['danger{}'.format(0)] = xy
                break
            ratio = np.random.uniform(0, 1)
            xy = ratio * goal_xy_for_dangers + (1 - ratio) * object_xy_for_dangers
            time_steps_danger += 1
        return danger_regions

    @property
    def danger_region_sample_range(self):
        x_min = 1.0
        x_max = 1.8
        y_min = 0.37
        y_max = 1.07
        return x_min, x_max, y_min, y_max

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchSimpleEnv, self).render(mode, width, height)
