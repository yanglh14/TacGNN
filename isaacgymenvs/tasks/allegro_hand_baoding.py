import numpy as np
import os
import torch
import pickle
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from .base.vec_task import VecTask


class AllegroHandBaoding(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen","baoding"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "baoding": "tactile/objects/ball1.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])
            self.asset_files_dict["baoding"] = self.cfg["env"]["asset"].get("assetFileNameBall", self.asset_files_dict["baoding"])

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        # if obs contains touch sensors
        self.obs_touch = self.cfg["env"]["observationTouch"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 707 if self.obs_touch else 54
        }

        self.up_axis = 'z'
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 54 if self.obs_touch else 54

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 16

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.3, 0, 1)
            cam_target = gymapi.Vec3(0, 0, 0.7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

            self.gym.refresh_net_contact_force_tensor(self.sim)
            self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.net_cf = gymtorch.wrap_tensor(self._net_cf).view(self.num_envs,-1,3)
            self.sensors_handles = self.get_sensor_handles()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state_init = self.dof_state.clone()
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        if self.object_type == 'baoding':
            self.create_goal()

        self.shadow_hand_init_dof_pos = scale(torch.zeros([self.num_envs, self.num_shadow_hand_dofs], dtype=torch.float, device=self.device),
              self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
              self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        self.dof_state_init.view(self.num_envs,-1,2)[:,:,0] = self.shadow_hand_init_dof_pos
        self.shadow_hand_default_dof_pos = self.shadow_hand_init_dof_pos[0,:]

        self.object_angle = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float)
        self.object_angle_pre = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float)

        self.dataset = False
        if self.dataset:
            self.targets_log, self.actions_log, self.joints_log, self.tactile_log, self.tactile_pos_log, self.object_pos_log, self.object_noise_log, self.obs_log, self.object_pos_pre_log = [], [], [], [], [], [], [],[],[]
            self.log = {}
        self.log_ = False
        if self.log_:
            self.log = {}
            self.targets_log, self.actions_log, self.joints_log, self.tactile_log, self.tactile_pos_log, self.object_pos_log, self.object_pre_log, self.obs_log = [], [], [], [], [], [], [],[]
        self.step_num = 0
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        shadow_hand_asset_file = "tactile/allegro_hand/allegro_hand.xml"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        # load finger assets
        finger1_asset_file = "tactile/allegro_hand/allegro_finger1.xml"
        finger1_asset = self.gym.load_asset(self.sim, asset_root, finger1_asset_file, asset_options)
        finger2_asset_file = "tactile/allegro_hand/allegro_finger2.xml"
        finger2_asset = self.gym.load_asset(self.sim, asset_root, finger2_asset_file, asset_options)
        finger3_asset_file = "tactile/allegro_hand/allegro_finger3.xml"
        finger3_asset = self.gym.load_asset(self.sim, asset_root, finger3_asset_file, asset_options)
        finger4_asset_file = "tactile/allegro_hand/allegro_finger4.xml"
        finger4_asset = self.gym.load_asset(self.sim, asset_root, finger4_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset) + self.gym.get_asset_rigid_body_count(finger1_asset) + self.gym.get_asset_rigid_body_count(finger2_asset) + self.gym.get_asset_rigid_body_count(finger3_asset) + self.gym.get_asset_rigid_body_count(finger4_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset) + self.gym.get_asset_rigid_shape_count(finger1_asset) + self.gym.get_asset_rigid_shape_count(finger2_asset) + self.gym.get_asset_rigid_shape_count(finger3_asset) + self.gym.get_asset_rigid_shape_count(finger4_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset) + self.gym.get_asset_dof_count(finger1_asset) + self.gym.get_asset_dof_count(finger2_asset) + self.gym.get_asset_dof_count(finger3_asset) + self.gym.get_asset_dof_count(finger4_asset)
        print("Num dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs #self.gym.get_asset_actuator_count(shadow_hand_asset)

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # set shadow_hand dof properties
        shadow_hand_dof_props = np.concatenate((self.gym.get_asset_dof_properties(finger1_asset),self.gym.get_asset_dof_properties(finger2_asset),self.gym.get_asset_dof_properties(finger3_asset),self.gym.get_asset_dof_properties(finger4_asset)))

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            print("Max effort: ", shadow_hand_dof_props['effort'][i])
            shadow_hand_dof_props['effort'][i] = 0.5
            shadow_hand_dof_props['stiffness'][i] = 3
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001
        data = np.load('scripts/joint_sim2real.npy')
        shadow_hand_dof_props['stiffness'] = data[0]
        shadow_hand_dof_props['damping'] = data[1]

        self.shadow_hand_dof_props = shadow_hand_dof_props
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        if self.object_type == "baoding":
            object_asset_2 = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/12)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        pose_dx, pose_dy, pose_dz = 0.028, -0.01, 0.05

        object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "baoding":
            object_start_pose_2 = gymapi.Transform()
            object_start_pose_2.p = gymapi.Vec3()
            pose_dx, pose_dy, pose_dz = -0.028, -0.01, 0.05

            object_start_pose_2.p.x = shadow_hand_start_pose.p.x + pose_dx
            object_start_pose_2.p.y = shadow_hand_start_pose.p.y + pose_dy
            object_start_pose_2.p.z = shadow_hand_start_pose.p.z + pose_dz

        self.goal_displacement = gymapi.Vec3(0, -0.01, -0.1)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        if self.object_type == "baoding":
            goal_start_pose_2 = gymapi.Transform()
            goal_start_pose_2.p = object_start_pose_2.p + self.goal_displacement

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 8
        max_agg_shapes = self.num_shadow_hand_shapes + 8

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        shadow_hand_rb_count = self.num_shadow_hand_bodies
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # add finger1
            finger1_actor = self.gym.create_actor(env_ptr, finger1_asset, shadow_hand_start_pose, "finger1", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger1_actor, shadow_hand_dof_props[:4])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger1_actor)


            # add finger2
            finger2_actor = self.gym.create_actor(env_ptr, finger2_asset, shadow_hand_start_pose, "finger2", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger2_actor, shadow_hand_dof_props[4:8])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger2_actor)

            # add finger3
            finger3_actor = self.gym.create_actor(env_ptr, finger3_asset, shadow_hand_start_pose, "finger3", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger3_actor, shadow_hand_dof_props[8:12])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger3_actor)

            # add finger4
            finger4_actor = self.gym.create_actor(env_ptr, finger4_asset, shadow_hand_start_pose, "finger4", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger4_actor, shadow_hand_dof_props[12:16])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger4_actor)


            hand_idx = self.gym.get_actor_index(env_ptr, finger1_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)

            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add object2
            object_handle_2 = self.gym.create_actor(env_ptr, object_asset, object_start_pose_2, "object2", i, 0, 0)

            object_idx = self.gym.get_actor_index(env_ptr, object_handle_2, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0]+[object_start_pose_2.p.x, object_start_pose_2.p.y, object_start_pose_2.p.z,
                                           object_start_pose_2.r.x, object_start_pose_2.r.y, object_start_pose_2.r.z, object_start_pose_2.r.w,
                                           0, 0, 0, 0, 0, 0])

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # add goal object2
            goal_handle_2 = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose_2, "goal_object2", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle_2, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type == "baoding":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle_2, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.6, 0.6))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle_2, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.6, 0.6))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 26)
        self.goal_states = self.object_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.object_angle,self.object_angle_pre
        )
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()
        # print(self.rew_buf[0],dist_rew[0])

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

        self.object_pos = self.root_state_tensor[self.object_indices, 0:3].view(int(self.object_indices.shape[0]/2), 6)
        self.obs_noise_range = 0.0
        self.noise = torch.randn(self.num_envs,6,device=self.device)*self.obs_noise_range

        self.object_1 = self.object_pos[:,:2]
        self.object_2 = self.object_pos[:,3:5]
        self.object_vector = self.object_1 - self.object_2
        self.object_angle = torch.arccos(self.object_vector[:,0]/torch.linalg.norm(self.object_vector,dim=1)) * (180/torch.pi)
        self.object_angle[self.object_vector[:,1]<0] *= -1
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10].view(int(self.object_indices.shape[0]/2), 6)
        self.object_rot, self.goal_rot = torch.tensor([0]), torch.tensor([0])
        self.goal_pos = torch.cat((self.goal_states[:, 0:3],self.goal_states[:, 13:13+3]),1)

        if self.obs_type == "full_state":
             self.compute_full_state()
        else:
            print("Unkown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            pass
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            obj_obs_start = 2*self.num_shadow_hand_dofs  # 32
            self.object_noise = self.object_pos + self.noise

            self.obs_buf[:, obj_obs_start:obj_obs_start + 6] = self.object_noise

            goal_obs_start = obj_obs_start + 6  # 38

            touch_sensor_obs_start = goal_obs_start  # 38

            if self.obs_touch:
                touch_tensor = self.net_cf[:, self.sensors_handles, 2]
                touch_tensor = touch_tensor.abs()
                touch_tensor[touch_tensor<0.0005] = 0
                touch_tensor[self.progress_buf == 1, :] = 0

                self.obs_buf[:, touch_sensor_obs_start:touch_sensor_obs_start + 653] = self.force_torque_obs_scale * touch_tensor
                obs_end = touch_sensor_obs_start+653  #691
                # obs_total = obs_end + num_actions = 691 + 16 = 707

                self.obs_buf[:, obj_obs_start:obj_obs_start + 6] = self.object_pos *0
            else:

                obs_end = touch_sensor_obs_start  #38
                # obs_total = obs_end + num_actions = 38 + 16 = 54

            self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions

    def reset_target_pose(self, env_ids, apply_reset=False):

        self.goal_states[env_ids, 0:26] = self.goal_init_state[env_ids, 0:26]
        self.root_state_tensor[self.goal_object_indices[env_ids*2], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids*2], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids*2], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids*2], 7:13])
        self.root_state_tensor[self.goal_object_indices[env_ids*2+1], 0:3] = self.goal_states[env_ids, 13:13+3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids*2+1], 3:7] = self.goal_states[env_ids, 13+3:13+7]
        self.root_state_tensor[self.goal_object_indices[env_ids*2+1], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids*2+1], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids*2].to(torch.int32)
            goal_indices = torch.cat((goal_object_indices, goal_object_indices + 1))
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_indices), len(env_ids)*2)

            # generate random values
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5),
                                           device=self.device)

            # reset object
            self.root_state_tensor[self.object_indices[env_ids * 2]] = self.object_init_state[env_ids, :13].clone()

            self.root_state_tensor[self.object_indices[env_ids * 2], :7] = self.object_init_state[env_ids, :7] + \
                                                                           self.reset_position_noise * rand_floats[:,
                                                                                                       :7]

            self.root_state_tensor[self.object_indices[env_ids * 2 + 1]] = self.object_init_state[env_ids,
                                                                           13:26].clone()

            self.root_state_tensor[self.object_indices[env_ids * 2 + 1], :7] = self.object_init_state[env_ids,
                                                                               13:13 + 7] + \
                                                                               self.reset_position_noise * rand_floats[
                                                                                                           :, 13:13 + 7]

            object_indices = torch.unique(
                torch.cat([self.object_indices[env_ids * 2], self.object_indices[env_ids * 2 + 1],
                           self.goal_object_indices[env_ids * 2], self.goal_object_indices[env_ids * 2 + 1]]).to(torch.int32))
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(object_indices), len(object_indices))

            # reset shadow hand
            delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
            delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
            rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

            pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
            self.shadow_hand_dof_pos[env_ids, :] = pos
            self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
            self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
            self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

            hand_indices = self.hand_indices[env_ids].to(torch.int32)
            hand_indices = torch.cat((hand_indices,hand_indices+1,hand_indices+2,hand_indices+3))
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.prev_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids)*4)

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state_init),
                                                  gymtorch.unwrap_tensor(hand_indices), len(env_ids)*4)

        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids*2]] = self.object_init_state[env_ids,:13].clone()

        self.root_state_tensor[self.object_indices[env_ids*2], :2] = self.object_init_state[env_ids, :2] + \
            self.reset_position_noise * rand_floats[:, :2]


        self.root_state_tensor[self.object_indices[env_ids*2+1]] = self.object_init_state[env_ids,13:26].clone()

        self.root_state_tensor[self.object_indices[env_ids*2+1], :2] = self.object_init_state[env_ids,13:13+2] + \
            self.reset_position_noise * rand_floats[:, 13:13+2]


        object_indices = torch.unique(torch.cat([self.object_indices[env_ids*2],self.object_indices[env_ids*2+1],
                                                 self.goal_object_indices[env_ids*2],self.goal_object_indices[env_ids*2+1],
                                                 self.goal_object_indices[goal_env_ids*2],self.goal_object_indices[goal_env_ids*2+1]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta * 0
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        hand_indices = torch.cat((hand_indices,hand_indices+1,hand_indices+2,hand_indices+3))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids)*4)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state_init),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids)*4)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        # progress goal position
        if self.step_num == 0:
            self.object_pos_pre_init = self.root_state_tensor[self.object_indices, 0:3].view(int(self.object_indices.shape[0]/2), 6).clone()
        if self.object_type == 'baoding':
            for env_index in range(self.num_envs):
                self.goal_states[env_index, 0:3] = self.goal[env_index,self.progress_buf[env_index],0:3]
                self.goal_states[env_index, 13:16] = self.goal[env_index,self.progress_buf[env_index],3:6]

            env_ids = np.array(range(self.num_envs))
            self.root_state_tensor[self.goal_object_indices[env_ids*2], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
            self.root_state_tensor[self.goal_object_indices[env_ids*2+1], 0:3] = self.goal_states[env_ids, 13:13+3] + self.goal_displacement_tensor

            goal_object_indices = self.goal_object_indices[env_ids*2].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))

            goal_object_indices = self.goal_object_indices[env_ids*2+1].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        self.object_angle_pre = self.object_angle.clone()
        self.object_angle_pre[self.reset_buf ==1] = 0
        self.step_num += 1

        if self.dataset:

            self.touch_tensor = self.net_cf[:, self.sensors_handles, 2]
            self.touch_tensor = self.touch_tensor.abs()
            self.touch_tensor[self.touch_tensor < 0.0005] = 0
            self.touch_tensor[self.progress_buf == 1, :] = 0

            self.tactile_pose = self.rigid_body_states[:, self.sensors_handles, :3]
            self.object_pos_pre = self.object_pos.clone()
            self.object_pos_pre[self.progress_buf==1,:] = self.object_pos_pre_init[self.progress_buf==1,:]

            if (self.progress_buf != 1).any():
                self.object_pos_pre[self.progress_buf != 1, :] = torch.tensor(self.object_pos_pre_clone,dtype=torch.float, device=self.device)[self.progress_buf != 1, :]

            self.object_pos_pre_log.append(self.object_pos_pre.tolist())
            self.tactile_log.append(self.touch_tensor[:, :].tolist())
            self.tactile_pos_log.append(self.tactile_pose[:, :].tolist())
            self.object_pos_log.append(self.object_pos[:, :].tolist())
            self.object_noise_log.append((self.object_noise[:, :]).tolist())
            self.object_pos_pre_clone = self.object_pos_log[-1].copy()
            if self.step_num%100 ==0:

                # self.log['tactile_log'] = np.array(self.tactile_log)
                # self.log['tactile_pos_log'] = np.array(self.tactile_pos_log)
                # self.log['object_pos_log'] = np.array(self.object_pos_log)
                # self.log['object_noise_log'] = np.array(self.object_noise_log)
                # np.save('runs_tac/dataset_%d'%self.step_num, self.log)

                data = {'tactile': np.array(self.tactile_log),
                        'tac_pose': np.array(self.tactile_pos_log),
                        'object_pos': np.array(self.object_pos_log),
                        'object_pos_pre':np.array(self.object_pos_pre_log)
                        }
                print('saving step num%d'%self.step_num)
                with open('runs_tac2/dataset_%d'%self.step_num + '.pkl', 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

                self.targets_log, self.actions_log, self.joints_log, self.tactile_log, self.tactile_pos_log, self.object_pos_log, self.object_noise_log, self.obs_log, self.object_pos_pre_log = [], [], [], [], [], [], [], [], []
            if self.step_num == 20000:
                print('test done')
        if self.log_:
            self.touch_tensor = self.net_cf[:, self.sensors_handles, 2]
            self.touch_tensor = self.touch_tensor.abs()
            self.touch_tensor[self.touch_tensor < 0.0005] = 0
            self.touch_tensor[self.progress_buf == 1, :] = 0

            self.tactile_pose = self.rigid_body_states[:, self.sensors_handles, :3]
            self.actions_log.append(scale(self.actions,self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])[0].tolist())
            self.targets_log.append(self.cur_targets[0].tolist())
            self.joints_log.append(self.shadow_hand_dof_pos[0, :].tolist())
            self.tactile_log.append(self.touch_tensor[0, :].tolist())
            self.tactile_pos_log.append(self.tactile_pose[0, :].tolist())
            self.object_pos_log.append(self.object_pos[0, :].tolist())
            self.object_pre_log.append((self.object_noise[0, :]*100).tolist())
            self.obs_log.append(self.obs_buf[0, :].tolist())

            if self.reset_buf[0] == 1 :

                self.log['actions_log'] = self.actions_log
                self.log['targets_log'] = self.targets_log
                self.log['joints_log'] = self.joints_log
                self.log['tactile_log'] = self.tactile_log
                self.log['tactile_pos_log'] = self.tactile_pos_log
                self.log['object_pos_log'] = self.object_pos_log
                self.log['object_pre_log'] = self.object_pre_log
                self.log['obs_log'] = self.obs_log
                # np.save('runs/sim_log_2',self.log)
                self.log = {}
                self.targets_log, self.actions_log, self.joints_log, self.tactile_log, self.tactile_pos_log, self.object_pos_log, self.object_pre_log, self.obs_log = [], [], [], [], [], [], [], []

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def get_sensor_handles(self):

        sensors_handles = np.array([])
        for i in range(12):
            if i ==0:
                index_start = self.gym.find_actor_rigid_body_index(self.envs[0], 0, 'touch_111_1_1', gymapi.DOMAIN_SIM)
                index_end = self.gym.find_actor_rigid_body_index(self.envs[0], 0, 'touch_111_7_12', gymapi.DOMAIN_SIM)
                sensors_handles = np.concatenate((sensors_handles,np.array(range(index_start, index_end+1))))
            else:
                j = (i-1)//3 +1
                if (i-1)%3 == 0:
                    index_start = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_1_1'%(i-1), gymapi.DOMAIN_SIM)
                    index_end = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_6_12'%(i-1), gymapi.DOMAIN_SIM)
                    sensors_handles = np.concatenate((sensors_handles,np.array(range(index_start, index_end+1))))
                else:
                    index_start = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_1_1'%(i-1), gymapi.DOMAIN_SIM)
                    index_end = self.gym.find_actor_rigid_body_index(self.envs[0], j, 'touch_%d_6_6'%(i-1), gymapi.DOMAIN_SIM)
                    sensors_handles = np.concatenate((sensors_handles,np.array(range(index_start, index_end+1))))

        return sensors_handles

    def create_goal(self):

        # creat a goal for the object position.

        self.goal = torch.zeros((self.num_envs,self.max_episode_length+1,6), dtype=torch.float, device=self.device)
        self.center_pose = (self.object_init_state[:,:3] + self.object_init_state[:,13:13+3])/2
        self.y_radius = (self.object_init_state[:,0:1] - self.object_init_state[:,13:14])/2
        self.x_radius = self.y_radius
        for i in range(self.max_episode_length+1):
            if i <= 300:
                angle = i * np.pi / 300
            else:
                angle = 300 * np.pi / 300

            angle = angle
            goal_position = torch.cat([self.x_radius * np.cos(angle) + self.center_pose[:,0:1],
                             self.y_radius * np.sin(angle) + self.center_pose[:,1:2], self.center_pose[:,2:3],
                             -self.x_radius * np.cos(angle) + self.center_pose[:,0:1],
                             -self.y_radius * np.sin(angle) + self.center_pose[:,1:2], self.center_pose[:,2:3]],dim=1)

            self.goal[:,i,:] = goal_position
        return self.goal
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool, object_angle, object_angle_pre
):
    # Distance from the hand to the object
    angle_dist = object_angle - object_angle_pre
    # goal_dist = torch.norm(object_pos[:,:2] - target_pos[:,:2], p=2, dim=-1) + torch.norm(object_pos[:,3:5] - target_pos[:,3:5], p=2, dim=-1)
    fall_reset = (((object_pos[:,2]-0.5) <0) + ((object_pos[:,5]-0.5) <0) + ((object_pos[:, 2] - 0.55) > 0) + ((object_pos[:, 5] - 0.55) > 0)) > 0
    center_dist = torch.norm(object_pos[:,:1] + object_pos[:,3:4],p=2,dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    dist_rew = angle_dist * 0.5

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count

    goal_resets_index = object_angle > 160
    goal_resets = torch.where(goal_resets_index, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(fall_reset == 1, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(fall_reset == 1, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        # progress_buf = torch.where(torch.abs(goal_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
