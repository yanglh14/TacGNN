import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch
import matplotlib.pyplot as plt
import pickle

class AllegroHandNotask():

    def __init__(self,num_envs = 8,if_viewer =True):

        self.gym = gymapi.acquire_gym()
        self.device = 'cuda:0'
        self.num_envs = num_envs
        self.create_sim()
        self.if_viewer = if_viewer
        if self.if_viewer:
            cam_props = gymapi.CameraProperties()
            self.viewer = self.gym.create_viewer(self.sim, cam_props)

            cam_pos = gymapi.Vec3(0.3, 0, 1)
            cam_target = gymapi.Vec3(0, 0, 0.7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state_target = self.dof_state.clone()

        self.sensor_handles = self.get_sensor_handles()

    def create_sim(self):

        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 1 / 50
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # sim_params.use_gpu_pipeline = True

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1000.0
        sim_params.physx.default_buffer_size_multiplier = 10.0

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, 0.75, 8)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.root_state_tensor_init = self.root_state_tensor.clone()

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state_init = self.dof_state.clone()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        shadow_hand_asset_file = "tactile/allegro_hand/allegro_hand.xml"
        object_asset_file = "tactile/objects/cube_big.urdf"

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

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
            shadow_hand_dof_props['effort'][i] = 0.01
            shadow_hand_dof_props['stiffness'][i] = 3
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001

        # index = [0,4,8]
        # shadow_hand_dof_props['lower'][index] = 0
        # shadow_hand_dof_props['upper'][index] = 0

        index = [1,5,9]
        shadow_hand_dof_props['upper'][index] = 0.78

        index = [2,6,10]
        shadow_hand_dof_props['upper'][index] = 1.2

        index = [3,7,11]
        shadow_hand_dof_props['upper'][index] = 1.2


        index = [12,13]
        shadow_hand_dof_props['upper'][index] = 1.0

        index = [14,15]
        shadow_hand_dof_props['upper'][index] = 1.4

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, 2))
        # shadow_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/12)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.fix_base_link = False

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        pose_dx, pose_dy, pose_dz = 0.0, -0.05, 0.03

        object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        self.object_pose_init = torch.tensor([object_start_pose.p.x,object_start_pose.p.y,object_start_pose.p.z])
        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 1, 0)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # add finger1
            finger1_actor = self.gym.create_actor(env_ptr, finger1_asset, shadow_hand_start_pose, "finger1", i, 1, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger1_actor, shadow_hand_dof_props[:4])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger1_actor)


            # add finger2
            finger2_actor = self.gym.create_actor(env_ptr, finger2_asset, shadow_hand_start_pose, "finger2", i, 1, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger2_actor, shadow_hand_dof_props[4:8])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger2_actor)

            # add finger3
            finger3_actor = self.gym.create_actor(env_ptr, finger3_asset, shadow_hand_start_pose, "finger3", i, 1, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger3_actor, shadow_hand_dof_props[8:12])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger3_actor)

            # add finger4
            finger4_actor = self.gym.create_actor(env_ptr, finger4_asset, shadow_hand_start_pose, "finger4", i, 1, 0)

            self.gym.set_actor_dof_properties(env_ptr, finger4_actor, shadow_hand_dof_props[12:16])

            self.gym.enable_actor_dof_force_sensors(env_ptr, finger4_actor)


            hand_idx = self.gym.get_actor_index(env_ptr, finger1_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
        self.hand_indices = torch.tensor(self.hand_indices,dtype=torch.int32)

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

    # def step(self):
    #     # step the physics
    #     self.gym.simulate(self.sim)
    #     self.gym.fetch_results(self.sim, True)
    #
    #     # update the viewer
    #     if self.if_viewer:
    #         self.gym.step_graphics(self.sim);
    #         self.gym.draw_viewer(self.viewer, self.sim, True)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #
    #
    #     return self.rigid_body_states[0,self.sensor_handles,:3]

    def sim2real(self,act):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.act = torch.tensor(act,dtype=torch.float32)
        self.act = self.act.repeat(self.num_envs)
        self.dof_state_target[:,0] = self.act

        hand_indices = torch.tensor([1,2,3,4], dtype=torch.int32)
        # self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state_target),
        #                                         gymtorch.unwrap_tensor(hand_indices), len(hand_indices))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.act))

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # update the viewer
        if self.if_viewer:
            self.gym.step_graphics(self.sim);
            self.gym.draw_viewer(self.viewer, self.sim, True)

        # self.tactile_show()
        # self.record_data = self.record()
        return None
    def record(self):

        self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(self._net_cf).view(self.num_envs, -1, 3)

        self.sensors_handles = self.get_sensor_handles()
        self.touch_tensor = self.net_cf[:, self.sensors_handles, 2]
        self.touch_tensor = self.touch_tensor.abs()
        self.touch_tensor[self.touch_tensor < 0.0005] = 0
        self.tactile_pose = self.rigid_body_states[:, self.sensors_handles, :3]

        self.tensor = self.touch_tensor.cpu()[:]

        self.object_pos = self.root_state_tensor[(self.hand_indices + 4).tolist(), :7]

        return self.tensor,self.tactile_pose,self.object_pos
    def tactile_show(self):


        self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(self._net_cf).view(self.num_envs, -1, 3)

        self.sensors_handles = self.get_sensor_handles()
        self.touch_tensor = self.net_cf[:, self.sensors_handles, 2]
        self.touch_tensor = self.touch_tensor.abs()
        self.touch_tensor[self.touch_tensor < 0.0005] = 0
        # self.touch_tensor[self.progress_buf == 1, :] = 0
        self.tactile_pose = self.rigid_body_states[:, self.sensors_handles, :3]

        self.tensor = self.touch_tensor.cpu()[0]
        # self.tensor[self.tensor > 1] = 0

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.tactile_pose.cpu()[0, :, 0], self.tactile_pose.cpu()[0, :, 1], self.tactile_pose.cpu()[0, :, 2],
                   s=0.2)
        ax.scatter(self.tactile_pose.cpu()[0, :, 0], self.tactile_pose.cpu()[0, :, 1], self.tactile_pose.cpu()[0, :, 2],
                   s=self.tensor * 1000, c='r')

        print(self.tensor[self.tensor > 0])
        plt.show()

    def reset(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


        self.dof_state[:,:] = self.dof_state_init

        hand_indices = torch.cat((self.hand_indices, self.hand_indices + 1, self.hand_indices + 2, self.hand_indices+3))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(hand_indices))
        object_indices = self.hand_indices + 4
        self.root_state_tensor[object_indices.tolist(), :] = self.random_pose()

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def random_pose(self):
        pose = torch.zeros([self.num_envs,13])
        pose[:,1] = -0.03
        pose[:,2] = 0.55
        pose[:,6] = 1
        for i in range(self.num_envs):
            rand_z = torch.rand(1)*np.pi
            rand_quat = quat_from_angle_axis(rand_z, torch.tensor([0, 0, 1], dtype=torch.float))
            pose[i,3:7] = rand_quat
            pose[i,:2] = pose[i,:2] + (torch.rand(2)-0.5)*2 * 0.05
        return pose

if __name__ == '__main__':
    hand = AllegroHandNotask(num_envs = 2,if_viewer=True)
    tactile_log,tactile_pos_log,object_pos_log = [],[],[]
    for epoch in range(10000):
        hand.reset()
        act = np.array([0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0, ], dtype=np.float32)

        for step in range(50):
            if step < 10:
                index = [1,5,9]
                act[index] = 0.1
                index = [13]
                act[index] = 0.1

            if step >10:

                index = [2,6,10]
                act[index] = 0.1
            if step >20:

                index = [14, 15]
                act[index] = 0.05

                index = [3,7,11]
                act[index] = 0.05
            hand.sim2real(act)

        tactile, tactile_pose, object_pos = hand.record()
        # x, y, z = get_euler_xyz(torch.tensor(object_pos[:,3:7]))

        print(object_pos[:,3:7])
        hand.tactile_show()

        tactile_log.append(tactile[:, :].tolist())
        tactile_pos_log.append(tactile_pose[:, :].tolist())
        object_pos_log.append(object_pos[:, :].tolist())

        # if epoch % 100 == 0 and epoch != 0:
        #     data = {'tactile': np.array(tactile_log),
        #             'tac_pose': np.array(tactile_pos_log),
        #             'object_pos': np.array(object_pos_log),
        #             }
        #     print('saving epoch num%d' % epoch)
        #     with open('../runs_tac_cube2/dataset_%d' % epoch + '.pkl', 'wb') as f:
        #         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        #     tactile_log, tactile_pos_log, object_pos_log = [], [], []
