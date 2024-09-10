from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np
import matplotlib.pyplot as plt
import torch

gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 50
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.0
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.max_depenetration_velocity = 1000.0
sim_params.physx.default_buffer_size_multiplier = 10.0

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

gym.prepare_sim(sim)
# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

#### load desk asset
asset_root = "../../assets"
asset_file = "tactile/CoRL2022/corl2022.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01

asset1 = gym.load_asset(sim, asset_root, asset_file, asset_options)

#### load object asset
if_viewer = True
object_name = 'cal_tac_indentor'
asset_root = "../../assets"
# asset_file = "tactile/objects/ycb/"+object_name+"/"+object_name+ ".urdf"
asset_file = "tactile/objects/"+ object_name +".urdf"

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.disable_gravity = True

asset2 = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

def run_sim(if_viewer):
    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        gym.begin_aggregate(env, 300, 1000, True)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.1, 0.1, 0.1)
        actor_handle0 = gym.create_actor(env, asset1, pose, "Desk", i, 0)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.1, 0.1, 0.125)
        actor_handle1 = gym.create_actor(env, asset2, pose, "Object", i, 0)

        gym.end_aggregate(env)

    if if_viewer:
        cam_props = gymapi.CameraProperties()
        viewer = gym.create_viewer(sim, cam_props)
        cam_pos = gymapi.Vec3(0.3, 0.3, 0.3)
        cam_target = gymapi.Vec3(0, 0, 0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    f,p = [],[]
    j = 0
    while not gym.query_viewer_has_closed(viewer):
        j += 1
    # for i in range(1000):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        if if_viewer:
            gym.step_graphics(sim);
            gym.draw_viewer(viewer, sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        _net_cf = gym.acquire_net_contact_force_tensor(sim)
        net_cf = gymtorch.wrap_tensor(_net_cf).view(1, -1, 3)
        tactile = net_cf[0,3:3+225,:3]

        rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
        rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
        tac_pose = rigid_body_states[0,3:3+225,:3]

        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        indentor_pose = root_state_tensor[-1,2]

        object_indices = torch.tensor([1],dtype=torch.int32)
        root_state_tensor[-1,2] = 0.13-j*0.00001
        gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(root_state_tensor),gymtorch.unwrap_tensor(object_indices), len(object_indices))
        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)


        contacts =gym.get_env_rigid_contacts(env)

        for i in range(contacts.shape[0]):
            if contacts[i]['body1'] == 115:
                p.append(contacts[i]['initialOverlap'])
                # p.append(root_state_tensor[-1,2]-0.12)
                f.append( np.linalg.norm(np.array(tactile[112,:3])) )

        gym.sync_frame_time(sim)

    plot_forceposition(p, f)

def plot_forceposition(p,f):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(-np.array(p)*1000,np.array(f)*10)

    # ax.legend()
    plt.show()

if __name__ == "__main__":
    run_sim(if_viewer)
