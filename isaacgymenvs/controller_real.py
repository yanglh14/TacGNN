#!/home/yang/anaconda3/envs/rlgpu/bin/python
import os
from isaacgymenvs.tasks.allegro_hand_notask import AllegroHandNotask
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from allegro_tactile_cal.msg import tactile_msgs

import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaacgym.torch_utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

class AllegroHand():
    def __init__(self,runner):
        self.runner = runner

        self.player = self.runner.sim2real()

        self.env = self.player.env
        self.max_steps = 200
        self.is_determenistic = True
        self.num_obs = 54
        self.device = 'cuda:0'
        self.num_shadow_hand_dofs = 16
        self.num_actions = 16
        self.num_dofs = 16

        self.obs_buf = torch.zeros(
            (1, self.num_obs), device=self.device, dtype=torch.float)
        self.tac_buf = np.zeros(653)
        self.tac = np.zeros(653)
        self.joint_init = torch.zeros(
            (1, 16), device=self.device, dtype=torch.float)

        self.finger_jnt_pos = torch.zeros(
            (1, 16), device=self.device, dtype=torch.float)
        self.finger_jnt_vel = torch.zeros(
            (1, 16), device=self.device, dtype=torch.float)
        self.finger_jnt_force = torch.zeros(
            (1, 16), device=self.device, dtype=torch.float)
        self.actions = torch.zeros(
            (1, 16), device=self.device, dtype=torch.float)

        self.prev_targets = torch.zeros((1, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((1, self.num_dofs), dtype=torch.float, device=self.device)

        self.cur_targets[:, self.env.actuated_dof_indices] = scale(self.actions, self.env.shadow_hand_dof_lower_limits[
            self.env.actuated_dof_indices], self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])
        self.prev_targets[:, self.env.actuated_dof_indices] = scale(self.actions, self.env.shadow_hand_dof_lower_limits[
            self.env.actuated_dof_indices], self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])

        self.cal_table = np.load('scripts/calibration_table.npy')

        self.pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        self.sub_states = rospy.Subscriber("/allegroHand_0/joint_states", JointState,self.joint_state_callback)
        self.sub_tactile = rospy.Subscriber('allegro_tactile', tactile_msgs, self.allegro_tactile_callback)

        rospy.init_node('allegro_hand', anonymous=True)
        self.rate = rospy.Rate(50)

        self.log_ = True
        if self.log_:
            self.log = {}
            self.time_log,self.targets_log, self.actions_log, self.joints_log, self.tactile_log, self.tactile_pos_log, self.object_pos_log, self.object_pre_log, self.obs_log = [],[],[],[],[],[],[],[],[]
    def player_run(self):

        self.step_num = 0

        obses = self.player.env_reset(self.env)

        self.init = True
        self.step()
        self.initialize()
        self.step()
        self.init = False
        print("Start playing!")
        input()

        data_2 = np.load('runs/sim_log.npy', allow_pickle=True)
        data_2 = data_2.item()
        actions_list = data_2['actions_log']
        for n in range(actions_list.__len__()):

            self.get_obs()
            self.actions = self.player.get_action(self.obs_buf, self.is_determenistic)
            # self.actions = torch.as_tensor([[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]],device=self.device, dtype=torch.float)
            self.actions[:] = torch.as_tensor(actions_list[n],device=self.device, dtype=torch.float)
            self.step()

        if self.log_:
            self.log['time_log'] = self.time_log

            self.log['actions_log'] = self.actions_log
            self.log['targets_log'] = self.targets_log
            self.log['joints_log'] = self.joints_log
            self.log['tactile_log'] = self.tactile_log
            self.log['tactile_pos_log'] = self.tactile_pos_log
            self.log['object_pre_log'] = self.object_pre_log
            self.log['tac_init'] = self.tac_init
            self.log['joint_init'] = self.joint_init.tolist()
            self.log['obs_log'] = self.obs_log


            np.save('runs/real_log_%f'%int(time.time()),self.log)

    def initialize(self):
        self.tac_init = []
        for i in range(5):
            self.tac_init.append(self.tac_buf)
            time.sleep(0.1)
        self.tac_init = np.average(np.array(self.tac_init),axis=0)

        self.joint_list = torch.zeros((5, 16), device=self.device, dtype=torch.float)

        for i in range(5):
            self.joint_list[i,:] = self.finger_jnt_pos[0,:]
            time.sleep(0.1)
        self.joint_init_ = scale(torch.as_tensor([[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0]],device=self.device, dtype=torch.float),self.env.shadow_hand_dof_lower_limits[self.env.actuated_dof_indices],self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])
        self.joint_init = torch.mean(self.joint_list,dim=0,keepdim = True) - self.joint_init_

        print('initialize success!')

    def joint_state_callback(self, msg):
        self.finger_jnt_pos[0,:] = torch.tensor(np.asarray(msg.position),device=self.device, dtype=torch.float)
        self.finger_jnt_vel[0,:] = torch.tensor(np.asarray(msg.velocity),device=self.device, dtype=torch.float)

    def allegro_tactile_callback(self, msg):
        self.tac_buf[:113] = np.asarray(msg.palm_Value)

        self.tac_buf[113:113+72] = np.asarray(msg.index_tip_Value)
        self.tac_buf[113+72:113+72+36] = np.asarray(msg.index_mid_Value)
        self.tac_buf[113+72+36:113+144] = np.asarray(msg.index_end_Value)

        self.tac_buf[113+144:113+144+72] = np.asarray(msg.middle_tip_Value)
        self.tac_buf[113+144+72:113+144+108] = np.asarray(msg.middle_mid_Value)
        self.tac_buf[113+144+108:113+288] = np.asarray(msg.middle_end_Value)

        self.tac_buf[113+288:113+288+72] = np.asarray(msg.ring_tip_Value)
        self.tac_buf[113+288+72:113+288+108] = np.asarray(msg.ring_mid_Value)
        self.tac_buf[113+288+108:113+288+144] = np.asarray(msg.ring_end_Value)

        self.tac_buf[113+432:113+432+72] = np.asarray(msg.thumb_tip_Value)
        self.tac_buf[113+432+72:113+432+108] = np.asarray(msg.thumb_mid_Value)

    def get_obs(self):
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.finger_jnt_pos,
                                                               self.env.shadow_hand_dof_lower_limits,
                                                               self.env.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.env.vel_obs_scale * self.finger_jnt_vel/10000

        obj_obs_start = 2 * self.num_shadow_hand_dofs  # 32
        self.obs_buf[:, obj_obs_start:obj_obs_start + 6] = self.object_pre()

        goal_obs_start = obj_obs_start + 6  # 38

        obs_end = goal_obs_start  # 38
        # obs_total = obs_end + num_actions = 38 + 16 = 54

        self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions


    def step(self):

        self.step_num += 1

        # if self.init:
        #     self.cur_targets[:, self.env.actuated_dof_indices] = scale(self.actions,self.env.shadow_hand_dof_lower_limits[self.env.actuated_dof_indices],self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])
        # else:
        #     self.cur_targets[:, self.env.actuated_dof_indices] = self.actions
        # self.cur_targets[:, self.env.actuated_dof_indices] = scale(self.actions,self.env.shadow_hand_dof_lower_limits[self.env.actuated_dof_indices],self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])
        self.cur_targets[:, self.env.actuated_dof_indices] = self.actions

        self.cur_targets[:, self.env.actuated_dof_indices] = self.cur_targets[:, self.env.actuated_dof_indices]- self.joint_init[:,:]
        self.cur_targets[:, self.env.actuated_dof_indices] = self.env.act_moving_average * self.cur_targets[:,self.env.actuated_dof_indices] + (1.0 - self.env.act_moving_average) * self.prev_targets[:,self.env.actuated_dof_indices]
        self.cur_targets[:, self.env.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.env.actuated_dof_indices],self.env.shadow_hand_dof_lower_limits[self.env.actuated_dof_indices],
                                                                      self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])
        self.prev_targets[:, self.env.actuated_dof_indices] = self.cur_targets[:, self.env.actuated_dof_indices]

        joint_command = JointState()
        joint_command.header = Header()
        joint_command.header.stamp = rospy.Time.now()
        joint_command.name = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0','joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0','joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0','joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0']
        joint_command.position = self.cur_targets[0].tolist()
        joint_command.velocity = []
        joint_command.effort = []
        for i in range(1):
            if not self.init and self.log_:

                self.time_log.append(time.time())

                self.actions_log.append(torch.squeeze(scale(self.actions,self.env.shadow_hand_dof_lower_limits[self.env.actuated_dof_indices], self.env.shadow_hand_dof_upper_limits[self.env.actuated_dof_indices])).tolist())
                self.targets_log.append(self.cur_targets[0].tolist())
                self.joints_log.append(self.finger_jnt_pos[0, :].tolist())

                self.tactile_log.append(self.tac_record[:].tolist())
                self.tactile_pos_log.append(self.sensor_pos[0, :].tolist())
                self.object_pre_log.append(self.pos[0, :].tolist())
                self.obs_log.append(self.obs_buf[0,:].tolist())

            self.pub.publish(joint_command)
            self.rate.sleep()

    def object_pre(self):
        self.tac_record = self.tac_buf.copy()
        self.tac_buf_ = self.tac_buf - self.tac_init
        self.tac_buf_[self.tac_buf_<0] = 0
        self.tac_buf_[self.tac_buf_<self.cal_table[:,1]] = 0
        self.tac_buf_ *= self.cal_table[:, 0]

        self.tac[:15] = [self.tac_buf_[109],self.tac_buf_[105],self.tac_buf_[100],self.tac_buf_[95],self.tac_buf_[90],self.tac_buf_[85],self.tac_buf_[80],self.tac_buf_[75],self.tac_buf_[69],self.tac_buf_[59],self.tac_buf_[48],self.tac_buf_[36],self.tac_buf_[24],self.tac_buf_[12],self.tac_buf_[0]]
        self.tac[15:30] = [self.tac_buf_[109+1],self.tac_buf_[105+1],self.tac_buf_[100+1],self.tac_buf_[95+1],self.tac_buf_[90+1],self.tac_buf_[85+1],self.tac_buf_[80+1],self.tac_buf_[75+1],self.tac_buf_[69+1],self.tac_buf_[59+1],self.tac_buf_[48+1],self.tac_buf_[36+1],self.tac_buf_[24+1],self.tac_buf_[12+1],self.tac_buf_[0+1]]
        self.tac[30:45] = [self.tac_buf_[109+2],self.tac_buf_[105+2],self.tac_buf_[100+2],self.tac_buf_[95+2],self.tac_buf_[90+2],self.tac_buf_[85+2],self.tac_buf_[80+2],self.tac_buf_[75+2],self.tac_buf_[69+2],self.tac_buf_[59+2],self.tac_buf_[48+2],self.tac_buf_[36+2],self.tac_buf_[24+2],self.tac_buf_[12+2],self.tac_buf_[0+2]]
        self.tac[45:60] = [self.tac_buf_[109+3],self.tac_buf_[105+3],self.tac_buf_[100+3],self.tac_buf_[95+3],self.tac_buf_[90+3],self.tac_buf_[85+3],self.tac_buf_[80+3],self.tac_buf_[75+3],self.tac_buf_[69+3],self.tac_buf_[59+3],self.tac_buf_[48+3],self.tac_buf_[36+3],self.tac_buf_[24+3],self.tac_buf_[12+3],self.tac_buf_[0+3]]
        self.tac[60:73] = [self.tac_buf_[100+4],self.tac_buf_[95+4],self.tac_buf_[90+4],self.tac_buf_[85+4],self.tac_buf_[80+4],self.tac_buf_[75+4],self.tac_buf_[69+4],self.tac_buf_[59+4],self.tac_buf_[48+4],self.tac_buf_[36+4],self.tac_buf_[24+4],self.tac_buf_[12+4],self.tac_buf_[0+4]]
        self.tac[73:79] = [self.tac_buf_[59+5],self.tac_buf_[48+5],self.tac_buf_[36+5],self.tac_buf_[24+5],self.tac_buf_[12+5],self.tac_buf_[0+5]]
        self.tac[79:85] = [self.tac_buf_[59+6],self.tac_buf_[48+6],self.tac_buf_[36+6],self.tac_buf_[24+6],self.tac_buf_[12+6],self.tac_buf_[0+6]]
        self.tac[85:91] = [self.tac_buf_[59+7],self.tac_buf_[48+7],self.tac_buf_[36+7],self.tac_buf_[24+7],self.tac_buf_[12+7],self.tac_buf_[0+7]]
        self.tac[91:97] = [self.tac_buf_[59+8],self.tac_buf_[48+8],self.tac_buf_[36+8],self.tac_buf_[24+8],self.tac_buf_[12+8],self.tac_buf_[0+8]]
        self.tac[97:102] = [self.tac_buf_[48+9],self.tac_buf_[36+9],self.tac_buf_[24+9],self.tac_buf_[12+9],self.tac_buf_[0+9]]
        self.tac[102:106] = [self.tac_buf_[36+10],self.tac_buf_[24+10],self.tac_buf_[12+10],self.tac_buf_[0+10]]
        self.tac[106:113] = [self.tac_buf_[74],self.tac_buf_[68],self.tac_buf_[458],self.tac_buf_[36+11],self.tac_buf_[24+11],self.tac_buf_[12+11],self.tac_buf_[0+11]]

        self.tac[113:113+72] = self.tac_buf_[113:113+72]
        self.tac[113+72:113 + 72+36] = self.tac_buf_[113+72:113+72+36].reshape(6,6).transpose().reshape(36)
        self.tac[113+72+36:113+144] = self.tac_buf_[113+72+36:113+144].reshape(6,6).transpose().reshape(36)

        self.tac[113+144:113+144+72] = self.tac_buf_[113+144:113+144+72]
        self.tac[113+144+72:113+144+108] = self.tac_buf_[113+144+72:113+144+108].reshape(6,6).transpose().reshape(36)
        self.tac[113+144+108:113+288] = self.tac_buf_[113+144+108:113+288].reshape(6,6).transpose().reshape(36)

        self.tac[113+288:113+288+72] = self.tac_buf_[113+288:113+288+72]
        self.tac[113+288+72:113+288+108] = self.tac_buf_[113+288+72:113+288+108].reshape(6,6).transpose().reshape(36)
        self.tac[113+288+108:113+288+144] = self.tac_buf_[113+288+108:113+288+144].reshape(6,6).transpose().reshape(36)

        self.tac[113+432:113+432+72] = self.tac_buf_[113+432:113+432+72]
        self.tac[113+432+72:113+432+108] = self.tac_buf_[113+432+72:113+432+108].reshape(6,6).transpose().reshape(36)

        self.pos, self.sensor_pos = self.env.sim2real(self.finger_jnt_pos,self.tac)

        return self.pos


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.task),
        cfg.task_name,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        multi_gpu=cfg.multi_gpu,
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    hand = AllegroHand(runner)
    hand.player_run()

if __name__ == '__main__':
    try:
        runner = launch_rlg_hydra()
    except rospy.ROSInterruptException:
        print('over!')


