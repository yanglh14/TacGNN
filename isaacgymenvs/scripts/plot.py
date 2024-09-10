import numpy as np
import time
import matplotlib.pyplot as plt
from IPython import display
import math

def time_plot(data):

    time_log = data['time_log']
    time_log = np.array(time_log)-time_log[0]
    control_freq =[]
    for i in range(time_log.shape[0]-1):
        control_freq.append(1/(time_log[i+1]-time_log[i]))

    fig, axs = plt.subplots(1)
    axs.plot(control_freq)
    plt.show()
def tactile_plot_sim(data):

    tactile_log = np.array(data['tactile_log'])
    tactile_pos_log = np.array(data['tactile_pos_log'])
    object_pre_log = np.array(data['object_pre_log'])/100
    object_pos_log = np.array(data['object_pos_log'])

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    for i in range(tactile_log.shape[0]):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        i= i
        print(i)
        x = tactile_pos_log[i,:,0]
        y = tactile_pos_log[i,:,1]
        z = tactile_pos_log[i,:,2]
        tac = tactile_log[i]

        ax.clear()
        ax.scatter(x, y, z, s=(tac) * 100+1)
        ax.scatter(x, y, z, c='r', s=(tac) * 100)
        ax.scatter(object_pre_log[i,0], object_pre_log[i,1], object_pre_log[i,2], c='g', s=6000)
        ax.scatter(object_pre_log[i,3], object_pre_log[i,4], object_pre_log[i,5], c='g', s=6000)
        ax.scatter(object_pos_log[i,0], object_pos_log[i,1], object_pos_log[i,2], c='orange', s=6000)
        ax.scatter(object_pos_log[i,3], object_pos_log[i,4], object_pos_log[i,5], c='orange', s=6000)
        ax.view_init(elev=45, azim=45)
        ax.set(xlim=[-0.1, 0.1], ylim=[-0.1, 0.1], zlim=[0.5, 0.6])

        display.clear_output(wait=True)
        plt.pause(0.00000001)
        plt.show()
def tactile_plot(data):
    cal_table = np.load('calibration_table.npy')

    tactile_log = np.array(data['tactile_log'])
    tactile_pos_log = np.array(data['tactile_pos_log'])
    tac_init = np.array(data['tac_init'])
    tac = np.zeros(653)
    object_pre_log = np.array(data['object_pre_log'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(tactile_pos_log.shape[0]):
        i= i
        print(i)
        x = tactile_pos_log[i,:,0]
        y = tactile_pos_log[i,:,1]
        z = tactile_pos_log[i,:,2]
        tac_ = tactile_log[i,:] - tac_init

        tac_[tac_ < cal_table[:, 1]] = 0
        tac_ *= cal_table[:, 0]

        tac[:15] = [  tac_[109],  tac_[105],  tac_[100],  tac_[95],  tac_[90],  tac_[85],  tac_[80],  tac_[75],  tac_[69],  tac_[59],  tac_[48],  tac_[36],  tac_[24],  tac_[12],  tac_[0]]
        tac[15:30] = [  tac_[109+1],  tac_[105+1],  tac_[100+1],  tac_[95+1],  tac_[90+1],  tac_[85+1],  tac_[80+1],  tac_[75+1],  tac_[69+1],  tac_[59+1],  tac_[48+1],  tac_[36+1],  tac_[24+1],  tac_[12+1],  tac_[0+1]]
        tac[30:45] = [  tac_[109+2],  tac_[105+2],  tac_[100+2],  tac_[95+2],  tac_[90+2],  tac_[85+2],  tac_[80+2],  tac_[75+2],  tac_[69+2],  tac_[59+2],  tac_[48+2],  tac_[36+2],  tac_[24+2],  tac_[12+2],  tac_[0+2]]
        tac[45:60] = [  tac_[109+3],  tac_[105+3],  tac_[100+3],  tac_[95+3],  tac_[90+3],  tac_[85+3],  tac_[80+3],  tac_[75+3],  tac_[69+3],  tac_[59+3],  tac_[48+3],  tac_[36+3],  tac_[24+3],  tac_[12+3],  tac_[0+3]]
        tac[60:73] = [  tac_[100+4],  tac_[95+4],  tac_[90+4],  tac_[85+4],  tac_[80+4],  tac_[75+4],  tac_[69+4],  tac_[59+4],  tac_[48+4],  tac_[36+4],  tac_[24+4],  tac_[12+4],  tac_[0+4]]
        tac[73:79] = [  tac_[59+5],  tac_[48+5],  tac_[36+5],  tac_[24+5],  tac_[12+5],  tac_[0+5]]
        tac[79:85] = [  tac_[59+6],  tac_[48+6],  tac_[36+6],  tac_[24+6],  tac_[12+6],  tac_[0+6]]
        tac[85:91] = [  tac_[59+7],  tac_[48+7],  tac_[36+7],  tac_[24+7],  tac_[12+7],  tac_[0+7]]
        tac[91:97] = [  tac_[59+8],  tac_[48+8],  tac_[36+8],  tac_[24+8],  tac_[12+8],  tac_[0+8]]
        tac[97:102] = [  tac_[48+9],  tac_[36+9],  tac_[24+9],  tac_[12+9],  tac_[0+9]]
        tac[102:106] = [  tac_[36+10],  tac_[24+10],  tac_[12+10],  tac_[0+10]]
        tac[106:113] = [  tac_[74],  tac_[68],  tac_[458],  tac_[36+11],  tac_[24+11],  tac_[12+11],  tac_[0+11]]

        tac[113:113+72] =   tac_[113:113+72]
        tac[113+72:113 + 72+36] =   tac_[113+72:113+72+36].reshape(6,6).transpose().reshape(36)
        tac[113+72+36:113+144] =   tac_[113+72+36:113+144].reshape(6,6).transpose().reshape(36)

        tac[113+144:113+144+72] =   tac_[113+144:113+144+72]
        tac[113+144+72:113+144+108] =   tac_[113+144+72:113+144+108].reshape(6,6).transpose().reshape(36)
        tac[113+144+108:113+288] =   tac_[113+144+108:113+288].reshape(6,6).transpose().reshape(36)

        tac[113+288:113+288+72] =   tac_[113+288:113+288+72]
        tac[113+288+72:113+288+108] =   tac_[113+288+72:113+288+108].reshape(6,6).transpose().reshape(36)
        tac[113+288+108:113+288+144] =   tac_[113+288+108:113+288+144].reshape(6,6).transpose().reshape(36)

        tac[113+432:113+432+72] =   tac_[113+432:113+432+72]
        tac[113+432+72:113+432+108] =   tac_[113+432+72:113+432+108].reshape(6,6).transpose().reshape(36)

        ax.clear()

        ax.scatter(x, y, z, s=(tac) * 100+1)
        ax.scatter(x, y, z, c='r', s=(tac) * 100)
        ax.scatter(object_pre_log[i,0], object_pre_log[i,1], object_pre_log[i,2], c='g', s=6000)
        ax.scatter(object_pre_log[i,3], object_pre_log[i,4], object_pre_log[i,5], c='g', s=6000)
        ax.view_init(elev=45, azim=45)
        ax.set(xlim=[-0.1, 0.1], ylim=[-0.1, 0.1], zlim=[0.5, 0.6])

        display.clear_output(wait=True)
        plt.pause(0.00000001)

def object_plot(data):
    object_pos_log = np.array(data['object_pos_log'])*100
    object_pre_log = np.array(data['object_pre_log'])

    fig, axs = plt.subplots(2,3)
    for i in range(6):
        axs[int(i / 3), i % 3].plot(np.array(object_pos_log)[:,i], label='pos')
        axs[int(i / 3), i % 3].plot(np.array(object_pre_log)[:,i], label='pre')
        axs[int(i / 3), i % 3].legend()
    plt.show()

def object_plot_2(data):
    object_pos_log = np.array(data['object_pos_log'])*100
    object_pre_log = np.array(data['object_pre_log'])*100

    fig, axs = plt.subplots(2)

    for i in range(object_pre_log.shape[0]):
        axs[0].clear()
        axs[1].clear()

        axs[0].set_xlim(-5,5)
        axs[1].set_xlim(-5,5)
        axs[0].set_ylim(-8,5)
        axs[1].set_ylim(-8,5)

        if i < 20:
            # axs[0].scatter(np.array(object_pos_log)[:i,0],np.array(object_pos_log)[:i,1], label='object1_pos')
            # axs[1].scatter(np.array(object_pos_log)[:i,3],np.array(object_pos_log)[:i,4], label='object2_pos')

            axs[0].scatter(np.array(object_pre_log)[:i,0],np.array(object_pre_log)[:i,1], label='object1_pre')
            axs[1].scatter(np.array(object_pre_log)[:i,3],np.array(object_pre_log)[:i,4], label='object2_pre')
        else:
            # axs[0].scatter(np.array(object_pos_log)[i-20:i,0],np.array(object_pos_log)[i-20:i,1], label='object1_pos')
            # axs[1].scatter(np.array(object_pos_log)[i-20:i,3],np.array(object_pos_log)[i-20:i,4], label='object2_pos')

            axs[0].scatter(np.array(object_pre_log)[i-20:i,0],np.array(object_pre_log)[i-20:i,1], label='object1_pre')
            axs[1].scatter(np.array(object_pre_log)[i-20:i,3],np.array(object_pre_log)[i-20:i,4], label='object2_pre')

        axs[0].legend()
        axs[1].legend()
        display.clear_output(wait=True)
        plt.pause(0.00000001)
    # plt.show()

def joint_plot(data,data_2=None):
    joint_pos = np.array(data['joints_log'])
    targets_log = np.array(data['targets_log'])
    actions_log = np.array(data['actions_log'])

    if data_2 != None:
        joint_pos_2 = np.array(data_2['joints_log'])
        targets_log_2 = np.array(data_2['targets_log'])
        actions_log_2 = np.array(data_2['actions_log'])

    fig, axs = plt.subplots(3,4)
    for i in range(12):
        axs[int(i / 4), i % 4].plot(np.array(joint_pos)[:,i], label='joint_sim')
        # axs[int(i / 4), i % 4].plot(np.array(actions_log)[:,i], label='action')
        axs[int(i / 4), i % 4].plot(np.array(targets_log)[:,i], label='target')

        if data_2 != None:
            axs[int(i / 4), i % 4].plot(np.array(joint_pos_2)[:, i], label='joint_real')
            # axs[int(i / 4)+4, i % 4].plot(np.array(actions_log_2)[:,i], label='action_2')
            # axs[int(i / 4)+4, i % 4].plot(np.array(targets_log_2)[:, i], label='target_2')

        axs[int(i / 4), i % 4].legend()
    plt.show()

def obs_plot(index_1,index_2,data,data_2=None):
    obs = np.array(data['obs_log'])
    if data_2 != None:

        obs_2 = np.array(data_2['obs_log'])

    num = index_2-index_1
    fig, axs = plt.subplots(math.ceil(num / 4),4)

    for i in range(index_2-index_1):
        j  = i + index_1
        axs[int(i / 4), i % 4].plot(obs[:,j], label='real')

        if data_2 != None:
            axs[int(i / 4), i % 4].plot(obs_2[:, j], label='sim')

        axs[int(i / 4), i % 4].legend()
    plt.show()

def vel_plot(data):
    obs = np.array(data['obs_log'])
    pos = np.array(obs[:,:16])
    vel = np.array(obs[:,16:32])/10000
    time_log = data['time_log']
    time_log = np.array(time_log)-time_log[0]

    control_freq =[]
    for i in range(time_log.shape[0]-1):
        control_freq.append((time_log[i+1]-time_log[i]))

    fig, axs = plt.subplots(8,4)

    for i in range(16):
        pos_gra = []
        for j in range(pos.shape[0]-1):
            pos_gra.append((pos[j+1,i]-pos[j,i])/control_freq[i])

        axs[int(i / 4), i % 4].plot(vel[:, i], label='vel')
        axs[int(i / 4)+4, i % 4].plot(pos_gra[:], label='pos_gra')
        # axs[int(i / 4), i % 4].legend()
    plt.show()
if __name__ == '__main__':
    data = np.load('../runs/sim_log.npy', allow_pickle=True)
    data = data.item()
    # data_2 = np.load('../runs/real_log_1661807304.662783.npy', allow_pickle=True)
    # data_2 = data_2.item()
    tactile_plot_sim(data)