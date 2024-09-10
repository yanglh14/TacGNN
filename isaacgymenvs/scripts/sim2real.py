import matplotlib.pyplot as plt
import numpy as np

stiffness_min = []
damping_min = []
real_joint_offset = []

for i in range(16):
    index = "%d"%i
    print(index)
    stiffness_list = [1, 3, 5]
    damping_list = [0, 0.1, 0.5]

    error_list = []
    fig, axs = plt.subplots(3,3)

    for j in range(9):

        stiffness = stiffness_list[int(j / 3)]
        damping = damping_list[int(j % 3)]

        joint_pos_real = np.load('real/joint_real_'+index+'.npy')

        joint_pos_sim = np.load('sim2real/joint_{}_stiffness{}_damping{}.npy'.format(i,stiffness,damping))

        joint_pos_real = joint_pos_real [:,i]
        joint_pos_sim = joint_pos_sim[:,i]

        if j == 0:
            real_joint_offset.append(joint_pos_real[0])
        joint_pos_real[:] = joint_pos_real[:] - joint_pos_real[0]
        if i == 12 or i == 14:
            joint_pos_real[:] += 20 * np.pi / 180

        error = np.linalg.norm(joint_pos_real-joint_pos_sim)
        error_list.append(error)

        if i == 12 or i == 14:
            angle_list = [20, 30, 45, 60, 75, 60, 45, 30, 20]
        else:
            if i % 4 == 0:
                angle_list = [0, 15, 25, 15, 0, -15, -25, -15, 0]
            else:
                angle_list = [0, 15, 30, 45, 60, 45, 30, 15, 0]

        joint_pos_target = np.concatenate(
            [np.ones(50) * np.pi * angle_list[0] / 180, np.ones(50) * np.pi * angle_list[1] / 180,
             np.ones(50) * np.pi * angle_list[2] / 180,
             np.ones(50) * np.pi * angle_list[3] / 180, np.ones(50) * np.pi * angle_list[4] / 180,
             np.ones(50) * np.pi * angle_list[5] / 180,
             np.ones(50) * np.pi * angle_list[6] / 180, np.ones(50) * np.pi * angle_list[7] / 180,
             np.ones(50) * np.pi * angle_list[8] / 180,
             ])

        print(i,stiffness,damping,error)
        axs[int(j/3),j%3].plot(joint_pos_real[:], label='real')
        axs[int(j/3),j%3].plot(joint_pos_sim[:], label='sim')
        axs[int(j/3),j%3].plot(joint_pos_target[:], label='target')
        axs[int(j/3),j%3].legend()
        axs[int(j/3),j%3].set(title = "joint{},stiffness{},damping{},error{}".format(i,stiffness,damping,error))

    stiffness_min.append(stiffness_list[int(np.argmin(error_list)/3)])
    damping_min.append(damping_list[int(np.argmin(error_list)%3)])

    plt.show()
