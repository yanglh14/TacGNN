from isaacgymenvs.encoder.model import *
# from isaacgymenvs.encoder.gnn import *
import os.path
from torch.utils.data import TensorDataset

import torch
from torch.utils.data import random_split
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

class cnn_model():

    def __init__(self,device,num_envs, touchmodedir, touchmodelexist, test):
        ### Set the random seed for reproducible results
        torch.manual_seed(43)

        self.training_episode_num = 0
        self.touchmodedir, self.touchmodelexist,self.test = touchmodedir, touchmodelexist, test
        self.save_dir = 'runs/'+self.touchmodedir+'/touchmodel'

        ### Define the loss function
        self.loss_fn = torch.nn.MSELoss()
        self.diz_loss = {'train_loss': [], 'val_loss': []}

        if self.touchmodelexist:
            self.model = torch.load(self.save_dir+'/model.pt', map_location='cuda:0')
            self.diz_loss['train_loss'] = list(np.load(self.save_dir+'/train_loss.npy'))
            self.diz_loss['val_loss'] = list(np.load(self.save_dir+'/val_loss.npy'))
        else:
            self.model = CNNEncoder()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-05)

        self.horizon_length = 100
        self.num_envs = num_envs
        self.epoch_num = 10
        self.device = device
        self.step_n = 0
        self.train_bool = True
        self.model.to(self.device)

        self.obs_buf = torch.zeros(
            (self.num_envs*self.horizon_length, 653, 1), device=self.device, dtype=torch.float)

        self.pos_buf = torch.zeros(
            (self.num_envs*self.horizon_length, 653, 3), device=self.device, dtype=torch.float)

        self.y_buf = torch.zeros(
            (self.num_envs*self.horizon_length, 6), device=self.device, dtype=torch.float)



    def step(self,obs,pos,y,pos_pre = None):

        pos = pos * 100
        y = y * 100

        if not self.test and self.train_bool:
            self.obs_buf[self.step_n * self.num_envs:(self.step_n + 1) * self.num_envs, :, :] = obs.view(self.num_envs,
                                                                                                         -1, 1)
            self.pos_buf[self.step_n * self.num_envs:(self.step_n + 1) * self.num_envs, :, :] = pos
            self.y_buf[self.step_n * self.num_envs:(self.step_n + 1) * self.num_envs, :] = y

            self.step_n += 1
            if self.step_n == self.horizon_length and self.train_bool:
                self.data_process()
                self.step_n = 0
                self.train()

        return self.forward(obs.view(self.num_envs,-1,1),pos)

    @torch.no_grad()
    def forward(self, obs, pos):

        self.model.eval()

        tac = self.tactile_process(obs.view(-1,653))
        x = tac.reshape(-1, 1, 36, 36)

        logits = self.model(x)

        return logits

    def train(self):

        print('Start Training Encoder! Episode Num {} \t Training data set {} \t Validation data set {}'.format(self.training_episode_num,self.train_loader.dataset.__len__(),self.valid_loader.dataset.__len__()))
        self.training_episode_num +=1
        for epoch in range(self.epoch_num):
            train_loss = self.train_epoch()
        val_loss = self.test_epoch()

        print('\n train loss {} \t val loss {}'.format(train_loss, val_loss))

        self.diz_loss['train_loss'].append(train_loss)
        self.diz_loss['val_loss'].append(val_loss)

        os.makedirs(self.save_dir,exist_ok = True)
        if not self.training_episode_num == 1:
            if self.diz_loss['val_loss'][-1] < self.diz_loss['val_loss'][-2]:
                torch.save(self.model, os.path.join(self.save_dir, 'model.pt'))

        torch.save(self.model, os.path.join(self.save_dir, 'model_%d_%f.pt'%(self.training_episode_num,self.diz_loss['val_loss'][-1])))
        np.save(os.path.join(self.save_dir, 'train_loss'), np.array(self.diz_loss['train_loss']))
        np.save(os.path.join(self.save_dir, 'val_loss'), np.array(self.diz_loss['val_loss']))

    def data_process(self):

        tac = self.tactile_process(self.obs_buf.view(-1,653))
        x = tac.reshape(-1, 1, 36, 36)
        y = self.y_buf
        tactile_dataset = TensorDataset(x, y)

        m = len(tactile_dataset)

        train_data, val_data = random_split(tactile_dataset, [int(m * 0.8), m - int(m * 0.8)])

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
        self.valid_loader = torch.utils.data.DataLoader(val_data, batch_size=32)


    def train_epoch(self):

        self.model.train()

        train_loss = []

        for features, labels in self.train_loader:
            self.optimizer.zero_grad()  # Clear gradients.
            logits = self.model(features)  # Forward pass.
            loss = self.loss_fn(logits, labels)  # Loss computation.
            loss.backward()  # Backward pass.
            self.optimizer.step()  # Update model parameters.
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    @torch.no_grad()
    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.model.eval()
        test_loss = []
        for features, labels in self.valid_loader:
            logits = self.model(features)  # Forward pass.
            loss = self.loss_fn(logits, labels)  # Loss computation.
            test_loss.append(loss.detach().cpu().numpy())

        return np.mean(test_loss)


    def tactile_process(self,data):
        data = data.detach().cpu().numpy()
        tactile_list = []
        for j in range(data.shape[0]):

            tactile = data[j]
            tac = np.zeros([36,36])

            tac[24,18:18+15] = tactile[:15]
            tac[25,18:18+15] = tactile[15:30]
            tac[26,18:18+15] = tactile[30:45]
            tac[27,18:18+15] = tactile[45:60]

            tac[28,19:19+13] = tactile[60:73]
            tac[29,18+8:18+8+6] = tactile[73:79]
            tac[30,18+8:18+8+6] = tactile[79:85]
            tac[31,18+8:18+8+6] = tactile[85:91]
            tac[32,18+8:18+8+6] = tactile[91:97]

            tac[33,18+8:18+8+5] = tactile[97:102]
            tac[34,18+8:18+8+4] = tactile[102:106]
            tac[35,18+8:18+8+7] = tactile[106:113]

            for i in range(12):
                tac[12-(i+1),18:18+6] = tactile[113+i*6:113+(i+1)*6]
            for i in range(12):
                tac[12+i,18:18+6] = tactile[113+72+i*6:113+72+(i+1)*6]

            for i in range(12):
                tac[12-(i+1),24:24+6] = tactile[113+144+i*6:113+144+(i+1)*6]
            for i in range(12):
                tac[12+i,24:24+6] = tactile[113+144+72+i*6:113+144+72+(i+1)*6]

            for i in range(12):
                tac[12-(i+1),30:36] = tactile[113+288+i*6:113+288+(i+1)*6]
            for i in range(12):
                tac[12+i,30:36] = tactile[113+288+72+i*6:113+288+72+(i+1)*6]

            for i in range(12):
                tac[-6:,12-(i+1)] = tactile[113+432+i*6:113+432+(i+1)*6]
            for i in range(6):
                tac[-6:,12+i] = tactile[113+432+72+i*6:113+432+72+(i+1)*6]
            tactile_list.append(tac)
        return torch.tensor(tactile_list,device=self.device, dtype=torch.float)