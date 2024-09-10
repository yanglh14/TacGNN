from isaacgymenvs.encoder.model import *
# from model import *
from torch_geometric.data import Data
import os.path

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

class gnn_model_binary_pre():

    def __init__(self,device,num_envs, touchmodedir, touchmodelexist, test):
        ### Set the random seed for reproducible results
        torch.manual_seed(43)

        self.training_episode_num = 0
        self.touchmodedir, self.touchmodelexist,self.test = touchmodedir, touchmodelexist, test
        self.save_dir = 'runs/'+self.touchmodedir+'/touchmodel'

        ### Define the loss function
        self.loss_fn = torch.nn.MSELoss()
        self.diz_loss = {'train_loss': [], 'val_loss': []}
        self.pos_pre_bool = True
        if self.touchmodelexist:
            self.diz_loss['train_loss'] = list(np.load(self.save_dir+'/train_loss.npy'))
            self.diz_loss['val_loss'] = list(np.load(self.save_dir+'/val_loss.npy'))
            self.model = torch.load(self.save_dir+'/model.pt', map_location='cuda:0')
        else:
            self.model = GNNEncoderB(device=device,pos_pre_bool= self.pos_pre_bool)
        # self.model = torch.load('model_binary.pt', map_location=device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-05)

        self.horizon_length = 100
        self.num_envs = num_envs
        self.epoch_num = 10
        self.device = device
        self.step_n = 0
        self.train_bool = True

        self.obs_buf = torch.zeros(
            (self.num_envs*self.horizon_length, 653, 1), device=self.device, dtype=torch.float)

        self.pos_buf = torch.zeros(
            (self.num_envs*self.horizon_length, 653, 3), device=self.device, dtype=torch.float)

        self.y_buf = torch.zeros(
            (self.num_envs*self.horizon_length, 6), device=self.device, dtype=torch.float)

        self.pos_pre_buf = torch.zeros(
            (self.num_envs * self.horizon_length, 6), device=self.device, dtype=torch.float)

    def step(self,obs,pos,y,pos_pre = None):

        pos = pos * 100
        y = y * 100
        self.obs_buf[self.step_n*self.num_envs:(self.step_n+1)*self.num_envs,:,:] = obs.view(self.num_envs,-1,1)
        self.pos_buf[self.step_n*self.num_envs:(self.step_n+1)*self.num_envs,:,:] = pos
        self.y_buf[self.step_n*self.num_envs:(self.step_n+1)*self.num_envs,:] = y
        if self.pos_pre_bool:
            self.pos_pre_buf[self.step_n * self.num_envs:(self.step_n + 1) * self.num_envs, :] = pos_pre *100

        self.step_n += 1
        if not self.test:

            if self.step_n == self.horizon_length and self.train_bool:
                self.data_process()
                self.step_n = 0
                self.train()

        return self.forward(obs.view(self.num_envs,-1,1),pos,pos_pre)

    @torch.no_grad()
    def forward(self, obs, pos, pos_pre):
        pos_predict = pos_pre.clone()
        self.model.eval()
        tactile_dataset = []
        for i in range(obs.shape[0]):

            if obs[i].max() >0:
                data = Data(x=obs[i, obs[i, :, 0] != 0, :], pos=pos[i, obs[i, :, 0] != 0, :], pos_pre = pos_pre[i,:].view(1,-1))

                tactile_dataset.append(data)
        if tactile_dataset.__len__() !=0:
            data_loader = DataLoader(tactile_dataset, batch_size=tactile_dataset.__len__())

            for data in data_loader:
                logits = self.model(data.x, data.pos, data.batch, data.pos_pre)

            pos_predict[obs[:,:,0].max(1).values >0,:] = logits

        return pos_predict

    def train(self):

        print('Start Training Encoder! Episode Num {} \t Training data set {} \t Validation data set {}'.format(self.training_episode_num,self.train_loader.dataset.__len__(),self.valid_loader.dataset.__len__()))
        self.training_episode_num +=1
        for epoch in range(self.epoch_num):
            train_loss = self.train_epoch()
        val_loss = self.test_epoch()

        print('\n train loss {} \t val loss {}'.format(train_loss, val_loss))

        self.diz_loss['train_loss'].append(train_loss)
        self.diz_loss['val_loss'].append(val_loss)

        os.makedirs(self.save_dir,exist_ok =True)
        if not self.training_episode_num ==1:
            if self.diz_loss['val_loss'][-1] < self.diz_loss['val_loss'][-2]:
                torch.save(self.model, os.path.join(self.save_dir, 'model.pt'))

        torch.save(self.model, os.path.join(self.save_dir, 'model_%d_%f.pt'%(self.training_episode_num,self.diz_loss['val_loss'][-1])))
        np.save(os.path.join(self.save_dir, 'train_loss'), np.array(self.diz_loss['train_loss']))
        np.save(os.path.join(self.save_dir, 'val_loss'), np.array(self.diz_loss['val_loss']))

    def data_process(self):

        tactile_dataset = []
        for i in range(self.obs_buf.shape[0]):

            if self.obs_buf[i].max() >0:

                data = Data(x=self.obs_buf[i, self.obs_buf[i, :, 0] != 0, :], pos=self.pos_buf[i, self.obs_buf[i, :, 0] != 0, :], y=self.y_buf[i].view(1,-1), pos_pre = self.pos_pre_buf[i].view(1,-1))
                tactile_dataset.append(data)

        m = len(tactile_dataset)

        train_data, val_data = random_split(tactile_dataset, [m - int(m * 0.2), int(m * 0.2)])

        self.train_loader = DataLoader(train_data, batch_size=32)
        self.valid_loader = DataLoader(val_data, batch_size=32)

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        for data in self.train_loader:
            self.optimizer.zero_grad()  # Clear gradients.
            logits = self.model(data.x, data.pos, data.batch, data.pos_pre)  # Forward pass.
            loss = self.loss_fn(logits, data.y)  # Loss computation.
            loss.backward()  # Backward pass.
            self.optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.model.eval()
        total_loss = 0
        for data in self.valid_loader:
            logits = self.model(data.x, data.pos, data.batch, data.pos_pre)  # Forward pass.
            loss = self.loss_fn(logits, data.y)  # Loss computation.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.valid_loader.dataset)
