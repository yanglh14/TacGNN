from isaacgymenvs.encoder.model import *
from torch_geometric.data import Data
import os.path

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import pickle
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # plotting library

class gnn_lstm_model():

    def __init__(self, device, num_envs, touchmodedir, touchmodelexist, test):
        ### Set the random seed for reproducible results
        torch.manual_seed(43)
        self.training_episode_num = 0
        self.touchmodedir, self.touchmodelexist, self.test = touchmodedir, touchmodelexist, test
        self.save_dir = 'runs/'+self.touchmodedir+'/touchmodel'

        ### Define the loss function
        self.loss_fn = torch.nn.MSELoss()
        self.diz_loss = {'train_loss': [], 'val_loss': []}

        if self.touchmodelexist:
            self.diz_loss['train_loss'] = list(np.load(self.save_dir+'/train_loss.npy'))
            self.diz_loss['val_loss'] = list(np.load(self.save_dir+'/val_loss.npy'))

            self.gnn = torch.load(self.save_dir+'/gnn.pt')
            self.lstm = torch.load(self.save_dir+'/lstm.pt')

        else:
            self.gnn = PointNet(device=device,output_dim=16)
            input_dim,hidden_dim,layer_dim,output_dim = 16,16,3,6
            self.lstm = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim,device)

        params_to_optimize = [
            {'params': self.gnn.parameters()},
            {'params': self.lstm.parameters()}
        ]

        self.optimizer = torch.optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)

        self.horizon_length = 10
        self.buf_size = 100
        self.num_envs = num_envs
        self.epoch_num = 10
        self.device = device
        self.step_n = 0
        self.batch_size = 16
        self.obs_buf = torch.zeros(
            (self.num_envs*self.buf_size,self.horizon_length, 653, 1), device=self.device, dtype=torch.float)

        self.pos_buf = torch.zeros(
            (self.num_envs*self.buf_size,self.horizon_length, 653, 3), device=self.device, dtype=torch.float)

        self.y_buf = torch.zeros(
            (self.num_envs*self.buf_size,self.horizon_length, 6), device=self.device, dtype=torch.float)

        self.obs_lstm = torch.zeros(
            (self.num_envs,self.horizon_length, 653, 1), device=self.device, dtype=torch.float)

        self.pos_lstm = torch.zeros(
            (self.num_envs,self.horizon_length, 653, 3), device=self.device, dtype=torch.float)

        self.y_lstm = torch.zeros(
            (self.num_envs,self.horizon_length, 6), device=self.device, dtype=torch.float)

    def step(self,obs,pos,y):

        pos = pos*100
        y = y *100

        self.obs_lstm[:,:self.horizon_length-1,:,:] = self.obs_lstm[:,1:self.horizon_length,:,:].clone()
        self.pos_lstm[:,:self.horizon_length-1,:,:] = self.pos_lstm[:,1:self.horizon_length,:,:].clone()
        self.y_lstm[:,:self.horizon_length-1,:] = self.y_lstm[:,1:self.horizon_length,:].clone()

        self.obs_lstm[:,-1,:,:] = obs.view(self.num_envs,-1,1)
        self.pos_lstm[:,-1,:,:] = pos
        self.y_lstm[:,-1,:] = y

        self.obs_buf[self.step_n*self.num_envs:(self.step_n+1)*self.num_envs,:,:,:] = self.obs_lstm
        self.pos_buf[self.step_n*self.num_envs:(self.step_n+1)*self.num_envs,:,:,:] = self.pos_lstm
        self.y_buf[self.step_n*self.num_envs:(self.step_n+1)*self.num_envs,:,:] = self.y_lstm

        self.step_n += 1
        if not self.test:

            if self.step_n == self.buf_size:
                self.data_process()
                self.step_n = 0
                self.train()

        return self.forward(self.obs_lstm,self.pos_lstm)

    @torch.no_grad()
    def forward(self, obs, pos):

        self.gnn.eval()
        self.lstm.eval()

        obs = obs.view(-1,653,1)
        pos = pos.view(-1,653,3)

        tactile_dataset = []
        for i in range(obs.shape[0]):

            if obs.max() >0:

                obs[i] /= obs[i].max(0, keepdim=True)[0]

                data = Data(x=obs[i, obs[i, :, 0] != 0, :], pos=pos[i, obs[i, :, 0] != 0, :])

            else:
                data = Data(x=obs[i, :, :], pos=pos[i, :, :])

            tactile_dataset.append(data)

        data_loader = DataLoader(tactile_dataset, batch_size=tactile_dataset.__len__())

        for data in data_loader:
            logits = self.gnn(data.x, data.pos, data.batch)
            logits = self.lstm(logits.view(-1, self.horizon_length, logits.shape[-1]))

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

        os.makedirs(self.save_dir,exist_ok =True)

        if not self.training_episode_num ==1:
            if self.diz_loss['val_loss'][-1] > self.diz_loss['val_loss'][-2]:
                torch.save(self.gnn, os.path.join(self.save_dir, 'gnn.pt'))
                torch.save(self.lstm, os.path.join(self.save_dir, 'lstm.pt'))

        torch.save(self.gnn, os.path.join(self.save_dir, 'gnn_%d_%f.pt'%(self.training_episode_num,self.diz_loss['val_loss'][-1])))
        torch.save(self.lstm, os.path.join(self.save_dir, 'lstm_%d_%f.pt'%(self.training_episode_num,self.diz_loss['val_loss'][-1])))


        np.save(os.path.join(self.save_dir, 'train_loss'), np.array(self.diz_loss['train_loss']))
        np.save(os.path.join(self.save_dir, 'val_loss'), np.array(self.diz_loss['val_loss']))

    def data_process(self):

        tactile_dataset = []
        self.obs_buf = self.obs_buf.view(-1,653,1)
        self.pos_buf = self.pos_buf.view(-1,653,3)
        self.y_buf = self.y_buf.view(-1,6)

        for i in range(self.obs_buf.shape[0]):

            if self.obs_buf[i].max() >0:

                self.obs_buf[i] /= self.obs_buf[i].max(0, keepdim=True)[0]

                data = Data(x=self.obs_buf[i, self.obs_buf[i, :, 0] != 0, :], pos=self.pos_buf[i, self.obs_buf[i, :, 0] != 0, :], y=self.y_buf[i].view(1,-1))
                tactile_dataset.append(data)
            else:
                data = Data(x=self.obs_buf[i], pos=self.pos_buf[i], y=self.y_buf[i].view(1,-1))
                tactile_dataset.append(data)

        m = len(tactile_dataset)

        self.train_loader = DataLoader(tactile_dataset[:int(m*0.8)], batch_size=self.horizon_length*self.batch_size)
        self.valid_loader = DataLoader(tactile_dataset[int(m*0.8):], batch_size=self.horizon_length*self.batch_size)

        self.obs_buf = self.obs_buf.view(-1,self.horizon_length,653,1)
        self.pos_buf = self.pos_buf.view(-1,self.horizon_length,653,3)
        self.y_buf = self.y_buf.view(-1,self.horizon_length,6)

    def train_epoch(self):

        self.gnn.train()
        self.lstm.train()

        total_loss = 0
        for data in self.train_loader:
            self.optimizer.zero_grad()  # Clear gradients.
            logits = self.gnn(data.x, data.pos, data.batch)  # Forward pass.
            logits = self.lstm(logits.view(-1,self.horizon_length,logits.shape[-1]))

            loss = self.loss_fn(logits, data.y[self.horizon_length-1::self.horizon_length])  # Loss computation.
            loss.backward()  # Backward pass.
            self.optimizer.step()  # Update model parameters.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.gnn.eval()
        self.lstm.eval()
        total_loss = 0
        for data in self.valid_loader:
            logits = self.gnn(data.x, data.pos, data.batch)  # Forward pass.
            logits = self.lstm(logits.view(-1,self.horizon_length,logits.shape[-1]))
            loss = self.loss_fn(logits, data.y[self.horizon_length-1::self.horizon_length])  # Loss computation.
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.valid_loader.dataset)
