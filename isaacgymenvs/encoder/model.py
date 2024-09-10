from torch import nn
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch_cluster import fps

class MLPEncoder(nn.Module):
    def __init__(self, encoded_space_dim = 6):
        super().__init__()

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(653, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )

    def forward(self, x):

        x = self.encoder_lin(x)
        return x

class CNNEncoder(nn.Module):

    def __init__(self, encoded_space_dim = 6):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(7 * 7 * 16, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 9 * 9 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 9, 9))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=1,
                               padding=0, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=1,
                               padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,device):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.device = device

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,device=self.device)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim,device=self.device)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,device):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels*2 + 3, out_channels,device=device),
                              ReLU(),
                              Linear(out_channels, out_channels,device=device))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_i, h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.

class GNNEncoderB(torch.nn.Module):
    def __init__(self,device,output_dim=6,pos_pre_bool= False,channels = 64):
        super().__init__()
        self.pos_pre_bool = pos_pre_bool
        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, channels,device=device)
        self.conv2 = PointNetLayer(channels, channels,device=device)
        self.conv3 = PointNetLayer(channels, channels,device=device)
        if pos_pre_bool:
            self.regression = Linear(channels+6, output_dim,device=device)
        else:
            self.regression = Linear(channels, output_dim, device=device)

    def forward(self,x, pos, batch, pos_pre=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        index = fps(pos, batch=batch, ratio=0.5)
        pos = pos[index]
        h = h[index]
        batch = batch[index]

        edge_index = knn_graph(pos, k=4, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        index = fps(pos, batch=batch, ratio=0.5)
        pos = pos[index]
        h = h[index]
        batch = batch[index]

        edge_index = knn_graph(pos, k=3, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        if self.pos_pre_bool:
            assert pos_pre != None
            h = torch.cat([h,pos_pre],axis = 1)
        o = self.regression(h)

        return o

class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,device):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels, out_channels,device=device),
                              ReLU(),
                              Linear(out_channels, out_channels,device=device))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_i, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = h_j  # Compute spatial relation.

        return self.mlp(input)  # Apply our final MLP.

class GCNEncoder(torch.nn.Module):
    def __init__(self,device,output_dim=6,channels = 32):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = GCNLayer(4, channels,device=device)
        self.conv2 = GCNLayer(channels, channels,device=device)
        self.conv3 = GCNLayer(channels, channels,device=device)

        self.regression = Linear(channels, output_dim,device=device)

    def forward(self,x, pos, batch, pos_pre=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.

        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        h = self.conv1(h=torch.cat([x, pos], dim=-1), pos=pos, edge_index=edge_index)
        h = h.relu()

        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        h = self.conv3(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        o = self.regression(h)

        return o

if __name__ == "__main__":
    model = torch.load('../model.pt', map_location='cuda:0')
    print('sss')