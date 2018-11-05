import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, state_size, action_size, hidden_in_dim, hidden_out_dim, activation=F.relu, is_actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size,hidden_in_dim)
        self.bn1 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2_actor = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc2_critic = nn.Linear(hidden_in_dim+action_size,hidden_out_dim)
        self.bn2 = nn.BatchNorm1d(hidden_out_dim)
        self.fc3_actor = nn.Linear(hidden_out_dim,action_size)
        self.fc3_critic = nn.Linear(hidden_out_dim,1)
        self.activation = activation 
        self.is_actor = is_actor
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x, action=None):
        if self.is_actor:
            # return a vector of the force
            x = self.bn0(x)
            x = self.activation(self.bn1(self.fc1(x)))
            x = self.activation(self.bn2(self.fc2_actor(x)))
            return torch.tanh(self.fc3_actor(x))
        
        else:
            # critic network simply outputs a number
            x = self.bn0(x)
            x = self.activation(self.bn1(self.fc1(x)))
            x = torch.cat((x, action), dim=-1)
            x = self.activation(self.fc2_critic(x))
            return self.fc3_critic(x)