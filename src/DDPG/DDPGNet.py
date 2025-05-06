import torch.nn as nn
import torch
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, nb_actions, action_space_bound, action_space_boundMove, CNN,input_shape=(4, 512,512)):
        super(PolicyNet, self).__init__()
        self.CNN = CNN
        self.action_space_bound = action_space_bound
        self.action_space_boundMove = action_space_boundMove

        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dummy_input = torch.zeros(1, *input_shape).to(device)
            cnn_out = self.CNN(dummy_input)
            self.cnn_output_dim = int(np.prod(cnn_out.shape[1:]))

        self.model = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, nb_actions)
        )

    def forward(self, status):
        status = status.to(next(self.parameters()).device)
        status = self.CNN(status)
        status = status.view(status.size(0), -1).to(torch.float32)
        status = self.model(status)
        status = torch.tanh(status)
        # action = status * self.action_space_bound + self.action_space_boundMove
        return status

def compute_cnn_output_dim(cnn_module, input_shape):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_input = torch.zeros(1, *input_shape).to(device)
        out = cnn_module(dummy_input)
        return int(np.prod(out.size()))
class CriticNet(nn.Module):
    def __init__(self, nb_actions, CNN, input_shape=(4, 512,512)):
        super(CriticNet, self).__init__()
        self.CNN = CNN

        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dummy_input = torch.zeros(1, *input_shape).to(device)
            cnn_out = self.CNN(dummy_input)
            self.cnn_output_dim = int(np.prod(cnn_out.shape[1:]))

        self.model = nn.Sequential(
            nn.Linear(self.cnn_output_dim + nb_actions, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, status, action):
        status = status.to(next(self.parameters()).device)
        status = self.CNN(status)
        status = status.view(status.size(0), -1)
        status = torch.cat([status, action], dim=1).to(torch.float32)
        critic = self.model(status)
        return critic


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CNN = CNN().to(device)
# A = torch.tensor(RoadNet.action_space_bound).to(device)
# B = torch.tensor(RoadNet.action_space_boundMove).to(device)
# policy = PolicyNet(2, A, B, CNN).to(device)
# critic = CriticNet(2, CNN).to(device)
# print(policy)
# print(critic)
# summary(policy, input_size=(4, 512, 512))