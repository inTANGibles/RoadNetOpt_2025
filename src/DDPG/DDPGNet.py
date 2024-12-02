import torch.nn as nn
import torch

class PolicyNet(nn.Module):
    def __init__(self, nb_actions, action_space_bound, action_space_boundMove, CNN):
        super(PolicyNet, self).__init__()
        self.CNN = CNN
        self.action_space_bound = action_space_bound
        self.action_space_boundMove = action_space_boundMove

        self.model = nn.Sequential(
            nn.Linear(5*5*128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, nb_actions)
        )

    def forward(self, status):
        status = self.CNN(status)
        status = status.view(status.size(0), -1).to(torch.float32)
        status = self.model(status)
        status = torch.tanh(status)
        # action = status * self.action_space_bound + self.action_space_boundMove
        return status


class CriticNet(nn.Module):
    def __init__(self, nb_actions, CNN):
        super(CriticNet, self).__init__()
        self.CNN = CNN
        self.model = nn.Sequential(
            nn.Linear(5*5*128 + nb_actions, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, status, action):
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