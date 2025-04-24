import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """Decentralized policy: 只看单个 agent 的局部观测 o_i -> a_i"""
    def __init__(self, nb_actions, action_space_bound, action_space_boundMove, cnn: nn.Module):
        super().__init__()
        self.cnn = cnn
        self.action_space_bound = action_space_bound
        self.action_space_boundMove = action_space_boundMove
        self.mlp = nn.Sequential(
            nn.Linear(5*5*128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, nb_actions),
            nn.Tanh(),  # 输出范围 [-1,1]
        )

    def forward(self, obs: torch.Tensor):
        # obs: (batch, C, H, W)
        feat = self.cnn(obs)                           # -> (batch, 128, 5, 5)
        feat = feat.view(feat.size(0), -1)             # -> (batch, 5*5*128)
        a = self.mlp(feat)                             # -> (batch, nb_actions)
        # 若需要线性映射到实际 action_space，可在外面做： a * bound + move
        return a


class CriticNet(nn.Module):
    """Centralized critic: 看所有 agents 的 obs 和 all actions"""
    def __init__(self, nb_actions, num_agents, cnn: nn.Module):
        super().__init__()
        self.num_agents = num_agents
        self.cnn = cnn
        # 每个 obs 经 CNN 得到 5*5*128 大小的特征向量
        feat_dim = 5*5*128
        # Critic 输入是： num_agents*feat_dim  +  num_agents*nb_actions
        input_dim = num_agents * (feat_dim + nb_actions)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self,
                obs_all: torch.Tensor,
                act_all: torch.Tensor):
        """
        obs_all: (batch, N, C, H, W)
        act_all: (batch, N, nb_actions)
        """
        B, N, C, H, W = obs_all.shape
        # 1) flatten N agents into batch*N for CNN
        obs_flat = obs_all.view(B * N, C, H, W)
        feat_flat = self.cnn(obs_flat)                     # -> (B*N, 128, 5, 5)
        feat_flat = feat_flat.view(B * N, -1)              # -> (B*N, feat_dim)
        # 2) restore per-agent grouping
        feat = feat_flat.view(B, N * feat_flat.size(-1))   # -> (B, N*feat_dim)
        act = act_all.view(B, N * act_all.size(-1))        # -> (B, N*nb_actions)
        # 3) concat 全部特征 + 全部动作
        x = torch.cat([feat, act], dim=1)                  # -> (B, input_dim)
        # 4) 得到 Q 值
        q = self.net(x)                                    # -> (B, 1)
        return q
