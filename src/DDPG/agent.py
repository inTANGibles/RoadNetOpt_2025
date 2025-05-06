import torch.nn as nn
import torch
import torch.nn.functional as F
from DDPG.CNN import CNN
# from DDPG.DDPGNet import PolicyNet, CriticNet
from DDPG.DDPGNet import PolicyNet, CriticNet
class Agent:
    def __init__(self,
                 nb_actions: int,
                 num_agents: int,
                 action_space_bound,
                 action_space_boundMove,
                 actor_lr: float,
                 critic_lr: float,
                 tau: float,
                 gamma: float,
                 device):
        """
        nb_actions: 单个 agent 的动作维度
        num_agents: 一共多少个 agent（用于 CriticNet）
        """
        self.device = device
        self.gamma  = gamma
        self.tau    = tau

        # 1) 共享 CNN 特征提取器
        self.cnn = CNN().to(device)

        # 2) 去中心化 actor
        self.actor = PolicyNet(nb_actions,
                               action_space_bound.to(device),
                               action_space_boundMove.to(device),
                               self.cnn).to(device)
        self.target_actor = PolicyNet(nb_actions,
                                      action_space_bound.to(device),
                                      action_space_boundMove.to(device),
                                      self.cnn).to(device)

        # 3) 中心化 critic
        self.critic = CriticNet(nb_actions, self.cnn).to(device)
        self.target_critic = CriticNet(nb_actions, self.cnn).to(device)

        # 同步参数
        self.target_actor .load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)


    def take_action(self, obs_i: torch.Tensor):
        """
        obs_i: np.ndarray or Tensor, shape = (C,H,W) —— 单个 agent 的观测
        return: a_i (numpy) shape = (nb_actions,)
        """
        if not torch.is_tensor(obs_i):
            obs_i = torch.tensor(obs_i, dtype=torch.float32)
        obs_i = obs_i.unsqueeze(0).to(self.device) / 255.0  # → (1,C,H,W)
        a = self.actor(obs_i)                              # → (1, nb_actions)
        return a.cpu().detach().numpy()[0]


    def soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_( self.tau * p.data + (1-self.tau)*tp.data )

    def multi_agent_update(self, transition):
        # —— 数据准备 ——
        S = torch.tensor(transition['states'], dtype=torch.float32, device=self.device) / 255.0  # (B, C, H, W)
        A = torch.tensor(transition['actions'], dtype=torch.float32, device=self.device)  # (B, act_dim)
        R = torch.tensor(transition['rewards'], dtype=torch.float32, device=self.device)  # (B, 1)
        S2 = torch.tensor(transition['next_states'], dtype=torch.float32, device=self.device) / 255.0  # (B, C, H, W)
        D = torch.tensor(transition['dones'], dtype=torch.float32, device=self.device)  # (B, 1)

        # —— Critic 更新 ——
        with torch.no_grad():
            a_next = self.target_actor(S2)  # (B, act_dim)
            q_next = self.target_critic(S2, a_next)  # (B, 1)
            q_target = R + self.gamma * q_next * (1 - D)
            # print("R", R.shape, "q_next", q_next.shape, "D", D.shape, "q_target", q_target.shape)

        q_val = self.critic(S, A)  # (B, 1)
        critic_loss = F.mse_loss(q_val, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # —— Actor 更新 ——
        a_pred = self.actor(S)  # (B, act_dim)
        actor_loss = -self.critic(S, a_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # —— 软更新 target 网络 ——
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

    def update(self,
               transition: dict,
               agent_idx: int):
        """
        transition 中的每个 tensor shape：
          'states'      : (B, N, C, H, W)
          'actions'     : (B, N, nb_actions)
          'rewards'     : (B, N)           （每个 agent 的即时 scalar reward）
          'next_states' : (B, N, C, H, W)
          'dones'       : (B, N)           （bool 或 0/1）
        agent_idx: 要更新第几个 agent
        """
        # —— 准备数据 ——
        S  = torch.tensor(transition['states'],      dtype=torch.float32, device=self.device) / 255.0  # (B,N,C,H,W)
        A  = torch.tensor(transition['actions'],     dtype=torch.float32, device=self.device)       # (B,N,act_dim)
        R  = torch.tensor(transition['rewards'][:,agent_idx], dtype=torch.float32, device=self.device)  # (B,1)
        S2 = torch.tensor(transition['next_states'], dtype=torch.float32, device=self.device) / 255.0
        D  = torch.tensor(transition['dones'][:,agent_idx],   dtype=torch.float32, device=self.device)  # (B,1)

        B, N, *_ = S.shape

        # —— Critic 更新（中心化）——
        with torch.no_grad():
            # 先算出 next_actions 全体 agents 的动作
            next_as = []
            for i in range(N):
                obs_i = S2[:,i]                   # (B,C,H,W)
                next_as.append( self.target_actor(obs_i) )
            next_as = torch.stack(next_as, dim=1)  # (B,N,act_dim)

            q_next = self.target_critic(S2, next_as)  # (B,1)

        q_target = R + self.gamma * q_next * (1 - D)
        q_val    = self.critic(S, A)               # (B,1)
        q_target = q_target.view(q_val.shape)
        critic_loss = F.mse_loss(q_val, q_target)
        # assert q_val.shape == q_target.shape, f"Shape mismatch: q_val {q_val.shape}, q_target {q_target.shape}"

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()


        # —— Actor 更新（去中心化）——
        # 只更新自己的策略 πᵢ
        # 1) 用当前 actor 计算第 i 个 agent 的 aᵢ
        as_pred = []
        for j in range(N):
            if j == agent_idx:
                obs_j = S[:,j]                   # (B,C,H,W)
                a_j   = self.actor(obs_j)        # (B,act_dim)
            else:
                # 其它 agent 的动作固定为经验里那条
                a_j = A[:,j]
            as_pred.append(a_j)
        as_pred = torch.stack(as_pred, dim=1)      # (B,N,act_dim)

        # 2) 中心化 critic 给出 Q(S, a₁…aᵢ_pred…a_N)
        actor_loss = - self.critic(S, as_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        # —— 软更新 target 网络 ——
        self.soft_update(self.actor,  self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

# class Agent:
#     def __init__(self, nb_actions, action_space_bound, action_space_boundMove,
#                  actor_lr, critic_lr, tau, gamma, device):
#         """ 训练网络（真实）-- CNN 策略 价值（评价）"""
#         self.cnn = CNN().to(device)
#         self.critic = CriticNet(nb_actions, self.cnn).to(device)
#         self.actor = PolicyNet(nb_actions, action_space_bound.to(device),
#                                action_space_boundMove.to(device), self.cnn).to(device)
#
#         """ 目标网络（预测）-- CNN 策略 价值（评价）"""
#         self.target_cnn = CNN().to(device)
#         self.target_critic = CriticNet(nb_actions, self.target_cnn).to(device)
#         self.target_actor = PolicyNet(nb_actions, action_space_bound.to(device),
#                                       action_space_boundMove.to(device), self.target_cnn).to(device)
#
#         """ 使训练网络和目标网络的参数相同 """
#         self.target_cnn.load_state_dict(self.cnn.state_dict())
#         self.target_critic.load_state_dict(self.critic.state_dict())
#         self.target_actor.load_state_dict(self.actor.state_dict())
#
#         """ 生成优化器 """
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
#
#         """ 属性分配 """
#         self.gamma = gamma  # 折扣因子
#         # self.sigma = sigma  # 高斯声标准差，均值设为0
#         self.tau = tau  # 目标网络软更新参数
#         self.n_actions = nb_actions  # 动作空间维度
#         self.device = device
#
#     def take_action(self, state):
#         state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
#         action = self.actor(state / 255).cpu().detach().numpy()
#         # action = action + self.sigma * np.random.randn(self.n_actions)
#         return action
#
#     def soft_update(self, net, target_net):
#         for param_target, param in zip(target_net.parameters(), net.parameters()):
#             param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)
#
#     def update(self, i, transition_dict):  # 当多个智能体一起时，第i个智能体更新
#         states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
#         states = states / 255
#         actions = torch.tensor(transition_dict['actions'][:, i], dtype=torch.float32).to(self.device)
#         rewards = torch.tensor(transition_dict['rewards'][:, i], dtype=torch.float32).to(self.device)
#         next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
#         next_states = next_states / 255
#         dones = torch.tensor(transition_dict['dones'][:, i], dtype=torch.float32).to(self.device)
#         Dones = torch.tensor(transition_dict['Dones'], dtype=torch.float32).view(-1, 1).to(self.device)
#
#         next_actions = self.target_actor(next_states)
#         next_q_values = self.target_critic(next_states, next_actions)
#         """
#         单条路结束了，就返回单条路结束的reward值，且此时没有之后的预测期望
#         若整体结束了，也没有之后的预测期望
#         """
#         q_targets = rewards + self.gamma * next_q_values * (1 - dones) * (1 - Dones)  # 目标值
#         q_values = self.critic(states, actions)  # 预测值
#         critic_loss = torch.mean(nn.functional.mse_loss(q_values, q_targets))  # 预测值和目标值之间的均方差损失
#
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
#
#         actor_actions = self.actor(states)
#         score = self.critic(states, actor_actions)
#         actor_loss = -torch.mean(score)
#
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
#
#         self.soft_update(self.cnn, self.target_cnn)
#         self.soft_update(self.actor, self.target_actor)
#         self.soft_update(self.critic, self.target_critic)
#
#         return critic_loss, actor_loss
