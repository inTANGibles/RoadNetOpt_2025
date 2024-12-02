import os
import datetime
import numpy as np
import torch
from tqdm import tqdm
from utils import io_utils
from DDPG.legacy.env import RoadNet
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.agent import Agent



def train():
    log_folder = f'logs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    data = io_utils.load_data(r"../data/VirtualEnv/0414.bin")
    env = RoadNet(data)

    path_actor = ''
    path_critic = ''
    path_cnn = ''
    nb_actions = 2
    action_space_bound = torch.tensor(env.action_space_bound)
    action_space_boundMove = torch.tensor(env.action_space_boundMove)
    sigma = 1
    actor_lr = 1e-4
    critic_lr = 1e-3
    tau = 0.001
    gamma = 0.9
    batch_size = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    replay_buffer: ReplayBuffer = ReplayBuffer(capacity=1000)
    agent: Agent = Agent(nb_actions, action_space_bound, action_space_boundMove, actor_lr, critic_lr, tau, gamma,
                         device)

    return_list = []
    mean_return_list = []

    """ 具体的还没有细化， 画图部分没加入， 多线程并行没写， 模型权重没存， 但目前基于env可以运行， 但还没有仔细调试 """
    if path_actor != '':
        agent.actor.load_state_dict(torch.load(path_actor))
    if path_critic != '':
        agent.critic.load_state_dict(torch.load(path_critic))
    if path_cnn != '':
        agent.cnn.load_state_dict(torch.load(path_cnn))
    for episode in tqdm(range(200)):  # 先玩200轮
        state = env.reset()
        done = np.zeros((env.nb_new_roads, 1))
        while True:
            action_list = []
            for i in range(0, env.nb_new_roads):
                action = agent.take_action(state)
                noise = sigma * np.random.randn(nb_actions)  # 后面可能换成ou噪声，增加噪声强度逐渐衰减的功能
                action = action + noise
                if done[i]:
                    action = np.zeros((1, 2))
                action_list.append(action)
            action = np.array(action_list).reshape(-1, 2)
            next_state, reward, done, Done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done, Done)
            state = next_state
            if Done:
                break

    for train_episode in tqdm(range(10000)):
        state = env.reset()
        done = np.zeros((env.nb_new_roads, 1))
        episode_return = 0
        while True:
            action_list = []
            for i in range(0, env.nb_new_roads):
                action = agent.take_action(state)
                noise = sigma * np.random.randn(nb_actions)  # 后面可能换成ou噪声，增加噪声强度逐渐衰减的功能
                action = action + noise
                if done[i]:
                    action = np.zeros((1, 2))
                action_list.append(action)
            action = np.array(action_list).reshape(-1, 2)
            next_state, reward, done, Done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done, Done)
            state = next_state
            episode_return += reward

            if replay_buffer.size() > batch_size:
                s, a, r, ns, d, D = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'rewards': r,
                    'next_states': ns,
                    'dones': d,
                    'Dones': D,
                }
                for i in range(0, env.nb_new_roads):
                    agent.update(i, transition_dict)
            if Done:
                sigma *= 0.98
                break
        return_list.append(episode_return)
        mean_return_list.append(np.mean(return_list[-10:]))

        if train_episode % 100 == 0:
            print(f'iter:{train_episode}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}')

            torch.save(agent.actor.state_dict(), os.path.join(log_folder, f'actor_{train_episode}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(log_folder, f'critic_{train_episode}.pth'))
            torch.save(agent.cnn.state_dict(), os.path.join(log_folder, f'cnn_{train_episode}.pth'))