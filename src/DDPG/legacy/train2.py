import time
import os
from argparse import Namespace
from collections import deque
from typing import Optional

import numpy as np
import torch
import datetime
import logging
from tqdm import tqdm

from DDPG.env import RoadEnv
from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.agent import Agent
from utils import io_utils
from geo import Road, Building, Region


logging.basicConfig(level=logging.INFO)

DEFAULT_ENV_ARGS = Namespace(
    data_path=r'../data/VirtualEnv/0312_ty.bin',  # 地图数据文件， bin格式。如果留空，则使用已经加载的数据
    num_agents=1,  # 智能体数量
    region_min=(-40, -50),
    region_max=(500, 500),
    observation_img_size=(128, 128),
    observation_view_size=(400.0, 400.0),
    action_step_range=(20, 60),
    max_episode_step=50,  # 每个智能体最多走多少步
)

DEFAULT_TRAIN_ARGS = Namespace(
    path_actor='',  # 缓存的actor的pth文件
    path_critic='',  # 缓存的critic的pth文件
    path_cnn='',  # 缓存的cnn的pth文件
    path_buffer='',  # 缓存的buffer文件
    nb_actions=2,  # action的维度
    warmup_epochs=10000,  # 预热轮次
    train_epochs=10001,  # 训练轮次
    sigma=1.0,  # 自由探索初始值
    sigma_decay=0.9995,  # 自由探索的衰减值
    actor_lr=1e-4,  #
    critic_lr=1e-3,  #
    tau=0.001,  #
    gamma=0.9,  #
    batch_size=64,  #
    buffer_capacity=10000,  # 经验空间最大容量
    save_interval=10000,  # 保存间隔
    log_folder='',  # 保存pth的文件夹， 留空则将在logs文件夹自动生成一个当前时间的文件夹用于保存
    save_warmup_buffer=False,  # 在warmup完成后保存replay buffer
    update_interval=5
)

DEFAULT_GUI_ARGS = Namespace(
    headless=False  # 无头模式， 默认为True， 要在GUI中显示，请改为False
)


def train(env_args=DEFAULT_ENV_ARGS,
          train_args=DEFAULT_TRAIN_ARGS,
          gui_args=DEFAULT_GUI_ARGS,
          shared_data=None):
    """
    :param env_args:
    :param train_args:
    :param gui_args:
    :param shared_data: 共享信息 其中：slow_ratio: 1表示使用正常的yield， 0表示不使用yield（UI不会更新）， 数字越大，等待时间越长
    :return:
    """
    if shared_data is None:
        shared_data = {'slow_ratio': 1}
    elif 'slow_ratio' not in shared_data:
        shared_data['slow_ratio'] = 1

    class MyTimer:
        def __init__(self, name: str, level: int):
            self.name = name
            self.level = level

        def __enter__(self):
            self.start_time = time.time()

            if 'time' not in shared_data:
                shared_data['time'] = {}
            if self.level not in shared_data['time']:
                shared_data['time'][self.level] = {}
            if self.name not in shared_data['time'][self.level]:
                shared_data['time'][self.level][self.name] = None

        def __exit__(self, exc_type, exc_value, traceback):
            shared_data['time'][self.level][self.name] = time.time() - self.start_time

    class LimitedQueue(deque):
        def __init__(self, max_len=10):
            super().__init__(maxlen=max_len)

        def append(self, item):
            if len(self) == self.maxlen:
                self.popleft()
            super().append(item)

    # init data
    with MyTimer('init_data_time', level=0):
        if env_args.data_path != '':
            data = io_utils.load_data(env_args.data_path)
            Building.data_to_buildings(data)
            Region.data_to_regions(data)
            Road.data_to_roads(data)
        yield None
    # create env
    with MyTimer('init_env_time', level=0):
        print(f'🎯creating env...')
        env = RoadEnv(num_road_agents=env_args.num_agents,
                      max_episode_step=env_args.max_episode_step,
                      region_min=env_args.region_min,
                      region_max=env_args.region_max,
                      observation_img_size=env_args.observation_img_size,
                      observation_view_size=env_args.observation_view_size,
                      action_step_range=env_args.action_step_range,
                      headless=gui_args.headless,
                      shared_data=shared_data)
        yield None
    # create agent
    with MyTimer('init_agent_time', level=0):
        action_space_bound = torch.tensor(env.action_space_bound)
        action_space_boundMove = torch.tensor(env.action_space_boundMove)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'creating replay buffer...')
        replay_buffer: ReplayBuffer = ReplayBuffer(capacity=train_args.buffer_capacity)
        agent: Agent = Agent(nb_actions=train_args.nb_actions,
                             action_space_bound=action_space_bound,
                             action_space_boundMove=action_space_boundMove,
                             actor_lr=train_args.actor_lr,
                             critic_lr=train_args.critic_lr,
                             tau=train_args.tau,
                             gamma=train_args.gamma,
                             device=device)
        yield None
    sigma = train_args.sigma

    _return_queue = LimitedQueue(max_len=10)  # 这里的长度是用来平均的长度
    _mean_return_arr = np.zeros(train_args.train_epochs, dtype=np.float32)
    _critic_loss_queue = LimitedQueue(max_len=10)  # 这里的长度是用来平均的长度
    _mean_critic_loss_arr = np.zeros(train_args.train_epochs, dtype=np.float32)
    _actor_loss_queue = LimitedQueue(max_len=10)
    _mean_actor_loss_arr = np.zeros(train_args.train_epochs, dtype=np.float32)

    # load state
    with MyTimer('load_state_time', level=0):
        if train_args.path_actor != '':
            agent.actor.load_state_dict(torch.load(train_args.path_actor))
        if train_args.path_critic != '':
            agent.critic.load_state_dict(torch.load(train_args.path_critic))
        if train_args.path_cnn != '':
            agent.cnn.load_state_dict(torch.load(train_args.path_cnn))
        if train_args.path_buffer != '':
            print(f'loading buffer from {train_args.path_buffer}')
            replay_buffer.load(train_args.path_buffer)

    if train_args.log_folder == '':
        train_args.log_folder = f'logs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    shared_data['log_folder'] = train_args.log_folder
    print(f'log files will be saved to {os.path.abspath(train_args.log_folder)}')

    _start_time_dict = {}

    # region [warm_up]
    with MyTimer('warmup_time', level=0):
        print('🤖warmup...')
        shared_data['stage'] = 'warmup'
        for episode in tqdm(range(train_args.warmup_epochs)):  # 先玩n轮
            # region [warm_up episode]
            with MyTimer('epoch_time', level=1):
                shared_data['epoch'] = episode
                with MyTimer('env_reset_time', level=2):
                    state = env.reset()
                done = env.done_in_ndarray_format
                # play game
                with MyTimer('total_play_time', level=2):
                    while True:
                        # region [agent step]
                        # take action
                        with MyTimer('agent_action_time', level=3):
                            action_list = []
                            for i in range(env.num_road_agents):
                                if done[i]:
                                    action = np.zeros((1, 2))
                                else:
                                    action = agent.take_action(state)
                                    noise = sigma * np.random.randn(train_args.nb_actions)  # 后面可能换成ou噪声，增加噪声强度逐渐衰减的功能
                                    action += noise
                                action_list.append(action)
                            action = np.array(action_list).reshape(-1, 2)
                        # env step
                        with MyTimer('env_step_time', level=3):
                            next_state, reward, done, Done = env.step(action)
                        # add to replay buffer
                        with MyTimer('replay_buffer_add_time', level=3):
                            replay_buffer.add(state, action, reward, next_state, done, Done)
                        state = next_state
                        if Done: break
                        for i in range(shared_data['slow_ratio']):
                            yield None
                        # endregion

                for i in range(shared_data['slow_ratio']):
                    yield None
            # endregion
    # endregion

    # save replay buffer
    if train_args.save_warmup_buffer:
        print(f'saving warmup buffer...')
        path = f'logs/replay_buffer_{replay_buffer.size()}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pkl.gz'
        replay_buffer.save(path)
        print(f'replay buffer saved to {path}')

    # region [train]
    with MyTimer('train_time', level=0):
        print('🚀training...')
        shared_data['stage'] = 'train'
        for train_episode in tqdm(range(train_args.train_epochs)):
            # region [train episode]
            with MyTimer('epoch_time', level=1):
                shared_data['epoch'] = train_episode
                with MyTimer('env_reset_time', level=2):
                    state = env.reset()
                done = env.done_in_ndarray_format
                episode_return = 0
                # play game
                with MyTimer('total_play_time', level=2):
                    while True:
                        # region [agent step]
                        # take action
                        with MyTimer('agent_action_time', level=3):
                            action_list = []
                            for i in range(0, env.num_road_agents):
                                if done[i]:
                                    action = np.zeros((1, 2))
                                else:
                                    action = agent.take_action(state)
                                    shared_data['action_0'] = action[0]
                                    noise = sigma * np.random.randn(train_args.nb_actions)  # 后面可能换成ou噪声，增加噪声强度逐渐衰减的功能
                                    action += noise
                                action_list.append(action)
                            action = np.array(action_list).reshape(-1, 2)

                        # env step
                        with MyTimer('env_step_time', level=3):
                            next_state, reward, done, Done = env.step(action)

                        # add to replay buffer
                        with MyTimer('replay_buffer_add_time', level=3):
                            replay_buffer.add(state, action, reward, next_state, done, Done)  #

                        state = next_state
                        episode_return += reward
                        # update agent
                        if replay_buffer.size() > train_args.batch_size:
                            # start update
                            with MyTimer('agent_update_time', level=3):
                                s, a, r, ns, d, D = replay_buffer.sample(train_args.batch_size)
                                transition_dict = {
                                    'states': s,
                                    'actions': a,
                                    'rewards': r,
                                    'next_states': ns,
                                    'dones': d,
                                    'Dones': D,
                                }
                                for i in range(0, env.num_road_agents):
                                    critic_loss, actor_loss = agent.update(i, transition_dict)
                            # end update
                        # region [events(each step)]
                        if not gui_args.headless:
                            shared_data['episode_step'] = env.episode_step
                            shared_data['action'] = action  # (num_agents, num_actions)
                            shared_data['reward'] = reward  # (num_agents, 1)
                            shared_data['episode_return'] = episode_return
                        # endregion
                        if Done: break
                        with MyTimer('yield_time', level=3):
                            for i in range(shared_data['slow_ratio']):
                                yield None
                        # endregion
                episode_return = np.mean(episode_return)
                sigma *= train_args.sigma_decay
                _return_queue.append(episode_return)  # 将这一轮的reward和添加到return_list
                _critic_loss_queue.append(critic_loss)
                _actor_loss_queue.append(actor_loss)
                _mean_return_arr[train_episode] = sum(_return_queue) / len(_return_queue)
                _mean_critic_loss_arr[train_episode] = sum(_critic_loss_queue) / len(_critic_loss_queue)
                _mean_actor_loss_arr[train_episode] = sum(_actor_loss_queue) / len(_actor_loss_queue)
                # region [events (by epoch)]
                shared_data['sigma'] = sigma
                n = train_episode + 1
                if n < 100:
                    shared_data['mean_return_list'] = _mean_return_arr[:n]
                    shared_data['mean_critic_loss'] = _mean_critic_loss_arr[:n]
                    shared_data['mean_actor_loss'] = _mean_actor_loss_arr[:n]
                else:
                    sample_size = 100
                    step = (n - 1) // (sample_size - 1)  # 计算平均间隔（除了最后一个元素）
                    sampled_mean_return_arr = _mean_return_arr[:n:step]
                    sampled_critic_loss_arr = _mean_critic_loss_arr[:n:step]
                    sampled_actor_loss_arr = _mean_actor_loss_arr[:n:step]
                    shared_data['mean_return_list'] = np.append(sampled_mean_return_arr, _mean_return_arr[train_episode])
                    shared_data['mean_critic_loss'] = np.append(sampled_critic_loss_arr, _mean_critic_loss_arr[train_episode])
                    shared_data['mean_actor_loss'] = np.append(sampled_actor_loss_arr, _mean_actor_loss_arr[train_episode])
                # endregion

                with MyTimer('yield_time', level=2):
                    for i in range(shared_data['slow_ratio']):
                        yield None
            # endregion
            # save state dict
            if train_episode % train_args.save_interval == 0 and train_episode > 0:
                with MyTimer('save_time', level=1):
                    if not os.path.exists(train_args.log_folder):
                        os.makedirs(train_args.log_folder)
                    torch.save(agent.actor.state_dict(),
                               os.path.join(train_args.log_folder, f'actor_{train_episode}.pth'))
                    torch.save(agent.critic.state_dict(),
                               os.path.join(train_args.log_folder, f'critic_{train_episode}.pth'))
                    torch.save(agent.cnn.state_dict(), os.path.join(train_args.log_folder, f'cnn_{train_episode}.pth'))
            # endregion
    # endregion
    print(f'😊complete')
