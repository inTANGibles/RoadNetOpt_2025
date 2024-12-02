import copy
import datetime
import logging
import math
import os
import random
from argparse import Namespace
from collections import deque
from enum import Enum
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from DDPG.ReplayBuffer import ReplayBuffer
from DDPG.agent import Agent
from DDPG.env import RoadEnv
from DDPG.utils.my_timer import MyTimer
from DDPG.utils.reward_utils import *
from geo import Road, Building, Region
from utils import io_utils

__version__ = 312
logging.basicConfig(level=logging.INFO)

'''提醒事项：
 - 这里给定的数值和参数名称会被gui解析并显示在窗口中
 - 以下args给定的默认数值，如果是float的话一定要写成float的形式，例如0.0，不要写成0，否则会被gui解析为int
'''
DEFAULT_ENV_ARGS = Namespace(
    data_path=r'../data/VirtualEnv/only_roads.bin',  # 地图数据文件， bin格式。如果留空，则使用已经加载的数据
    num_agents=1,  # 智能体数量
    region_min=None,  # None表示自动计算范围
    region_max=None,
    observation_img_size=(128, 128),
    observation_view_size=(400.0, 400.0),
    observation_center=(200.0, 200.0),  # 摄像机视角的中心点坐标
    still_mode=False,  # 是否固定摄像机视角
    action_step_range=(15, 40),#(20,60)
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
    sigma=1.000,  # 自由探索初始值
    sigma_decay=0.9995,  # 自由探索的衰减值
    actor_lr=1e-4,  #
    critic_lr=1e-3,  #
    tau=0.001,  #
    gamma=0.9,  #
    batch_size=64,  #
    buffer_capacity=10000,  # 经验空间最大容量
    save_interval=10000,  # 保存间隔
    log_folder='',  # 保存pth的文件夹， 留空则将在logs文件夹自动生成一个当前时间的文件夹用于保存
    save_warmup_buffer=False  # 在warmup完成后保存replay buffer
)

DEFAULT_GUI_ARGS = Namespace(
    headless=False  # 无头模式， 默认为True， 要在GUI中显示，请改为False
)

_shared_data: Optional[dict] = None
_sigma = 1.0
_return_queue = deque(maxlen=10)  # 这里的长度是用来平均的长度
_mean_return_arr: Optional[np.ndarray] = None  # 在train中根据需要的长度创建
_critic_loss_queue = deque(maxlen=10)  # 这里的长度是用来平均的长度
_mean_critic_loss_arr: Optional[np.ndarray] = None  # 在train中根据需要的长度创建
_actor_loss_queue = deque(maxlen=10)
_mean_actor_loss_arr: Optional[np.ndarray] = None


class TrainMode(Enum):
    WARMUP = 0
    TRAIN = 1


def train(env_args=DEFAULT_ENV_ARGS,
          train_args=DEFAULT_TRAIN_ARGS,
          gui_args=DEFAULT_GUI_ARGS,
          shared_data=None):
    """
    :param env_args:
    :param train_args:
    :param gui_args:
    :param shared_data: 共享信息
    :return:
    """
    # init global data
    global _shared_data, _sigma
    global _return_queue, _mean_return_arr
    global _critic_loss_queue, _mean_critic_loss_arr
    global _actor_loss_queue, _mean_actor_loss_arr
    _shared_data = shared_data
    _sigma = train_args.sigma
    _return_queue.clear()
    _critic_loss_queue.clear()
    _actor_loss_queue.clear()
    _mean_return_arr = np.zeros(train_args.train_epochs, dtype=np.float32)
    _mean_critic_loss_arr = np.zeros(train_args.train_epochs, dtype=np.float32)
    _mean_actor_loss_arr = np.zeros(train_args.train_epochs, dtype=np.float32)

    # init shared data
    if _shared_data is None:
        _shared_data = {'slow_ratio': 1,  # 放慢速度
                        'save_buffer': False  # 设为True时保存当前replay buffer， 随即设为False
                        }
    # _shared_data 中一定有 slow_ratio
    if 'slow_ratio' not in _shared_data:
        _shared_data['slow_ratio'] = 1
    # _shared_data中一定有 save_buffer
    if 'save_buffer' not in _shared_data:
        _shared_data['save_buffer'] = False

    MyTimer.register_shared_data(_shared_data)

    # init map data
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
                      still_mode=env_args.still_mode,
                      observation_center=env_args.observation_center,
                      action_step_range=env_args.action_step_range,
                      headless=gui_args.headless,
                      shared_data=_shared_data,
                      )
        yield None
    # create agent
    with MyTimer('init_agent_time', level=0):
        action_space_bound = torch.tensor(env.action_space_bound)
        action_space_boundMove = torch.tensor(env.action_space_boundMove)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'creating replay buffer...')
        replay_buffer: ReplayBuffer = ReplayBuffer(capacity=train_args.buffer_capacity)
        reward_info_buffer: deque = deque(maxlen=train_args.buffer_capacity)

        agent: Agent = Agent(nb_actions=train_args.nb_actions,
                             action_space_bound=action_space_bound,
                             action_space_boundMove=action_space_boundMove,
                             actor_lr=train_args.actor_lr,
                             critic_lr=train_args.critic_lr,
                             tau=train_args.tau,
                             gamma=train_args.gamma,
                             device=device)
        yield None

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
    _shared_data['log_folder'] = train_args.log_folder
    print(f'log files will be saved to {os.path.abspath(train_args.log_folder)}')

    # region [🤖warm_up]
    with MyTimer('warmup_time', level=0):
        _shared_data['stage'] = 'warmup'
        for epoch in tqdm(range(train_args.warmup_epochs)):  # 先玩n轮
            with MyTimer('epoch_time', level=1):
                gen = epoch_generator(epoch, TrainMode.WARMUP, env, agent, train_args, env_args, gui_args, replay_buffer,
                                      reward_info_buffer)
                try:
                    while True:
                        next(gen)
                        yield None
                except StopIteration:
                    pass
    # save replay buffer
    if train_args.save_warmup_buffer:
        print(f'saving warmup_buffer...')
        save_replay_buffer(replay_buffer, env_args, train_args, gui_args, reward_info_buffer)
    # endregion
    # region [🚀train]
    with MyTimer('train_time', level=0):
        _shared_data['stage'] = 'train'
        for epoch in tqdm(range(train_args.train_epochs)):
            with MyTimer('epoch_time', level=1):
                gen = epoch_generator(epoch, TrainMode.TRAIN, env, agent, train_args,env_args, gui_args, replay_buffer,
                                      reward_info_buffer)
                try:
                    while True:
                        next(gen)
                        yield None
                except StopIteration:
                    pass
            save_model(epoch, train_args, agent)
            # endregion
    # endregion
    print(f'😊complete')


def epoch_generator(epoch: int, mode: TrainMode, env: RoadEnv, agent: Agent, train_args: Namespace, env_args, gui_args,
                    replay_buffer: ReplayBuffer, reward_info_buffer: deque):
    """
    每轮epoch运行的函数, 对于warmup和train都是一样的，通过设置mode来区分

    :param epoch:现在是第几轮
    :param mode:训练模式TrainMode.TRAIN  或 TrainMode.WARMUP
    :param env:
    :param agent:
    :param train_args:
    :param replay_buffer:
    :param reward_info_buffer:
    :return:
    """
    global _sigma
    # region [warm_up/train episode]
    _shared_data['epoch'] = epoch
    with MyTimer('env_reset_time', level=2):
        state = env.reset()
    done = env.done_in_ndarray_format
    # play game
    with MyTimer('total_play_time', level=2):

        buffer_list: list[list] = []
        """
        buffer_list 是一个作用域范围为每一轮epoch的变量，
        
        其用于临时记录每一轮智能体的replay buffer信息，收集完一整轮的buffer信息后经过reward偏移再交由replay_buffer保存。
        
        智能体在运行的期间会动态给buffer_list添加值。
        
        buffer_list的结构为 step_idx: [state, action, reward, next_state, done, Done] : ...
        
        """

        reward_info_list: list[dict[int:dict[int:float]]] = []
        """
        reward_info_list是一个作用域范围为每一轮epoch的变量，

        其记录了每一个step中，每个agent的各项reward的值， 具体的reward名称由reward_agent.py文件定义。

        智能体在运行的期间会动态给reward_info_list添加值。

        reward_info_list的一般结构为 int step_idx: int agent_idx: int reward_name: float reward_value
        [
            step_idx : dict{
                agent_idx : dict{
                    reward_name: reward_value,
                    ...
                    sum: reward_value,  # 特殊项
                    final_reward_name: final_reward_value,
                    ...
                    final_reward_shift: value,  # 特殊项
                    final_reward_weight: value,  # 特殊项
                    }
                }
                ...
            }
            ...
        ]
        """

        gen = agent_play(epoch, env, mode, agent, done, state, train_args, replay_buffer, buffer_list, reward_info_list)
        """
        定义一个agent play的generator
        
        使用 next(gen)来调用该迭代器， 让agent走一步， 当agent没有下一步时，会抛出 StopIteration Exception
        
        其中参数中的buffer_list与reward_info_list会在agent play的过程中动态更新
        """
        try:
            while True:
                next(gen)
                yield None
        except StopIteration:
            pass
    with MyTimer('final_reward_time', level=2):
        final_reward = env.calculate_final_reward()  # 结束一局游戏后，计算最终的reward
    final_reward_info: dict[int:dict[str:float]] = _shared_data['final_reward_info']  # 拿到具体的final reward info
    """
    final_reward_info保存了final reward的具体每一项的值
    
    env.calculate_final_reward()会更新shared_data的'final_reward_info'项，
    
    final_reward_info这个key的名称是由env定义的
    
    final_reward_info的结构为 agent_idx: reward_name: reward_value
    """
    shift_reward(final_reward, final_reward_info,buffer_list, reward_info_list)  # 偏移原有的buffer中的reward信息
    add_to_replay_buffer(buffer_list, replay_buffer)  # 将偏移后的buffer数据添加到replay_buffer中
    add_to_reward_info_buffer(reward_info_list, reward_info_buffer)  # 将reward info数据添加到reward_info_buffer中
    if mode == TrainMode.TRAIN:
        _sigma *= train_args.sigma_decay  # 降低sigma
        _shared_data['sigma'] = _sigma
        _return_queue.append(calculate_episode_return(buffer_list))  # 将这一轮的reward和添加到return_list
        _mean_return_arr[epoch] = sum(_return_queue) / len(_return_queue)
        n = epoch + 1
        if n < 100:
            _shared_data['mean_return_list'] = _mean_return_arr[:n]
            _shared_data['mean_critic_loss'] = _mean_critic_loss_arr[:n]
            _shared_data['mean_actor_loss'] = _mean_actor_loss_arr[:n]
        else:
            sample_size = 100
            step = (n - 1) // (sample_size - 1)  # 计算平均间隔（除了最后一个元素）
            sampled_mean_return_arr = _mean_return_arr[:n:step]
            sampled_critic_loss_arr = _mean_critic_loss_arr[:n:step]
            sampled_actor_loss_arr = _mean_actor_loss_arr[:n:step]
            _shared_data['mean_return_list'] = np.append(sampled_mean_return_arr, _mean_return_arr[epoch])
            _shared_data['mean_critic_loss'] = np.append(sampled_critic_loss_arr, _mean_critic_loss_arr[epoch])
            _shared_data['mean_actor_loss'] = np.append(sampled_actor_loss_arr, _mean_actor_loss_arr[epoch])

    if _shared_data['save_buffer']:
        print(f'saving buffer...')
        save_replay_buffer(replay_buffer, env_args, train_args, gui_args, reward_info_buffer)
        _shared_data['save_buffer'] = False

    with MyTimer('yield_time', level=2):
        for i in range(_shared_data['slow_ratio']):
            yield None
    # endregion


def agent_play(epoch, env, mode, agent, done, state, train_args, replay_buffer, out_data_list: list,
               out_reward_info_list: list[dict[int:dict[str:float]]]):
    out_data_list.clear()
    out_reward_info_list.clear()
    critic_loss = 0
    actor_loss = 0
    while True:
        # region [agent step]
        # take action
        with MyTimer('agent_action_time', level=3):
            action = agent_take_action(env, done, agent, state, train_args)
        # env step
        with MyTimer('env_step_time', level=3):
            next_state, reward, done, Done = env.step(action)
        # add to replay buffer  (out_data_buffer)
        out_data_list.append([state, action, reward, next_state, done, Done])
        reward_info: dict[int:dict[str:float]] = _shared_data['reward_info']  # agent_idx: reward_name : reward_value
        out_reward_info_list.append(copy.copy(reward_info))  # 这里一定要注意用copy，不然加入的会是dict的地址，会随着dict的改变而改变
        state = next_state  # state changed, should be returned and updated

        # agent_update(mode, env, agent, replay_buffer, train_args)
        _shared_data['episode_step'] = env.episode_step

        if mode == TrainMode.TRAIN:
            # update agent
            critic_loss, actor_loss = agent_update(env, agent, replay_buffer, train_args)  # 随机采样batch size个数据喂给agent学习

        if Done: break

        with MyTimer('yield_time', level=3):
            for i in range(_shared_data['slow_ratio']):
                yield None

        # endregion
    _critic_loss_queue.append(critic_loss)
    _actor_loss_queue.append(actor_loss)
    _mean_critic_loss_arr[epoch] = sum(_critic_loss_queue) / len(_critic_loss_queue)
    _mean_actor_loss_arr[epoch] = sum(_actor_loss_queue) / len(_actor_loss_queue)


def agent_update(env, agent, replay_buffer, train_args):
    if replay_buffer.size() <= train_args.batch_size:
        return 0, 0
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
    return critic_loss, actor_loss


def agent_take_action(env, done, agent, state, train_args):
    action_list = []
    for i in range(env.num_road_agents):
        if done[i]:
            action = np.zeros((1, 2))
        else:
            action = agent.take_action(state)
            noise = np.random.randn(train_args.nb_actions)  # 后面可能换成ou噪声，增加噪声强度逐渐衰减的功能
            # alpha = random.uniform(0, _sigma)
            action[0][0] = (action[0][0] + noise[0] * _sigma + 3) % 2.0 - 1  # -1:1 -> -2:2 -> 1:5 -> 0:2 -> -1:1
            action[0][1] = (action[0][1] + noise[1] * _sigma + 3) % 2.0 - 1
        action_list.append(action)
    action = np.array(action_list).reshape(-1, 2)
    _shared_data['action_0'] = action[0]
    return action


def shift_reward(final_reward: np.ndarray,
                 final_reward_info: dict[int:dict[str:float]],
                 buffer_list: list[list],
                 reward_info_list: list[dict[int:dict[str, float]]],
                 ):
    """
    偏移原有的buffer中的reward信息

    其中buffer_list与reward_info_list中的内容会被动态修改，不需要返回

    示例：

    # >>> done:bool    = buffer_list[step: int][4][agent_index: int][0]

    # >>> reward:float = buffer_list[step: int][2][agent_index: int][0]

    :param final_reward: 最终的reward， 形状为(n, 1)， n表示n个智能体
    :param buffer_list: 每一轮epoch的临时buffer_list,
    其内容格式为 step_idx : [state, action, reward, next_state, done, Done] : ...
    :param reward_info_list:其内容格式为 step_idx: agent_idx: reward_name: reward_value
    :param final_reward_info: 其内容格式为 agent_idx: reward_name: reward_value
    :return:
    """

    num_steps = len(buffer_list)
    rewards_per_agent = []  # [agent_index][step]
    for agent_idx in range(len(final_reward)):
        # for each agent
        rewards = [buffer_list[step][2][agent_idx][0] for step in range(num_steps)]
        dones = [int(buffer_list[step][4][agent_idx][0]) for step in range(num_steps)]
        reward_shift = final_reward[agent_idx][0]

        first_done_step = len(dones) - sum(dones)
        x = 0  # x is the distance from last step
        for step in range(first_done_step, -1, -1):
            # from last step to first step
            weight = assign_reward_weights(x)  # calculate weight
            shift_value = weight * reward_shift
            # shift rewards
            rewards[step] = rewards[step] + shift_value

            # add final_reward_info to reward_info_list
            reward_info_list[step][agent_idx][FINAL_REWARD_SHIFT] = shift_value
            reward_info_list[step][agent_idx][FINAL_REWARD_WEIGHT] = weight
            for reward_name, reward_value in final_reward_info[agent_idx].items():
                reward_info_list[step][agent_idx][reward_name] = reward_value * weight
            x += 1
        rewards_per_agent.append(rewards)

    rewards_per_agent = np.array(rewards_per_agent)
    rewards_per_step = np.transpose(rewards_per_agent)
    for step in range(num_steps):
        buffer_list[step][2] = rewards_per_step[step].reshape((-1, 1))


def assign_reward_weights(x):
    """
    reward权重分配方法
    :param x: 离最后一步的距离。 最后一步x=0， 离最后一步越近x越小
    :return: reward分配权重
    """
    return 1.0 / (x + 1.0)  # 1/x+1


def calculate_episode_return(data_buffer):
    # (epoch, agent, 1)
    rewards_arr: np.ndarray = np.array([data_buffer[epoch][2] for epoch in range(len(data_buffer))])
    return np.sum(rewards_arr)


def add_to_replay_buffer(buffer_list: list[list], replay_buffer: ReplayBuffer):
    for item in buffer_list:
        replay_buffer.add(*item)


def add_to_reward_info_buffer(reward_info_list, reward_info_buffer):
    for item in reward_info_list:
        reward_info_buffer.append(item)


def save_model(epoch, train_args, agent):
    if epoch % train_args.save_interval != 0:
        return
    if epoch <= 0:
        return
    with MyTimer('save_time', level=1):
        if not os.path.exists(train_args.log_folder):
            os.makedirs(train_args.log_folder)
        torch.save(agent.actor.state_dict(),
                   os.path.join(train_args.log_folder, f'actor_{epoch}.pth'))
        torch.save(agent.critic.state_dict(),
                   os.path.join(train_args.log_folder, f'critic_{epoch}.pth'))
        torch.save(agent.cnn.state_dict(),
                   os.path.join(train_args.log_folder, f'cnn_{epoch}.pth'))


def save_replay_buffer(replay_buffer: ReplayBuffer, env_args, train_args, gui_args, reward_info_buffer):
    import getpass
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    meta_data = {
        'env_args': env_args,
        'train_args': train_args,
        'gui_args': gui_args,
        'reward_info_buffer': reward_info_buffer,
        'user_name': getpass.getuser(),
        'save_time': time_str,
        'train_version': __version__
    }
    path = f'logs/replay_buffer_{replay_buffer.size()}_{time_str}.pkl.gz'
    replay_buffer.save(path, meta_data=meta_data)
    print(f'replay buffer saved to {path}')
