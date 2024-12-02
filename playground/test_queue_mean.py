import collections
import random

import matplotlib.pyplot as plt
from tqdm import tqdm

# 一些设置参数
EPOCH = 100000  # 测试轮数
ADD_PERCENTAGE = 0.7  # 当reward小于平均时，添加到queue的概率
QUEUE_CAPACITY = 1000  # 手动设置queue的容量


def reward_generator():
    return random.random() ** 2  # 自定义一个0的概率大，1的概率小的随机数生成器


def add_reward_to_queue(_reward, _queue, _curr_avg):
    """
    添加reward到queue中
    :param _reward: 要添加的reward
    :param _queue: 要添加的queue
    :param _curr_avg: 这个queue当前的平均值
    :return: 新的平均值
    """
    _sum = len(_queue) * _curr_avg  # 添加前的queue的总量
    _queue.append(_reward)  # 添加到queue
    _sum += _reward  # 更新总量
    if len(_queue) > QUEUE_CAPACITY:  # 判断是否大于最大容量
        _first_number = _queue.popleft()  # 如果大于最大容量，则删除最左边的数
        _sum -= _first_number  # 更新总量
    new_avg = _sum / len(_queue)  # 计算新的平均值
    return new_avg


if __name__ == '__main__':
    # 对照组，没有实行筛选的deque
    data_queue_before = collections.deque()
    curr_avg_before = 0

    # 实行筛选的deque
    data_queue_after = collections.deque()
    curr_avg_after = 0

    # 画图用的数据
    reward_history = [0] * EPOCH
    avg_history_before = [0] * EPOCH
    num_numbers_in_queue_history_before = [0] * EPOCH
    avg_history_after = [0] * EPOCH
    num_numbers_in_queue_history_after = [0] * EPOCH

    for i in tqdm(range(EPOCH)):
        reward = reward_generator()  # 生成一个随机reward

        # 添加reward到queue中（没有实行筛选）
        curr_avg_before = add_reward_to_queue(reward, data_queue_before, curr_avg_before)

        # 筛选是否要加入queue
        if reward > curr_avg_after or random.random() < ADD_PERCENTAGE:
            # 添加reward到queue中（实行了筛选）
            curr_avg_after = add_reward_to_queue(reward, data_queue_after, curr_avg_after)

        # 统计数据
        reward_history[i] = reward

        avg_history_before[i] = curr_avg_before
        num_numbers_in_queue_history_before[i] = len(data_queue_before)
        avg_history_after[i] = curr_avg_after
        num_numbers_in_queue_history_after[i] = len(data_queue_after)

    # 使用 Matplotlib 绘制图表
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 绘制第一个子图
    x = range(EPOCH)  # 横坐标为列表的索引
    y1 = num_numbers_in_queue_history_before  # 纵坐标为列表中的数据
    y2 = num_numbers_in_queue_history_after

    axs[0].plot(x, y1, color='black', linestyle='-', label='Num Numbers in Queue (Before)')  # 绘制 num_numbers_in_queue_history 折线
    axs[0].plot(x, y2, color='b', linestyle='-', label='Num Numbers in Queue (After)')  # 绘制 num_numbers_in_queue_history 折线
    axs[0].set_title('Num Numbers in Queue')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Data Value')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制第二个子图
    y1 = avg_history_before  # 纵坐标为列表中的数据
    y2 = avg_history_after
    axs[1].plot(x, y1, color='black', linestyle='-', label='Average Reward (Before)')  # 绘制 avg_history 折线
    axs[1].plot(x, y2, color='b', linestyle='-', label='Average Reward (After)')  # 绘制 avg_history 折线
    axs[1].set_title('Average Reward')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Data Value')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylim(0, 1)  # 设置 y 坐标范围为 0 到 1

    # 绘制第三个子图
    axs[2].hist(reward_history, bins=10, color='skyblue', edgecolor='black')  # bins 参数指定柱状图的数量
    axs[2].set_title('Histogram of Reward')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(True)

    plt.suptitle(F'Data Analysis (EPOCHS={EPOCH}, ADD_PERCENTAGE={ADD_PERCENTAGE}, QUEUE_CAPACITY={QUEUE_CAPACITY})')
    # 显示图表
    plt.show()
