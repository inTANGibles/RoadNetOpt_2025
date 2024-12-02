import collections
import gzip
import os.path
import pickle
import random

import numpy as np
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)

    # 在队列中添加数据
    def add(self, state, action, reward, next_state, done, Done):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state, done, Done))

    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state, done, Done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), \
            np.array(done), Done

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)

    def save(self, path: str, meta_data: dict = None):
        assert path.endswith('.pkl.gz')

        data_to_save = {'buffer': self.buffer, 'meta_data': meta_data}
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with gzip.open(path, "wb") as f:
            print(f'writing data, please wait...')
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        assert path.endswith('.pkl.gz')
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
            buffer: 'collections.deque' = data['buffer']
            meta_data: dict = data['meta_data']
        for i in range(len(buffer)):
            self.buffer.append(buffer[i])

