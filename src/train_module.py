"""
Train Module
====================
模型训练模块

"""
from typing import Optional
import DDPG.train as train
# import DDPG.legacy.train2 as train
import copy


class TrainManager:
    """
    Train Manager
    """
    I: Optional['TrainManager'] = None

    def __init__(self):
        TrainManager.I = self
        self._is_training = False
        self._train_coroutine = None

    @property
    def is_training(self):
        """是否在训练"""
        return self._is_training

    @property
    def is_stopped(self):
        """是否停止"""
        return not self._is_training and self._train_coroutine is None

    @property
    def is_paused(self):
        return not self._is_training and self._train_coroutine is not None

    @property
    def train_coroutine(self):
        return self._train_coroutine

    @property
    def default_env_args_copy(self):
        return copy.deepcopy(train.DEFAULT_ENV_ARGS)

    @property
    def default_train_args_copy(self):
        return copy.deepcopy(train.DEFAULT_TRAIN_ARGS)

    @property
    def default_gui_args_copy(self):
        return copy.deepcopy(train.DEFAULT_GUI_ARGS)

    def step_train(self):
        if not self._is_training:
            return
        if self._train_coroutine is None:
            return
        try:
            next(self._train_coroutine)
        except StopIteration:
            self.stop_train()

    def start_train(self, *args, **kwargs):
        self._is_training = True
        self._train_coroutine = train.train(*args, **kwargs)

    def stop_train(self):
        self._is_training = False
        self._train_coroutine = None

    def pause_train(self):
        if self._train_coroutine is None:
            return
        self._is_training = False

    def continue_train(self):
        if self._train_coroutine is None:
            return
        self._is_training = True

    def train(self):
        """headless模式下调用，阻塞版本"""
        self.start_train()
        while not self.is_stopped:
            self.step_train()


_ = TrainManager()
