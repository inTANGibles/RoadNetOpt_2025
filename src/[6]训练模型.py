import os.path
import cv2
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.envs.registration import register
# from stable_baselines3 import TD3
# from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Image as SBImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from tqdm import tqdm

from geo import BuildingCollection, RegionCollection
from utils import headless_utils, io_utils

register(
    id='MyTestingEnv-v0',
    entry_point='[3]创建复杂gym环境:MyEnv',
)
register(
    id='MyTestingEnv-v1',
    entry_point='[3_1]创建复杂gym环境:MyEnv',
)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def lerp(value1, value2, t):
    return value1 + (value2 - value1) * t


def clamp(value, min_value, max_value):
    value = min(value, max_value)
    value = max(value, min_value)
    return value


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0,
                 enable_action_noise=True,
                 action_noise_start_value=1.0,
                 action_noise_end_value=0.0,
                 action_noise_start_percent=0.0,
                 action_noise_end_percent=0.5):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.progress_bar = None
        self.total_timesteps = -1

        self.enable_action_noise = enable_action_noise
        self.action_noise_start_value = action_noise_start_value
        self.action_noise_end_value = action_noise_end_value
        self.action_noise_start_percent = action_noise_start_percent
        self.action_noise_end_percent = action_noise_end_percent
        self.episode_images = []  # 用于存储当前 episode 的图片
        self.episode_count = 0  # 用于计数 episode 数量
        self.num_envs = None  # 初始化环境数量

    def _on_training_start(self) -> None:
        self.total_timesteps = self.locals['total_timesteps']
        self.progress_bar = tqdm(total=self.total_timesteps)
        self.num_envs = self.training_env.num_envs  # 获取并行环境数量
        self.episode_images = [[] for _ in range(self.num_envs)]  # 每个环境初始化一个子列表

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        percent = self.num_timesteps / self.total_timesteps
        self.progress_bar.update(self.training_env.num_envs)

        if self.enable_action_noise:
            real_percent = clamp((percent - self.action_noise_start_percent) / (
                        self.action_noise_end_percent - self.action_noise_start_percent), 0, 1)
            noise_value = lerp(self.action_noise_start_value, self.action_noise_end_value, real_percent)
            new_noise = NormalActionNoise(mean=np.zeros(2, ), sigma=np.ones(2, ) * noise_value)
            self.model.action_noise = VectorizedActionNoise(new_noise, self.training_env.num_envs)
            self.logger.record("custom/noise_value", noise_value)

        # 渲染所有并行环境的当前图片
        try:
            for i in range(self.training_env.num_envs):
                # 检查当前环境是否完成
                if self.locals['dones'][i]:  # 如果环境 i 完成
                    # print(f"Environment {i} completed an episode.")  # 输出完成信息
                    if hasattr(self.training_env, "envs"):
                        sub_env = self.training_env.envs[i].unwrapped  # 解封装第一个子环境
                        # print(f"Environment {i} completed unwrapped.")  # 输出完成信息
                        # 渲染完成的环境的最后一帧图片
                        image = sub_env.render()
                        # print(f"Environment {i} completed render.")  # 输出完成信息
                        if image is None:
                            print(f"Environment {i} did not return a valid image.")  # 检查渲染是否返回 None
                            continue
                        # 调整图片大小（可选）
                        resized_image = cv2.resize(image, (256, 256))  # 调整到 256x256

                        # 保存图片到 TensorBoard
                        self.episode_count += 1
                        tag_name = f"env_{i}_final_images"  # 同一环境的图片用同一标签
                        self.logger.record(
                            tag_name,
                            SBImage(resized_image, "HWC"),
                            exclude=("stdout", "log", "json", "csv")
                        )
                        self.logger.dump(self.episode_count)  # 强制写入日志
                        print(f"Saved final image for environment {i}, episode {self.episode_count}.")
        except Exception as e:
            print(f"Error during rendering or logging: {e}")

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        self.progress_bar.close()
        self.progress_bar = None


class MyGLContext(headless_utils.GLContext):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        DATA_PATH = "../data/VirtualEnv/0412data.bin"
        ENV_NAME = "MyTestingEnv-v1"

        data = io_utils.load_data(DATA_PATH)
        building_collection = BuildingCollection()
        building_collection.data_to_buildings(data)
        region_collection = RegionCollection()
        region_collection.data_to_regions(data)

        # 单独的gym环境
        env = gym.make(ENV_NAME,
                       data=data,
                       building_collection=building_collection,
                       region_collection=region_collection)
        eval_env = Monitor(env)

        # region DDPG
        # # 初始化模型
        # model = DDPG(
        #     policy='MlpPolicy',
        #     env=make_vec_env(ENV_NAME,
        #                      n_envs=4,  # 使用N个环境同时训练
        #                      env_kwargs={"data": data,
        #                                  "building_collection": building_collection,
        #                                  "region_collection": region_collection}
        #                      ),
        #     learning_rate=1e-5,
        #     buffer_size=10_000,  # replay_buff_size
        #     learning_starts=10_000,  # 从N步以后开始训练
        #     batch_size=128,  # 一次采样数据量
        #     tau=0.001,  # target网络每次软更新量
        #     gamma=0.9,
        #     train_freq=(5, 'step'),  # 训练频率
        #     verbose=0,
        #     tensorboard_log="./logs/tensorboard/MyTestingEnv/",
        #     policy_kwargs={
        #         'features_extractor_class': NatureCNN,
        #         'features_extractor_kwargs': {
        #             'features_dim': 512
        #         },
        #     },
        #     device="cuda",
        # )
        # endregion
        # region TD3
        # https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
        # Original implementation: https://github.com/sfujim/TD3
        # Paper: https://arxiv.org/abs/1802.09477
        # TD3介绍: https://spinningup.openai.com/en/latest/algorithms/td3.html
        # 初始化模型
        # model = TD3(
        #     policy='MlpPolicy',
        #     env=make_vec_env(ENV_NAME,
        #                      n_envs=4,  # 使用N个环境同时训练
        #                      env_kwargs={"data": data,
        #                                  "building_collection": building_collection,
        #                                  "region_collection": region_collection}
        #                      ),
        #     learning_rate=1e-5,
        #     buffer_size=10_000,  # replay_buff_size
        #     learning_starts=10_000,  # 从N步以后开始训练
        #     batch_size=256,  # 一次采样数据量
        #     tau=0.005,  # target网络每次软更新量
        #     gamma=0.99,  # the discount factor
        #     train_freq=(1, 'step'),  # 训练频率
        #     gradient_steps=1,  # How many gradient steps to do after each rollout (see train_freq) Set to -1 means to do as many gradient steps as steps done in the environment during the rollout.
        #     optimize_memory_usage=False,  # Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        #     policy_delay=2,  # Policy and target networks will only be updated once every policy_delay steps per training steps. The Q values will be updated policy_delay more often (update every training step).
        #     verbose=0,
        #     tensorboard_log="./logs/tensorboard/MyTestingEnv/",
        #     policy_kwargs={
        #         'features_extractor_class': NatureCNN,
        #         'features_extractor_kwargs': {
        #             'features_dim': 512
        #         },
        #     },
        #     device="cuda",
        # )
        # endregion
        # region DQN
        # 使用DQN替换TD3，适用于离散动作空间
        # model = DQN(
        #     policy='MlpPolicy',
        #     env=make_vec_env(ENV_NAME,
        #                      n_envs=4,  # 使用4个环境同时训练
        #                      env_kwargs={"data": data,
        #                                  "building_collection": building_collection,
        #                                  "region_collection": region_collection}
        #                      ),
        #     learning_rate=1e-4,  # 学习率
        #     buffer_size=10_000,  # 经验回放缓冲区大小
        #     learning_starts=10_000,  # 从10,000步以后开始训练
        #     batch_size=256,  # 每次更新的批量大小
        #     tau=0.005,  # 目标网络软更新参数（DQN中一般使用硬更新，但这里仍可设置）
        #     gamma=0.99,  # 折扣因子
        #     train_freq=(1, 'step'),  # 每步训练一次
        #     gradient_steps=1,  # 每次 rollout 后进行的梯度更新步数
        #     target_update_interval=500,  # 目标网络更新频率
        #     exploration_fraction=0.1,  # 探索与利用平衡，epsilon-greedy算法的epsilon线性下降比例
        #     exploration_final_eps=0.05,  # epsilon-greedy最终值
        #     verbose=1,
        #     tensorboard_log="./logs/tensorboard/MyTestingEnv/",
        #     device="cuda",
        # )
        # endregion
        # region PPO
        # 使用PPO替换TD3，适用于离散动作空间
        model = PPO(
            policy='MlpPolicy',
            # 一个VecEnv环境
            env=make_vec_env(ENV_NAME,
                             n_envs=4,  # 使用4个环境同时训练
                             env_kwargs={"data": data,
                                         "building_collection": building_collection,
                                         "region_collection": region_collection}
                             ),
            learning_rate=3e-4,  # 学习率
            n_steps=64,  # 每个环境中收集的步数
            batch_size=16,  # 每次更新的批量大小
            n_epochs=10,  # 每次数据重复训练的轮次
            gamma=0.99,  # 折扣因子
            gae_lambda=0.95,  # GAE（广义优势估计）参数
            clip_range=0.2,  # PPO的裁剪范围
            ent_coef=0.01,  # 熵损失系数，鼓励探索
            verbose=1,
            tensorboard_log="./logs/tensorboard/MyTestingEnv/",
            device="cuda",
        )
        # endregion
        # print(model.policy)

        # 回调函数,用于定期评估训练过程中的模型的表现
        folder_path = os.path.join("./logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=f"{folder_path}/best_model",
                                     log_path=f"{folder_path}/results",
                                     eval_freq=int(10_000 / model.env.num_envs)
                                     )
        custom_callback = CustomCallback(verbose=0,
                                         enable_action_noise=True,
                                         action_noise_start_value=1.0,
                                         action_noise_end_value=0.0,
                                         action_noise_start_percent=0.0,
                                         action_noise_end_percent=0.5
                                         )

        print("evaluating policy")
        eval_env.reset()
        result = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
        print(result)

        callback = CallbackList([custom_callback, eval_callback])

        print("start learning")
        model.learn(50, callback=callback)
        print("ending learning")

        print("evaluating policy")
        result = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
        print(result)

        # obs, info = env.reset()
        # print(type(env))
        # print(env)

        # # 使用训练好的模型测试并渲染
        # for _ in range(100):  # 测试100步
        #     action, _ = model.predict(obs, deterministic=True)  # 使用训练好的模型预测动作
        #     # obs, reward, done, info = env.step(action)
        #     obs, reward, done, truncated, info = env.step(action)  # VecEnv 的 step 返回多个值
        #     # 渲染第一个子环境
        #     env.render()  # 仅渲染第一个环境的画面
        #     print("env.render")
        #     # 处理完成的环境
        #
        #     if done:  # 如果某个子环境完成
        #         obs = env.reset()  # 单独重置完成的子环境


if __name__ == '__main__':
    MyGLContext.run()
