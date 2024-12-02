from __future__ import annotations
from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from utils import headless_utils


class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.pos = np.zeros((2,), dtype=np.float32)
        self.steps = 0

        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        self.pos = 0
        self.steps = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.steps += 1
        self.pos += action

        observation = self._get_observation()
        done = self.steps > 10
        reward = 1 if done else 0
        truncated = False
        info = {}
        return observation, reward, done, truncated, info

    def _get_observation(self):
        observation = (np.random.rand(*self.observation_space.shape) * 255).astype(np.uint8)
        return observation

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        print(self.pos)
        return None

    def close(self):
        pass


class MyGLContext(headless_utils.GLContext):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        env = MyEnv()
        state, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            env.render()
            state = next_state


if __name__ == '__main__':
    MyGLContext.run()
