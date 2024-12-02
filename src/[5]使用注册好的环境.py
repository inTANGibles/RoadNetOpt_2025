import cv2
import gymnasium as gym
from gymnasium.envs.registration import register

from utils import headless_utils, io_utils

register(
    id='MyTestingEnv-v0',
    entry_point='[3]创建复杂gym环境:MyEnv',
)


class MyGLContext(headless_utils.GLContext):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        data = io_utils.load_data("../data/和县/data.bin")
        env = gym.make('MyTestingEnv-v0', data=data)

        state, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state

            img = env.render()
            img = cv2.resize(img, (512, 512))
            cv2.imshow("render", img)
            key = cv2.waitKey(30)  # 设置帧率为 30fps
            if key == ord('q'):
                break
        env.close()


if __name__ == '__main__':
    MyGLContext.run()
