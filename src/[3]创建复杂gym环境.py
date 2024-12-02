from __future__ import annotations

import random
from typing import Any, SupportsFloat, Optional

import cv2
import geopandas as gpd
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.core import ObsType, ActType, RenderFrame

import graphic_module
from geo import RoadCollection, BuildingCollection, RegionCollection
from style_module import StyleManager
from utils import headless_utils, RoadLevel, RoadState, io_utils


def generate_chessboard(image_size, grid_size = 16):
    # 生成棋盘格图像
    chessboard = np.zeros((image_size, image_size, 4), dtype=np.uint8)

    # 生成棋盘格的行列索引
    rows, cols = np.indices((image_size, image_size)) // grid_size

    # 根据行列索引生成棋盘格
    chessboard[:, :, :3] = np.where((rows + cols) % 2 == 0, 200, 100)[:,:,np.newaxis] # RGB通道
    chessboard[:, :, 3] = 255  # 设置透明度通道为不透明
    return chessboard


class MyEnv(gym.Env):
    def __init__(self,
                 data,
                 building_collection=None,
                 region_collection=None,
                 max_episode_step: int = 50,
                 observation_img_size: tuple[int, int] = (512, 512),  # width, height
                 observation_view_size: tuple[float, float] = (400, 400),
                 action_step_max: float = 40,
                 ):
        super().__init__()
        print("env init")
        # parameters
        self.max_episode_step = max_episode_step
        self.observation_img_size = observation_img_size
        self.observation_view_size = observation_view_size
        self.action_step_max = action_step_max

        # action space and observation space
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (observation_img_size[1], observation_img_size[0], 4), dtype=np.uint8)  # H, W, C

        # create geometries
        self.Road: RoadCollection = RoadCollection()
        self.Road.data_to_roads(data)
        if building_collection is None:
            self.Building: BuildingCollection = BuildingCollection()
            self.Building.data_to_buildings(data)
        else:
            self.Building = building_collection
        if region_collection is None:
            self.Region: RegionCollection = RegionCollection()
            self.Region.data_to_regions(data)
        else:
            self.Region = region_collection
        self.road_bbox = self.Road.get_bbox()
        self.Road.cache()  # 保存当前路网状态，以备复原
        # variables [这里的变量在reset的时候需要被重置]
        self.curr_step = 0  # current step
        self.raw_roads: gpd.GeoDataFrame = self.Road.get_all_roads()  # 保存原有道路， 分裂后的新道路不在其中
        self.road_agent: Optional[pd.Series] = None  # 道路智能体

        # init
        # (init observers)
        W, H = self.observation_img_size
        view_size = self.observation_view_size
        self.raw_roads_observer = graphic_module.RoadObserver('raw_roads_obs', W, H, view_size, self.raw_roads)
        self.new_road_observer = graphic_module.RoadObserver('new_road_obs', W, H, view_size, None, )
        self.building_observer = graphic_module.BuildingObserver('building_obs', W, H, view_size, self.Building.get_all_buildings())
        self.region_observer = graphic_module.RegionObserver('region_obs', W, H, view_size, self.Region.get_all_regions())
        self.bound_observer = graphic_module.RegionObserver('bound_obs', W, H, view_size, self.Region.create_region_by_min_max(*self.road_bbox), StyleManager.I.env.region_simple_style_factory)
        self.node_observer = graphic_module.NodeObserver('node_obs', W, H, view_size, self.Road.get_all_nodes(), road_collection=self.Road)
        self.blend_observer = graphic_module.ObsBlender('blend_obs', W, H)
        # (init reward)
        # TODO
        # others
        self.STRTree_key = self.Road.build_STRTree(self.raw_roads)


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        self.Road.restore()
        self.curr_step = 0
        self.raw_roads = self.Road.get_all_roads()
        self._clear_and_spawn_agents()  # self.road_agent changed here

        observation = self._get_observation()
        info = {}
        return observation, info

    def _clear_and_spawn_agents(self):
        self.road_agent = None
        candidate_roads = [row for index, row in self.raw_roads.iterrows()]
        random.shuffle(candidate_roads)
        while len(candidate_roads) > 0:
            random_road = candidate_roads.pop(0)
            spawn_point = self.Road.interpolate_road_by_random_position(random_road)
            if spawn_point is None:
                continue  # 如果找不到符合路网间距规范的点，则重新选一条路
            spawn_point = spawn_point.reshape(-1, 2)
            self.Road.split_road_by_coord(random_road, spawn_point)  # 在路上随机一个点并尝试分裂
            uid = self.Road.add_road_by_coords(spawn_point, RoadLevel.TERTIARY, RoadState.OPTIMIZING)  # 生成新路
            self.road_agent = self.Road.get_road_by_uid(uid)
            return
        raise Exception("cannot find road")

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), "Action is not in action space"
        self.curr_step += 1

        # move agent
        self._move_road_agent(action)
        # update observers
        self._update_observers()

        # get observation, done, reward and info， 注意这里是有顺序的
        observation = self._get_observation()
        done = self._is_agent_done()
        reward = -10 if done else 1
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def _move_road_agent(self, action):
        # action: numpy.array, shape=[2, ], value from -1 to 1, float32
        vec = action * self.action_step_max  # [-1.0 - 1.0] -> [-action_step_max, action_step_max]
        vec = vec.reshape(1, 2)  # shape: [2, ] -> [1, 2]
        lst_pt = self.Road.get_road_last_point(self.road_agent)  # shape: [1, 2]
        self.road_agent = self.Road.add_point_to_road(self.road_agent, point=lst_pt + vec)

    def _update_observers(self):
        # update buffer
        self.new_road_observer.update_buffer(self.road_agent.to_frame().T)

        # update prog
        lst_pt = self.Road.get_road_last_point(self.road_agent)[0]  # shape: [2, ]
        self.raw_roads_observer.update_observation_center(lst_pt)
        self.new_road_observer.update_observation_center(lst_pt)
        self.building_observer.update_observation_center(lst_pt)
        self.region_observer.update_observation_center(lst_pt)
        self.bound_observer.update_observation_center(lst_pt)
        self.node_observer.update_observation_center(lst_pt)

        # render image
        self.raw_roads_observer.render()
        self.new_road_observer.render()
        self.building_observer.render()
        self.region_observer.render()
        self.bound_observer.render()
        self.node_observer.render()

        self.blend_observer.render(
            [self.bound_observer.texture,
             self.region_observer.texture,
             self.building_observer.texture,
             self.raw_roads_observer.texture,
             self.new_road_observer.texture,
             self.node_observer.texture]  # 这里的顺序需要和shader中的texture的顺序对应
        )

    def _get_observation(self) -> np.ndarray:
        # 这里和原先的get observation image有所不同
        # 其更新buffer和prog的操作被移至了_update_observers
        # 这里默认observers已经被更新过了
        observation = self.blend_observer.get_render_img()
        return observation

    def _is_agent_done(self) -> bool:
        if self.curr_step >= self.max_episode_step:
            return True

        # is in region?
        lst_pt = self.Road.get_road_last_point(self.road_agent)[0]  # shape: [2, ]
        in_region = True
        in_region &= self.road_bbox[0][0] < lst_pt[0] < self.road_bbox[1][0]  # bbox.min.x < pt.x < bbox.max.x
        in_region &= self.road_bbox[0][1] < lst_pt[1] < self.road_bbox[1][1]  # bbox.min.y < pt.y < bbox.max.y
        if not in_region:
            return True

        # is intersected with raw roads?
        intersect = self.Road.detect_intersection_fast(self.STRTree_key, self.road_agent)
        if intersect:
            return True

        return False

    def _get_reward(self):
        _ = self
        return 0

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._get_observation()

    def close(self):
        print("closing env...")
        self.Road.remove_STRTree(self.STRTree_key)


class MyGLContext(headless_utils.GLContext):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        chessboard = generate_chessboard(512, 10)

        data = io_utils.load_data("../data/和县/data.bin")
        env = MyEnv(data)
        state, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state

            img = env.render()
            img = cv2.resize(img, (512, 512))

            result = chessboard.copy()
            alpha = img[:, :, 3] / 255.0
            for c in range(3):
                result[:, :, c] = chessboard[:, :, c] * (1 - alpha) + img[:, :, c] * alpha
            # 将BGR图像转换为RGB格式
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            cv2.imshow("render", result)
            key = cv2.waitKey(0)  # 设置帧率为 30fps
            if key == ord('q'):
                break
        env.close()



if __name__ == '__main__':
    MyGLContext.run()
    # cv2.imshow("Chessboard", generate_chessboard(512, 10))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()