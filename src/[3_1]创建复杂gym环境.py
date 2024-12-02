from __future__ import annotations

import random
from enum import Enum
from typing import Any, SupportsFloat, Optional

import cv2
import geopandas as gpd
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.core import ObsType, ActType, RenderFrame

import graphic_module
from geo import RoadCollection, BuildingCollection, RegionCollection, Road, Building, Region
from style_module import StyleManager
from utils import headless_utils, RoadLevel, RoadState, io_utils


class RewardSystem:
    def __init__(self, env: "MyEnv"):
        """
        定义一个简单的reward系统
        :param env: parent env
        """
        self.env = env
        self.Road = env.Road
        self.Building = env.Building
        self.Region = env.Region

        # region 创建observer 渲染器
        self.observers = {
            "raw_road_observer": graphic_module.RoadObserver(
                "raw_road_observer",
                width=64, height=64,
                observation_size=(100, 100),
                initial_gdf=self.Road.get_all_roads(),
                sf=StyleManager.I.env.road_simple_style_factory_by_level,
                bg_color=(0, 0, 0, 1)
            ),
            "new_road_observer": graphic_module.RoadObserver(
                "new_road_observer",
                width=64, height=64,
                observation_size=(100, 100),
                initial_gdf=None,
                sf=StyleManager.I.env.road_simple_style_factory_by_level,
                bg_color=(0, 0, 0, 1)
            ),
            "parent_road_observer": graphic_module.RoadObserver(
                "reward_parent_road",
                width=64, height=64,
                observation_size=(100.0, 100.0),
                initial_gdf=None,
                sf=StyleManager.I.env.road_simple_style_factory_by_level,  # color white and width by level
                bg_color=(0, 0, 0, 1)
            ),
            "building_observer": graphic_module.BuildingObserver(
                name='reward_building',
                width=64, height=64,
                observation_size=(100.0, 100.0),
                initial_gdf=Building.get_all_buildings(),  # 后面可以筛选出需要计算碰撞的building
                sf=StyleManager.I.env.building_simple_style_factory,  # all colored white
                bg_color=(0, 0, 0, 1)
            ),

            "region_observer": graphic_module.RegionObserver(
                name='reward_region',
                width=64, height=64,
                observation_size=(100.0, 100.0),
                initial_gdf=Region.get_all_regions(),  # 后面可以筛选出需要计算碰撞的region
                sf=StyleManager.I.env.region_simple_style_factory,  # all colored white
                bg_color=(0, 0, 0, 1)
            ),

            "node_observer": graphic_module.NodeObserver(
                name='reward_node',
                width=64, height=64,
                observation_size=(100.0, 100.0),
                initial_gdf=Road.get_all_nodes(),
                sf=StyleManager.I.env.node_radius_penalty_factory,  # colored white
                bg_color=(0, 0, 0, 1),
                road_collection=Road
            ),
        }

        self.post_processings = {
            "raw_road_observer_blur": graphic_module.RewardBlurPostProcessing(
                "raw_road_observer_blur",
                self.observers['raw_road_observer'],
                radius=16
            ),
            "parent_road_observer_blur": graphic_module.RewardBlurPostProcessing(
                "parent_road_observer_blur",
                self.observers['parent_road_observer'],
                radius=16
            )
        }

        # endregion

    def get_reward(self, road_agent: pd.Series, done: bool, reason: DoneReason):
        """
        主函数， 计算reward
        :param road_agent:
        :param done:
        :param reason:
        :return:
        """
        # 首先更新observers
        self._update_observers(road_agent)

        # 计算每一步的reward
        reward = self.基础得分() + self.鼓励走出去的得分(road_agent) + self.转弯角度得分(road_agent) + self.与建筑相交的得分() + self.与区域相交的得分() + self.与路口相交的得分() + self.与出生道路相交的得分() + self.远离出生道路的得分()

        # 计算最终reward
        if done:
            reward += self.完成原因赋予的得分(reason)

        return reward

    def _update_observers(self, road_agent):
        # update buffer
        self.observers['new_road_observer'].update_buffer(road_agent.to_frame().T)

        # update prog
        lst_pt = self.Road.get_road_last_point(road_agent)[0]  # shape: [2, ]
        for _, observer in self.observers.items():
            observer.update_observation_center(lst_pt)

        # render image
        for _, observer in self.observers.items():
            observer.render()

        # render post_processing
        for _, observer in self.post_processings.items():
            observer.render()

    def _get_pixel_intersect_value(self, observer1: Optional[graphic_module.Observer, graphic_module.RewardBlurPostProcessing], observer2: Optional[graphic_module.Observer, graphic_module.RewardBlurPostProcessing]):
        _ = self
        arr1 = observer1.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        arr2 = observer2.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        overlap_arr = arr1 * arr2
        pixel_value = np.max(overlap_arr)
        return pixel_value  # 0: no intersect, 1: intersect

    def 基础得分(self):
        _ = self
        return 1.0

    def 鼓励走出去的得分(self, road_agent):
        coords = road_agent["coords"]
        if len(coords) < 3:  # 0, 1, 2个点时不做计算
            return 0
        length = self.Road.get_road_length_using_coords(road_agent)
        distance = np.linalg.norm(coords[0] - coords[-1])
        if distance == 0:
            return -2
        if length / distance > 3.0:
            return -1
        elif length / distance > 1.414:
            return -0.5
        return 0

    def 转弯角度得分(self, road_agent):
        _ = self
        coords = road_agent["coords"]
        if len(coords) < 3:  # 0, 1, 2个点时不做计算
            return 0

        vec1 = coords[-1] - coords[-2]
        vec2 = coords[-2] - coords[-3]
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        if len1 * len2 == 0:
            return 0
        norm_vec1 = vec1 / len1
        norm_vec2 = vec2 / len2
        dot_product = np.dot(norm_vec1, norm_vec2)
        if abs(dot_product) < 0.1:
            return 0  # 对于接近90度的转角， 不持态度
        elif dot_product > 0.7:
            return 0.1  # 对延续之前的方向，给轻微的支持
        elif dot_product < -0.7:
            return -1  # 对超过135度的转弯， 持否定态度
        return 0

    def 与建筑相交的得分(self):
        """-2 to 0"""
        intersect = self._get_pixel_intersect_value(self.observers['new_road_observer'], self.observers['building_observer'])
        return intersect * -5

    def 与区域相交的得分(self):
        """-2 to 0"""
        intersect = self._get_pixel_intersect_value(self.observers['new_road_observer'], self.observers['region_observer'])
        return intersect * -5

    def 与路口相交的得分(self):
        intersect = self._get_pixel_intersect_value(self.observers['new_road_observer'], self.observers['node_observer'])
        return intersect * -5

    def 与出生道路相交的得分(self):
        """-10 to 0"""
        intersect = self._get_pixel_intersect_value(self.observers['new_road_observer'], self.observers['parent_road_observer'])
        return intersect * -10  # 分数要足以抵消和道路相交的正奖励

    def 远离出生道路的得分(self):
        """-1 to 1"""
        intersect = self._get_pixel_intersect_value(self.observers['new_road_observer'], self.post_processings['parent_road_observer_blur'])
        return 2 * (intersect - 0.5)

    def 完成原因赋予的得分(self, reason: DoneReason):
        _ = self
        if reason == DoneReason.NoReason:
            return 0
        elif reason == DoneReason.OutOfRegion:
            return -10
        elif reason == DoneReason.ReachMaxEpisode:
            return -1
        elif reason == DoneReason.IntersectWithRoad:
            return 10
        return 0


class DoneReason(Enum):
    """智能体结束的原因(Enum)"""
    NoReason = -1  # 没有原因
    ReachMaxEpisode = 0  # 到达最大步数
    OutOfRegion = 1  # 超出区域
    IntersectWithRoad = 2  # 与道路相交


class MyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], 'render_fps': 120}

    def __init__(self,
                 data,
                 building_collection=None,
                 region_collection=None,
                 max_episode_step: int = 50,
                 observation_img_size: tuple[int, int] = (128, 128),  # width, height
                 observation_view_size: tuple[float, float] = (400, 400),
                 action_step_max: float = 40,
                 ):
        """
        创建训练环境
        :param data: 地图数据
        :param building_collection: 已经创建好的building_collection, 由于building在训练过程中通常是不变的，因此可以使用已经创建好的来加快环境加载速度
        :param region_collection: 已经创建好的region_collection, 由于region在训练过程中通常是不变的，因此可以使用已经创建好的来加快环境加载速度
        :param max_episode_step: 最大步数
        :param observation_img_size: 观察图像大小
        :param observation_view_size: 观察视野大小
        :param action_step_max: 最大步长
        """
        super().__init__()
        # parameters
        self.max_episode_step = max_episode_step
        self.observation_img_size = observation_img_size
        self.observation_view_size = observation_view_size
        self.action_step_max = action_step_max

        # action space and observation space
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (4, observation_img_size[1], observation_img_size[0]), dtype=np.uint8)  # C, H, W

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

        bbox_min, bbox_max = self.Road.get_bbox()
        bbox_min -= np.array([20, 20])
        bbox_max += np.array([20, 20])
        self.road_bbox = (bbox_min, bbox_max)

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
        self.blend_observer = graphic_module.ObsBlender('blend_obs', W, H)  # 合成渲染器
        # (init reward)
        self.reward_system = RewardSystem(env=self)
        # others
        self.STRTree_key = self.Road.build_STRTree(self.raw_roads)
        self.render_mode = "rgb_array"
        self.chessboard = self._generate_chessboard(H, W, 5)  # 创建一个棋盘格背景作为render画面的背景

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        self.Road.restore()  # 恢复原样
        self.curr_step = 0
        self.raw_roads = self.Road.get_all_roads()
        self._clear_and_spawn_agents()  # 清理现有智能体，并随机创建新智能体， self.road_agent changed here

        observation = self._get_observation()
        info = {}
        return observation, info

    def _generate_chessboard(self, height, width, grid_size=16):
        _ = self
        # 生成棋盘格图像
        chessboard = np.zeros((height, width, 4), dtype=np.uint8)

        # 生成棋盘格的行列索引
        rows, cols = np.indices((height, width)) // grid_size

        # 根据行列索引生成棋盘格
        chessboard[:, :, :3] = np.where((rows + cols) % 2 == 0, 200, 100)[:, :, np.newaxis]  # RGB通道
        chessboard[:, :, 3] = 255  # 设置透明度通道为不透明
        return chessboard

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
        done, reason = self._is_agent_done()
        reward = self.reward_system.get_reward(self.road_agent, done, reason)
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

    def _get_observation(self, fmt="CHW") -> np.ndarray:
        # 这里和原先的get observation image有所不同
        # 其更新buffer和prog的操作被移至了_update_observers
        # 这里默认observers已经被更新过了
        assert fmt == "CHW" or fmt == "HWC"
        observation = self.blend_observer.get_render_img()  # H, W, C
        if fmt == "CHW":
            observation = np.transpose(observation, (2, 0, 1))  # C, H, W
        return observation

    def _is_agent_done(self) -> tuple[bool, DoneReason]:
        if self.curr_step >= self.max_episode_step:
            return True, DoneReason.ReachMaxEpisode

        # is in region?
        lst_pt = self.Road.get_road_last_point(self.road_agent)[0]  # shape: [2, ]
        in_region = True
        in_region &= self.road_bbox[0][0] < lst_pt[0] < self.road_bbox[1][0]  # bbox.min.x < pt.x < bbox.max.x
        in_region &= self.road_bbox[0][1] < lst_pt[1] < self.road_bbox[1][1]  # bbox.min.y < pt.y < bbox.max.y
        if not in_region:
            return True, DoneReason.OutOfRegion

        # is intersected with raw roads?
        intersect = self.Road.detect_intersection_fast(self.STRTree_key, self.road_agent)
        if intersect:
            return True, DoneReason.IntersectWithRoad

        return False, DoneReason.NoReason

    def _get_reward(self):
        _ = self
        return 0

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        obs = self._get_observation(fmt="HWC")
        result = self.chessboard.copy()
        alpha = obs[:, :, 3] / 255.0
        for c in range(3):
            result[:, :, c] = self.chessboard[:, :, c] * (1 - alpha) + obs[:, :, c] * alpha
        return result[:, :, :3]

    def close(self):
        print("closing env...")
        self.Road.remove_STRTree(self.STRTree_key)

    def test_method(self):
        print("hello world")


class MyGLContext(headless_utils.GLContext):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        data_path = "../data/VirtualEnv/0414.bin"

        data = io_utils.load_data(data_path)
        env = MyEnv(data)
        state, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            print(reward)
            state = next_state
            img = env.render()
            result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
