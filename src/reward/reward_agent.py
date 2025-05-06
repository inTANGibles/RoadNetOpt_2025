import uuid
from typing import Optional

import pandas as pd
import numpy as np

import graphic_module as gm
from geo import Road, Building, Region
from graphic_module import GraphicManager
from style_module import StyleManager as sm
from utils import RoadState
from DDPG.utils.reward_utils import *

class RewardAgent:
    """reward值计算的Opengl实现版本"""

    def __init__(self, region_min: tuple[int, int], region_max: tuple[int, int], headless=True, road_collection=None):
        if road_collection is None:
            road_collection = Road
        self.Road = road_collection

        # bound
        self.region_min = np.array(region_min)
        self.region_max = np.array(region_max)

        self.orgin_road_collection = None
        self.analysis_net = None
        self.reward_net = None


        # observers
        self.observers: dict[str:gm.Observer] = {
            'raw_road_observer': gm.RoadObserver(
                name='reward_raw_roads',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=self.Road.get_all_roads(),
                sf=sm.I.env.road_simple_style_factory_by_level,  # color white and width by level
                bg_color=(0, 0, 0, 1)
            ),
            'new_road_observer': gm.RoadObserver(
                name='reward_new_roads',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=None,
                sf=sm.I.env.road_simple_style_factory_by_level,  # color white and width by level
                bg_color=(0, 0, 0, 1)
            ),
            'parent_road_observer': gm.RoadObserver(
                name='reward_parent_road',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=None,
                sf=sm.I.env.road_simple_style_factory_by_level,  # color white and width by level
                bg_color=(0, 0, 0, 1)
            ),
            'building_observer': gm.BuildingObserver(
                name='reward_building',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=Building.get_all_buildings(),  # 后面可以筛选出需要计算碰撞的building
                sf=sm.I.env.building_simple_style_factory,  # all colored white
                bg_color=(0, 0, 0, 1)
            ),
            'region_observer': gm.RegionObserver(
                name='reward_region',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=Region.get_all_regions(),  # 后面可以筛选出需要计算碰撞的region
                sf=sm.I.env.region_simple_style_factory,  # all colored white
                bg_color=(0, 0, 0, 1)
            ),
            'bound_observer': gm.RegionObserver(
                name='reward_bound',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=Region.create_region_by_min_max(self.region_min, self.region_max),
                sf=sm.I.env.region_simple_style_factory,  # colored white
                bg_color=(0, 0, 0, 1)
            ),
            # 'node_observer': gm.NodeObserver(
            #     name='reward_node',
            #     width=64, height=64,
            #     observation_size=(100.0, 100.0),
            #     initial_gdf=self.Road.get_all_nodes(),
            #     sf=sm.I.env.node_radius_penalty_factory,  # colored white
            #     bg_color=(0, 0, 0, 1),
            #     road_collection=self.Road
            # ),
            'dead_node_observer': gm.NodeObserver(
                name='reward_dead_nodes',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=self.Road.get_dead_nodes(),
                sf=sm.I.env.node_radius_penalty_factory,
                bg_color=(0, 0, 0, 1),
                road_collection=self.Road
            ),
            'connectable_node_observer': gm.NodeObserver(
                name='reward_connectable_nodes',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=self.Road.get_connectable_nodes(),
                sf=sm.I.env.node_radius_penalty_factory,
                bg_color=(0, 0, 0, 1),
                road_collection=self.Road
            ),
            'cross_node_observer': gm.NodeObserver(
                name='reward_cross_nodes',
                width=64, height=64,
                observation_size=(256.0, 256.0),
                initial_gdf=self.Road.get_cross_nodes(),
                sf=sm.I.env.node_radius_penalty_factory,
                bg_color=(0, 0, 0, 1),
                road_collection=self.Road
            ),

        }
        for observer in self.observers.values():
            GraphicManager.I.register_reward_observer(observer)
        self.post_processings = {
            'raw_road_blur': gm.RewardBlurPostProcessing(
                'reward_raw_road_blur',
                self.observers['raw_road_observer'],
                radius=5
            ),
            'new_road_blur': gm.RewardBlurPostProcessing(
                'reward_new_road_blur',
                self.observers['new_road_observer'],
                radius=5
            ),
            'parent_road_blur': gm.RewardBlurPostProcessing(
                'reward_parent_road_blur',
                self.observers['parent_road_observer'],
                radius=20
            ),
            'building_blur': gm.RewardBlurPostProcessing(
                'reward_building_blur',
                self.observers['building_observer'],
                radius=5
            ),
            'region_blur': gm.RewardBlurPostProcessing(
                'reward_region_blur',
                self.observers['region_observer'],
                radius=10
            ),
            'bound_blur': gm.RewardBlurPostProcessing(
                'reward_bound_blur',
                self.observers['bound_observer'],
                radius=10
            ),
            # 'node_blur': gm.RewardBlurPostProcessing(
            #     'reward_node_blur',
            #     self.observers['node_observer'],
            #     radius=10
            # ),
            'dead_node_blur': gm.RewardBlurPostProcessing(
                'reward_node_blur',
                self.observers['dead_node_observer'],
                radius=10
            ),
            'connectable_node_blur': gm.RewardBlurPostProcessing(
                'reward_node_blur',
                self.observers['connectable_node_observer'],
                radius=20
            ),
            'cross_node_blur': gm.RewardBlurPostProcessing(
                'reward_node_blur',
                self.observers['cross_node_observer'],
                radius=10
            )
        }
        for post_processing in self.post_processings.values():
            GraphicManager.I.register_reward_observer(post_processing)


        # step weights 得分区间统一为[-1,0]，规则计数为累加 现有总和为 250
        self.building_region_weight = 30 # -
        self.road_weight = 20 # -
        self.bound_weight = 80 # -
        self.back_weight = 10 # -
        self.dead_node_weight = 50 # +
        self.cross_node_weight = 40 # -
        self.exploration_weight = 20 # +


        # 结束后的基础得分为1 * final_weight，附加得分为final weights * 1，得分区间统一为[0,1]，规则计数为相加
        self.final_weight = 0
        self.node_final_weight = 100 # 75 if 相乘 10
        self.dist_weight = 30 # 100 if 相乘 15
        self.angle_weight = 70
        # TODO 后续补充调整其他评价的权重关系

        # 奖励规则中的参数设置
        # final reward中的distance规则参数
        self.step_count = 5 #超过多少步，才启用dist奖励
        self.ratio_range = [1,3] #在该范围内的比值可以得到奖励
        # final reward中的acute_angle规则参数
        self.acute_count = 5 #锐角低于这个数目，才可以得到奖励
        # utils
        self.headless = headless

    # STEP-LEVEL
    # 惩罚项

    def building_region_reward(self):
        """限制不能进入的区域,传入当前建筑图像与智能体位置
        返回规则: 得分区间[0，1],越靠近建筑越接近于0"""
        render1 = self.post_processings['building_blur']
        render2 = self.post_processings['region_blur']
        render3 = self.post_processings['new_road_blur']
        assert render1.width == render2.width == render3.width
        assert render1.height == render2.height == render3.height
        building_arr = render1.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        region_arr = render2.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        road_arr = render3.get_render_img()[:, :, 0].astype(np.float32) / 255.0

        blurred_arr = np.clip((building_arr + region_arr), 0.0, 1.0)
        overlap_arr = road_arr * blurred_arr * 255
        overlap_arr = overlap_arr.astype(np.uint8)
        if not self.headless:
            texture = GraphicManager.I.get_or_create_simple_texture(name='reward_overlap',
                                                                    default_width=overlap_arr.shape[1],
                                                                    default_height=overlap_arr.shape[0])
            texture.exposed = False
            texture.bilt_data(overlap_arr)
            GraphicManager.I.register_reward_observer(texture)
        pixel_value = np.max(overlap_arr)
        min1 = 0
        max1 = 255
        reward = 1 - (1 / np.log(max1 - min1 + 1) * np.log(np.abs(pixel_value - min1 + 1)))  # 非线性变化
        return reward


    def road_reward(self):
        """限制智能体靠近出生道路,传入当前出生道路与智能体位置
                返回规则: 得分区间[0,-1],越靠近路越接近于0"""
        parent_road_renderer = self.post_processings['parent_road_blur']
        height, width = parent_road_renderer.height, parent_road_renderer.width
        blurred_img = parent_road_renderer.get_render_img()
        pixel_value = blurred_img[int(height / 2), int(width / 2), 0]
        min1 = 0  # 126
        max1 = 255  # int(torch.max(blurred_img))
        reward =1 - (1 / np.log(max1 - min1 + 1) * np.log(np.abs(pixel_value - min1 + 1)))  # 非线性变化
        return reward


    def bound_reward(self):
        bound_renderer = self.post_processings['bound_blur']
        height, width = bound_renderer.height, bound_renderer.width
        blurred_img = bound_renderer.get_render_img()
        pixel_value = blurred_img[int(height / 2), int(width / 2), 0]
        if pixel_value < 30:  # 你可以设一个阈值，如 30
            return 0.0
        else:
            min1 = 0
            max1 = 255
            return (1 / np.log(max1 - min1 + 1) * np.log(np.abs(pixel_value - min1 + 1)))


    def crossnode_reward(self):
        """过程中将交叉路口视为障碍物"""
        node_renderer = self.post_processings['cross_node_blur']
        height, width = node_renderer.height, node_renderer.width
        img = node_renderer.get_render_img()
        pixel_value = img[int(height / 2), int(width / 2), 0]
        min1 = 0
        max1 = 255  # int(torch.max(blurred_img))
        reward = 1 - (1 / np.log(max1 - min1 + 1) * np.log(np.abs(pixel_value - min1 + 1)))  # 非线性变化
        return reward

    def backward_penalty(self,_is_forward):
        if _is_forward:
            return 1
        else:
            return 0


    # 奖励项

    def deadnode_reward(self):
        """
        越靠近断头路或 connectable 节点（degree 为 2 或 3），奖励越高。
        """
        # 获取图像渲染器
        # dead_renderer = self.post_processings['dead_node_blur']
        connect_renderer = self.post_processings['connectable_node_blur']
        h, w = connect_renderer.height, connect_renderer.width

        # 提取中心像素值
        connect_val = connect_renderer.get_render_img()[h // 2, w // 2, 0]

        # 叠加像素值，归一化到 [0, 1]
        combined_val = np.clip(connect_val, 0, 255)
        reward = (1 / np.log(256)) * np.log(combined_val + 1)  # 非线性归一化

        return reward


    def exploration_reward(self):
        """
        若 agent 所处位置的观测区域中 没有其他 new_road，给予奖励
        """
        obs = self.observers['new_road_observer']
        img = obs.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        h, w = img.shape
        # 以中心区域为核心，提取 3x3 或 5x5 区域的均值
        center = img[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
        avg_val = np.mean(center)

        # 如果该区域 brightness 低于阈值，表示比较“干净”
        if avg_val < 0.05:
            return 1.0  # 强奖励
        else:
            return 0.0

    # @staticmethod
    # def step_penalty():
    #     """每走一步，累计一个负分"""
    #     return -1
    #
    # @staticmethod


    def endnode_final_reward(self):
        """判断生成的新路口位置，不过于靠近出生路;且终点不在出生道路上
        使用node、parent_road_blur图像，与reward_new_roads图像overlap的关系来判断,没有overlap则得到附加奖励
        """
        render1 = self.post_processings['cross_node_blur']
        render2 = self.post_processings['parent_road_blur']
        render3 = self.post_processings['new_road_blur']
        render4 = self.post_processings['dead_node_blur']
        render5 = self.post_processings['connectable_node_blur']

        assert render1.width == render2.width == render3.width
        assert render1.height == render2.height == render3.height

        node_arr = render1.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        parent_road_arr = render2.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        new_road_arr = render3.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        dead_arr = render4.get_render_img()[:, :, 0].astype(np.float32) / 255.0
        connect_arr = render5.get_render_img()[:, :, 0].astype(np.float32) / 255.0


        # 惩罚终点重叠交叉点或原始出生路
        blurred_arr = np.clip((node_arr + parent_road_arr), 0.0, 1.0)
        overlap_arr = new_road_arr * blurred_arr * 255
        overlap_arr = overlap_arr.astype(np.uint8)
        pixel_value = np.max(overlap_arr)
        penalty = 1 if pixel_value == 0 else 0

        # 奖励终点靠近 dead_node
        proximity_arr = new_road_arr * dead_arr * 255
        proximity_arr = proximity_arr.astype(np.uint8)
        reward_dead = 10 if np.max(proximity_arr) > 0 else 0
        # 奖励靠近connectable
        reward_connectable = 1.0 if np.max((new_road_arr * connect_arr * 255).astype(np.uint8)) > 0 else 0.0

        reward = (penalty + reward_dead + reward_connectable) / 3.0

        return reward

    def distance_final_reward(self, road_agent):
        """直线距离和实际距离的比值，并且排除步长较少的情况"""
        count = len(list(road_agent['geometry'].coords))
        end = self.Road.get_road_last_point(road_agent)
        start = self.Road.get_road_first_point(road_agent)
        distance = self.Road.get_road_length_using_coords(road_agent)
        direct_distance = np.linalg.norm(end - start)
        ratio = distance / direct_distance
        if count < self.step_count + 1:
            reward = 0
        else:
            reward = 0 if ratio > self.ratio_range[1] else np.exp(-6 * (ratio - self.ratio_range[0]))
        return reward

    def acute_angle_final_reward(self, acute_count):
        """最后的锐角总数，放到[0,1],越少越大；超过self.acute_count为0"""
        if acute_count > self.acute_count:
            return 0
        else:
            min1 = 0
            max1 = self.acute_count
            reward = 1 - (1 / np.log(max1 - min1 + 1) * np.log(np.abs(acute_count - min1 + 1)))  # 非线性变化
            return reward

    # def intersect_angle(self,road_agent,intersect:bool):
    #     """预计添加起点和终点线段与路网相加的夹角，保持在（45，90之间），[0,1]"""
    #     end = self.Road.get_road_last_element(road_agent)
    #     start = self.Road.get_road_first_element(road_agent)
    #     # end_dot_poduct = point_utils.vector_dot(start_vec1, start_vec2)
    #     # start_dot_poduct = point_utils.vector_dot(start_vec1, start_vec2)
    #     pass



    def reset(self, parent_roads):
        """每一轮需要刷新的内容"""
        pass

    def get_rewards(self, dones, positions, parent_roads, is_forwards, debug_dict: Optional[dict] = None):
        """
        每次step需要刷新的内容
        先进行渲染，然后再计算reward

        :param dones: 是否结束
        :param positions: 新生成的道路的最后一个点的位置（即智能体当前位置）
        :param parent_roads: 智能体出生的道路
        :param is_forwards: 是否向前
        :param debug_dict: 用于输出debug信息，可以留空。格式为{i: {} for i in range(len(positions))}
        :return:
        """
        # region [更新整体buffer]
        # 更新new roads observer
        self.observers['new_road_observer'].update_buffer(
            self.Road.get_roads_by_attr_and_value('state', RoadState.OPTIMIZING))
        # endregion
        # region [计算每个智能体的reward]
        out_rewards_list: list[float] = [0] * len(positions)
        for i in range(len(positions)):  # 对于每个智能体

            done = dones[i]
            if done:
                out_rewards_list[i] = 0
                if debug_dict:
                    debug_dict[i] = {REWARD_SUM: 0}
                continue  # 如果done
            # get values
            position = positions[i]
            parent_road = parent_roads[i]
            # print("is_forwards",is_forwards)
            is_forward = is_forwards[i]
            self.observers['parent_road_observer'].update_buffer(
                pd.DataFrame(parent_road).T  # pd.Series转成pd.Dataframe
            )

            # update prog
            for observer in self.observers.values():
                observer.update_observation_center(position)
            # render
            for observer in self.observers.values():
                observer.render()
            for post_processing in self.post_processings.values():
                post_processing.render()

            # calculate reward
            reward_dict = {
                BUILDING_REGION_REWARD: self.building_region_reward() * self.building_region_weight,
                ROAD_REWARD: self.road_reward() * self.road_weight,
                BOUND_REWARD: self.bound_reward() * self.bound_weight,
                BACKWARD_PENALTY: self.backward_penalty(is_forward) * self.back_weight,
                DEAD_NODE_PENALTY: self.deadnode_reward() * self.dead_node_weight,
                CROSS_NODE_PENALTY: self.crossnode_reward() * self.cross_node_weight,
                EXPLORATION_REWARD: self.exploration_reward() * self.exploration_weight,
            }

            reward: float = sum(list(reward_dict.values()))
            reward = np.clip(reward, 0, 250)
            reward = reward / 250.0  # 归一化到 [0, 1]

            reward_dict[REWARD_SUM] = reward  # 添加sum项至reward_dict
            out_rewards_list[i] = reward
            if debug_dict:
                debug_dict[i] = reward_dict
        out_rewards_list = np.array(out_rewards_list).reshape((-1, 1))
        return out_rewards_list

    def get_final_rewards(self,
                          road_agents: dict[uuid.UUID, pd.Series],
                          intersect_with_road_dict: dict[uuid.UUID, bool],
                          acute_angle_count_dict: dict[uuid.UUID, int],
                          debug_dict: Optional[dict] = None,
                          ):
        # 原始路网 vs 当前优化后的路网


        """
        :param road_agents:
        :param intersect_with_road_dict:
        :param debug_dict: 用于输出debug信息，可以留空。格式为{i: {} for i in range(len(road_agents))}
        :return:
        """
        self.observers['new_road_observer'].update_buffer(
            self.Road.get_roads_by_attr_and_value('state', RoadState.OPTIMIZING))
        result = np.zeros(len(road_agents), dtype=np.float32)
        # 计算 reward
        i = 0
        for uid in road_agents.keys():
            intersect_with_road = intersect_with_road_dict[uid]
            acute_count = acute_angle_count_dict[uid]

            if intersect_with_road:
                final_reward_dict = {
                    FINAL_DEFAULT: 1 * self.final_weight,
                    FINAL_ENDNODE_REWARD: self.endnode_final_reward() * self.node_final_weight,
                    FINAL_DISTANCE_REWARD: self.distance_final_reward(road_agents[uid]) * self.dist_weight,
                    FINAL_ANGLE_REWARD: self.acute_angle_final_reward(acute_count) * self.angle_weight,
                }

            else:
                final_reward_dict = {
                    FINAL_DEFAULT: 0,
                    FINAL_ENDNODE_REWARD: 0,
                    FINAL_DISTANCE_REWARD: 0,
                    FINAL_ANGLE_REWARD: 0,
                }
            reward = sum(list(final_reward_dict.values())) /200
            # reward = np.prod(list(final_reward_dict.values()))
            result[i] = reward
            i += 1
        # print("get_final_reward:",result)
        return result # 这返回的应该是一个列表把..


class RewardRoadNet:
    def __init__(self, origin_road_collection, new_road_collection):
        self.origin_road_collection = origin_road_collection
        self.new_road_collection = new_road_collection

        # self.G_origin = origin_road_collection.to_graph()
        # self.G_new = new_road_collection.to_graph()

        self.origin_efficiency = self.street_efficiency_reward(self.origin_road_collection )
        self.origin_density = self.network_density_reward(self.origin_road_collection )
        self.origin_continuity = self.street_continuity_reward(self.origin_road_collection )
        self.origin_bearing = self.street_bearing_reward(self.origin_road_collection )

    def street_efficiency_reward(self, road_collection):
        """平均出行效率（近似定义为道路连接距离/节点数）"""
        edge_gdf = road_collection.get_all_roads()
        node_gdf = road_collection.get_all_nodes()
        if len(node_gdf) == 0:
            return 0.0
        total_length = edge_gdf.geometry.length.sum() * 0.001  # 转换为km
        return total_length / len(node_gdf)

    def network_density_reward(self, road_collection):
        """单位面积的道路密度"""
        area = roadnet_bound_area(road_collection)  # 单位: 平方米
        edge_gdf = road_collection.get_all_roads()
        return len(edge_gdf) / area if area > 0 else 0.0


    def street_continuity_reward(self, road_collection):
        """平均道路长度"""
        edge_gdf = road_collection.get_all_roads()
        if len(edge_gdf) == 0:
            return 0.0
        lengths = edge_gdf.geometry.length
        return lengths.mean()

    def street_bearing_reward(self, road_collection):
        """方向有序性，计算所有道路线段的方向标准差"""
        from math import atan2, degrees
        from scipy.stats import circstd
        edge_gdf = road_collection.get_all_roads()
        bearings = []

        for line in edge_gdf.geometry:
            if line.length == 0 or len(line.coords) < 2:
                continue
            x1, y1 = line.coords[0]
            x2, y2 = line.coords[-1]
            dx = x2 - x1
            dy = y2 - y1
            angle = (degrees(atan2(dy, dx)) + 360) % 180  # 归一到 [0, 180)
            bearings.append(angle)

        if not bearings:
            return 0.0

        # 使用 circular std 计算方向离散度
        bearing_std = circstd(np.radians(bearings), high=np.pi, low=0)
        return 180 - np.degrees(bearing_std)  # 越高表示越有序

    def get_roadnet_rewards(self):
        new_efficiency = self.street_efficiency_reward(self.new_road_collection)
        new_density = self.network_density_reward(self.new_road_collection)
        new_continuity = self.street_continuity_reward(self.new_road_collection)
        new_bearing = self.street_bearing_reward(self.new_road_collection)
        # print("origin_efficiency",self.origin_efficiency,"new_efficiency",new_efficiency)
        # print("origin_density", self.origin_density, "new_density", new_density)
        # print("origin_continuity", self.origin_continuity, "new_continuity", new_continuity)
        # print("origin_bearing",self.origin_bearing,"new_bearing",new_bearing)
        # 比值形式
        # 相对改进值
        diff_eff = (new_efficiency - self.origin_efficiency) / (self.origin_efficiency + 1e-6)
        diff_density = (new_density - self.origin_density) / (self.origin_density + 1e-6)
        diff_continuity = (new_continuity - self.origin_continuity) / (self.origin_continuity + 1e-6)
        diff_bearing = (self.origin_bearing - new_bearing) / (self.origin_bearing + 1e-6)  # 越小越好

        # 归一化（缩放到 [0,1]）
        reward_eff = (np.tanh(diff_eff) + 1) / 2
        reward_density = (np.tanh(diff_density) + 1) / 2
        reward_continuity = (np.tanh(diff_continuity) + 1) / 2
        reward_bearing = (np.tanh(diff_bearing) + 1) / 2

        # 加权总和（总分仍在 [0,1] 范围）
        total_reward = (
                0.3 * reward_eff +
                0.3 * reward_density +
                0.2 * reward_continuity +
                0.2 * reward_bearing
        )
        return total_reward

def roadnet_bound_area(roads):
    """roads:roadcollection()"""
    region_min, region_max = roads.get_bbox()
    x1,y1 = region_min
    x2,y2 = region_max
    area = np.abs(x2 - x1) * np.abs(y2 - y1)
    return area