import logging
import math
import traceback
import uuid
from enum import Enum
from typing import Union, Optional
from reward.reward_agent import RewardAgent, RewardRoadNet
import geopandas as gpd
import numpy as np
import pandas as pd
from geo.road import  RoadCollection
import graphic_module
from DDPG.utils.my_timer import MyTimer
from geo import Road, Building, Region
from graphic_module import GraphicManager
from gui import global_var as g
from style_module import StyleManager
from utils import RoadState, RoadLevel
from utils import point_utils, io_utils
import copy

print('env2 loaded')

__version__ = 200


class RenderBackend(Enum):
    """渲染后端"""
    MATPLOTLIB = 0
    OPENGL = 1


class RoadEnv:
    def __init__(self, num_road_agents: int,
                 max_episode_step: int,
                 region_min: Optional[tuple],
                 region_max: Optional[tuple],
                 observation_img_size: tuple[int, int],
                 observation_view_size: tuple[float, float],
                 action_step_range: tuple[float, float],
                 headless: bool,
                 shared_data: Optional[dict],
                 render_backend: RenderBackend = RenderBackend.OPENGL,
                 observation_center: tuple[float, float] = (0, 0),  # train3版本以后加入
                 still_mode: bool = False,  # train3版本以后加入
                 road_level: list[RoadLevel] = None,  # train3版本以后加入
                 ):
        """
        智能体运行环境

        .. note::
           请在道路、建筑、区域等加载完毕后再创建该环境。

           （在创建环境时会自动缓存原有道路，并且在注册观察渲染器时会自动将现有的道路、建筑、区域等作为buffer传入，

           而这些buffer后期将不再更新， 因此必须在道路、建筑、区域等加载完毕后再创建环境）


        :param num_road_agents: 智能体数量
        :param max_episode_step: 最大行动步数
        :param region_min: 可活动区域的左上角顶点坐标（世界坐标）
        :param region_max: 可活动区域的右下角坐标（世界坐标）
        :param observation_img_size: 观察图像的像素大小
        :param observation_view_size: 观察者的视野范围（世界坐标）
        :param observation_center: 观察的中心坐标， 仅在still_mode==True时生效
        :param still_mode: 摄像机是否静止，设为True时观察范围和位置固定， 并采用observation_center参数所提供的坐标作为观察中心， 否则摄像机跟随agent运动
        :param action_step_range: 步长最小值，步长最大值
        :param headless: 无头模式
        :param shared_data: 共享的数据
        :param render_backend: 渲染后端
        :param road_level: 生成智能体的道路等级
        """
        # [parameters 参数]
        print("WHAT IS ROAD:",Road)

        self.num_road_agents: int = num_road_agents
        self.max_episode_step: int = max_episode_step
        if region_min is None or region_max is None:
            region_min, region_max = Road.get_bbox()
        self.region_min = region_min
        self.region_max = region_max
        print(f"region_min = {region_min}, region_max = {region_max}")
        self.observation_img_size: tuple[int, int] = observation_img_size  # 观察者的图像像素大小
        self.observation_view_size: tuple[float, float] = observation_view_size  # only work in OPENGL mode
        self.observation_center: tuple[float, float] = None if not still_mode else observation_center
        self.still_mode: bool = still_mode
        self.headless: bool = headless
        self.road_levels: list[RoadLevel] = road_level if road_level is not None else [
                                                                                          RoadLevel.TERTIARY] * self.num_road_agents
        assert len(self.road_levels) == self.num_road_agents, "road level数量与智能体数量不匹配"
        if 'reward_info' not in shared_data:
            shared_data['reward_info'] = {i: {} for i in range(num_road_agents)}
        if 'final_reward_info' not in shared_data:
            shared_data['final_reward_info'] = {i: {} for i in range(num_road_agents)}
        self.shared_data: dict = shared_data
        self.render_backend: RenderBackend = render_backend if g.mWindowEvent is not None else RenderBackend.MATPLOTLIB
        _l = (action_step_range[1] - action_step_range[0]) / 2
        _b = action_step_range[1] - _l
        self.action_space_bound: np.ndarray = np.array([(math.pi, _l)])  # [((-π~π), (-l~l))]
        self.action_space_boundMove: np.ndarray = np.array([(0, _b)])  # [((-π~π), (-l + b, l + b))]

        # [variables 变量]
        self.episode_step: int = 0  # 当前步数
        self.raw_roads: gpd.GeoDataFrame = Road.get_all_roads()  # 保存原有道路， 分裂后的新道路不在其中
        self.road_agents: dict[uuid.UUID: pd.Series] = {}  # 所有的新的道路智能体
        self.agents_done: dict[uuid.UUID, bool] = {}  # 个体是否完成
        self.agents_forwards: dict[uuid.UUID, bool] = {}  # 个体是否向前
        self.agents_acute_count: dict[uuid.UUID, int] = {}  # 过程中个体形成锐角的次数
        self.agent_parent: dict[uuid.UUID, pd.Series] = {}  # 智能体的父对象（道路）
        self.agent_out_of_region: dict[uuid.UUID, bool] = {}  # 智能体是否超出边界
        self.agent_intersect_with_road: dict[uuid.UUID, bool] = {}  # 智能体是否与其他道路相撞
        Road.cache()  # 保存当前路网状态，以备复原

        # [init 初始化]
        if render_backend == RenderBackend.OPENGL:
            # 如果渲染后端为OpenGL， 则初始化观察者渲染器
            self._init_observers(*observation_img_size)
            self._init_reward_agent()
        else:
            raise Exception('渲染后端采用matplotlib ，已在版本0.5.2及以后的版本中不再受支持')

    def _init_observers(self, width, height):
        """
        初始化观察者渲染器

        创建原有道路， 建筑， 区域的observer渲染器

        observer本质上是一个FrameBufferTexture， 提供以自身坐标为中心的渲染范围，详细信息参见graphic_module.py 的Observer类

        关于raw_roads_observer、building_observer、region_observer这里获取的道路做了一次三角化后，就不会更新图像了，

        但是如果原始道路被删除了，或者图像视觉上发生了变化，就需要update buffer

        跟随模式下，多个agent可以公用一个observer， 只需要set_observation_space， 即可渲染出不同的图像
        """
        # 创建observer渲染器
        self.raw_roads_observer = graphic_module.RoadObserver(
            'raw_roads_obs', width, height, self.observation_view_size, self.raw_roads)
        self.new_roads_observer = graphic_module.RoadObserver(
            'new_roads_obs', width, height, self.observation_view_size, None
        )
        self.building_observer = graphic_module.BuildingObserver(
            'building_obs', width, height, self.observation_view_size, Building.get_all_buildings()
        )
        self.region_observer = graphic_module.RegionObserver(
            'region_obs', width, height, self.observation_view_size, Region.get_all_regions()
        )
        self.bound_observer = graphic_module.RegionObserver(
            'bound_obs', width, height, self.observation_view_size,
            initial_gdf=Region.create_region_by_min_max(self.region_min, self.region_max),
            sf=StyleManager.I.env.region_simple_style_factory  # colored white
        )
        self.node_observer = graphic_module.NodeObserver(
            'node_obs', width, height, self.observation_view_size, Road.get_all_nodes(),
            road_collection=Road
        )
        self.blend_observer = graphic_module.ObsBlender(
            'blend_obs', width, height
        )  # blend_observer是一个特殊的后处理渲染器，用于将其余渲染器的图像数据合并，因此不需要设置观察范围和位置
        if self.still_mode:
            # 如果是静止模式，则在一开始分配好观察摄像机位置
            self.raw_roads_observer.update_observation_center(self.observation_center)
            self.new_roads_observer.update_observation_center(self.observation_center)
            self.building_observer.update_observation_center(self.observation_center)
            self.region_observer.update_observation_center(self.observation_center)
            self.bound_observer.update_observation_center(self.observation_center)
            self.node_observer.update_observation_center(self.observation_center)
            # 如果是静止模式，则在一开始先渲染好不变的物体
            self.raw_roads_observer.render()
            self.building_observer.render()
            self.region_observer.render()
            self.bound_observer.render()
            self.node_observer.render()
        # 将observer 注册到GraphicManager，以在gui中显示
        GraphicManager.I.register_observer(self.blend_observer)
        GraphicManager.I.register_observer(self.raw_roads_observer)
        GraphicManager.I.register_observer(self.new_roads_observer)
        GraphicManager.I.register_observer(self.building_observer)
        GraphicManager.I.register_observer(self.region_observer)
        GraphicManager.I.register_observer(self.bound_observer)
        GraphicManager.I.register_observer(self.node_observer)

    def _init_reward_agent(self):
        """
        初始化reward系统
        """

        self.reward_agent = RewardAgent(region_min=self.region_min,
                                        region_max=self.region_max,
                                        headless=self.headless)

        # self.reward_roadnet = RewardRoadNet(origin_road_collection=self.original_road_collection,
        #                                     new_road_collection=self.new_road_collection) #TODO:是不是这样赋值啊！
    @property
    def done_in_ndarray_format(self) -> np.ndarray:
        """
        兼容性操作， 返回符合train要求的numpy格式的done

        注意：该操作不是计算done的方法，仅仅是将self.agents_done翻译为numpy格式
        """
        return np.array(list(self.agents_done.values())).reshape((self.num_road_agents, 1))

    def reset(self) -> np.ndarray:
        """初始化新的道路，分随机初始化、选定道路初始化(TODO)"""
        Road.restore()  # 复原路网
        self.original_road_collection=Road.copy()
        print("road_len" ,len(self.original_road_collection.get_all_roads()))
        self.raw_roads = Road.get_all_roads()
        self.episode_step = 0  # 重置步数
        self.clear_and_spawn_agents()  # 生成智能体
        return self.get_image_observation()

    def clear_and_spawn_agents(self):
        """
        清除原来的roads， 并且生成新的road（起点）

        更新self.road_agents 和 self.agents_done
        """
        self.road_agents = {}
        self.agents_done = {}
        self.agents_forwards = {}
        self.agent_parent = {}
        self.agent_out_of_region = {}
        self.agent_intersect_with_road = {}

        selected_road_uids = set()
        count = 0  # 这个变量是统计while 循环的次数的，防止始终找不到合适的路而陷入无限循环
        num_roads = 0  # 已成功创建的道路agent数量
        while num_roads < self.num_road_agents:
            count += 1
            if count > 100:
                logging.warning(f'count > 100')
                break  # 如果很多轮依旧找不满合适的路，则停止
            random_road = self.raw_roads.sample().iloc[0]  # 在原始的路网中随机一条路
            # random_road = self.raw_roads.iloc[0]  # 固定随机路网用以测试
            if random_road['uid'] in selected_road_uids:
                continue  # 如果随机到的路已经被选中了，则重新选
            if random_road['uid'] not in Road.get_all_roads()['uid'].values:
                continue
            selected_road_uids.add(random_road['uid'])  # 将随机到的路加入已被选择的路的set
            spawn_point = Road.interpolate_road_by_random_position(random_road)
            if spawn_point is None:
                continue  # 如果找不到符合路网间距规范的点，则重新选一条路
            spawn_point = spawn_point.reshape(-1, 2)
            Road.split_road_by_coord(random_road, spawn_point)  # 在路上随机一个点并尝试分裂
            road_level = self.road_levels[num_roads]
            road_state = RoadState.OPTIMIZING
            uid = Road.add_road_by_coords(spawn_point, road_level, road_state)  # 生成新路
            new_road = Road.get_road_by_uid(uid)

            self.road_agents[uid] = new_road  # 将新路加入self.agents
            self.agents_done[uid] = False  # 初始默认done的状态为False
            self.agents_forwards[uid] = True  # 初始默认forward的状态为True
            self.agents_acute_count[uid] = 0  # 初始默认锐角次数为0
            self.agent_parent[uid] = random_road
            num_roads += 1

    def clear_and_spawn_agents2(self):
        """
        清除原来的roads， 并且生成新的road（起点）

        更新self.road_agents 和 self.agents_done
        """
        self.road_agents = {}
        self.agents_done = {}
        self.agents_forwards = {}
        self.agent_parent = {}
        self.agent_out_of_region = {}
        self.agent_intersect_with_road = {}

        a = Road.get_roads_by_attr_and_value("level", RoadLevel.TERTIARY)
        b = Road.get_roads_by_attr_and_value("level", RoadLevel.FOOTWAY)
        road_candidates = pd.concat([a, b], axis=0)  # 符合条件的可以被删除的候选道路
        assert len(road_candidates) >= self.num_road_agents, "可删除道路数量小于要生成的智能体数量"
        roads_to_delete = road_candidates.sample(n=self.num_road_agents, replace=False)  # 从可被删除的道路中选择n条道路作为将要被删除的道路
        for i, road_to_delete in roads_to_delete.iterrows():

            Road.delete_road_by_uid(road_to_delete['uid'])
            start_point = road_to_delete['coords'][0].reshape(-1, 2)

            road_level = road_to_delete['level']
            road_state = RoadState.OPTIMIZING
            uid = Road.add_road_by_coords(start_point, road_level, road_state)  # 生成新路
            new_road = Road.get_road_by_uid(uid)

            self.road_agents[uid] = new_road  # 将新路加入self.agents
            self.agents_done[uid] = False  # 初始默认done的状态为False
            self.agents_forwards[uid] = True  # 初始默认forward的状态为True
            node_uid = road_to_delete['u']
            try:
                node = Road.get_node_by_uid(node_uid)
                self.agent_parent[uid] = Road.get_roads_by_node(node).iloc[0]  # TODO 将self.agent_parent改成支持dataframe格式
            except:
                self.agent_parent[uid] = None

    def get_image_observation(self) -> np.ndarray:
        """
        返回所有 agent 的状态，为图像格式（N, C, H, W）
        """
        if self.render_backend == RenderBackend.MATPLOTLIB:
            raise Exception('使用MATPLOTLIB获取observation image已不受支持，请使用OPENGL渲染后端')
        # OPENGL backend
        last_points = self._get_last_points()

        # if len(last_points) > 2 and not self.still_mode:
        #     logging.warning('当前模式为跟随模式，当前神经网络的state数据暂不支持多个智能体的观察空间，'
        #                     '而检测到环境的智能体数量大于1， 默认取第一个智能体的观察空间作为state，'
        #                     '要支持多个智能体，请将still mode设为True')
        # last_point = last_points[0]  # 对于多个agent，目前只支持第一个
        with MyTimer('observation_get_road_gdf', level=5):
            # TODO: 通过Road.get_roads_by_attr_and_value筛选正在优化的道路的操作仍是耗时操作，后期可以优化
            road_gdf = Road.get_roads_by_attr_and_value('state', RoadState.OPTIMIZING)
        with MyTimer('observation_update_buffer', level=5):
            self.new_roads_observer.update_buffer(road_gdf)


        images=[]
        for i in range(self.num_road_agents):
            if not self.still_mode:
                center = last_points[i]
                self.raw_roads_observer.update_observation_center(center)
                self.new_roads_observer.update_observation_center(center)
                self.building_observer.update_observation_center(center)
                self.region_observer.update_observation_center(center)
                self.bound_observer.update_observation_center(center)
                self.node_observer.update_observation_center(center)

                self.raw_roads_observer.render()
                self.building_observer.render()
                self.region_observer.render()
                self.bound_observer.render()
                self.node_observer.render()

            self.new_roads_observer.render()
            self.blend_observer.render([
                self.bound_observer.texture,
                self.region_observer.texture,
                self.building_observer.texture,
                self.raw_roads_observer.texture,
                self.new_roads_observer.texture,
                self.node_observer.texture,
            ])

            with MyTimer('observation_read_fbo', level=5):
                image_data = self.blend_observer.get_render_img()
                images.append(image_data.transpose((2, 0, 1)))  # to (C, H, W)
        # if not self.still_mode:
        #     # 如果为跟随模式，则更新摄像机位置
        #     self.raw_roads_observer.update_observation_center(last_point)
        #     self.new_roads_observer.update_observation_center(last_point)
        #     self.building_observer.update_observation_center(last_point)
        #     self.region_observer.update_observation_center(last_point)
        #     self.bound_observer.update_observation_center(last_point)
        #     self.node_observer.update_observation_center(last_point)
        # with MyTimer('observation_render', level=5):
        #     if not self.still_mode:
        #         # 如果为跟随模式，则重新渲染原有道路、建筑和区域
        #         # 如果是静止模式，则直接使用init时渲染好的texture即可
        #         self.raw_roads_observer.render()
        #         self.building_observer.render()
        #         self.region_observer.render()
        #         self.bound_observer.render()
        #         self.node_observer.render()
        #     # 无论是跟随模式还是静止模式，都需要渲染new roads和混合图层
        #     self.new_roads_observer.render()
        #     self.blend_observer.render(
        #         [self.bound_observer.texture,
        #          self.region_observer.texture,
        #          self.building_observer.texture,
        #          self.raw_roads_observer.texture,
        #          self.new_roads_observer.texture,
        #          self.node_observer.texture]  # 这里的顺序需要和shader中的texture的顺序对应
        #     )
        # with MyTimer('observation_read_fbo', level=5):
        #     # 从显存读取图片数据为numpy array
        #     image_data = self.blend_observer.get_render_img()
        # return image_data

        return np.stack(images)  # shape: (N, C, H, W)

    def get_org_image_observation(self):
        pass

    def render(self):
        raise NotImplementedError


    # 智能体动作action的更新逻辑
    def step(self, action):
        """
        返回new_observation, rewards, done, all_done
        :param action: 输入的action的范围为[-1, 1]
        :return:
        """
        action = action * self.action_space_bound + self.action_space_boundMove
        with MyTimer('step_move', level=4):
            dx = np.reshape(np.cos(action[:, 0]) * action[:, 1], (-1, 1))
            dy = np.reshape(np.sin(action[:, 0]) * action[:, 1], (-1, 1))
            moves = np.concatenate((dx, dy), axis=1)
            # 给每条路添加新的 最末点， 以此使路网生长
            for i, uid in enumerate(self.road_agents.keys()):
                if self.agents_done[uid]: continue  # 如果该道路已经停止，则不再添加
                lst_pt = Road.get_road_last_point(self.road_agents[uid])  # 获取道路的最后一个点
                new_pt = lst_pt + moves[i].reshape(1, 2)  # 根据move计算新的点的位置
                self.road_agents[uid] = Road.add_point_to_road(self.road_agents[uid], point=new_pt)  # 向道路添加点
        with MyTimer('step_cal_is_done', level=4):
            for i, uid in enumerate(self.road_agents.keys()):
                if self.agents_done[uid]: continue
                self.agents_done[uid] = self._is_agent_done(uid)  # 计算每条路是否结束
                forward_bool = self._is_way_forward(uid)  # 判断每条路是否往前
                self.agents_forwards[uid] = forward_bool
                if not forward_bool:
                    self.agents_acute_count[uid] += 1
                self.agent_intersect_with_road[uid] = self._is_intersect_with_raw_roads(uid)
        with MyTimer('step_get_img_observation', level=4):
            new_observation_img = self.get_image_observation()
        with MyTimer('step_get_reward', level=4):
            reward = self.calculate_reward()  # 计算每一步的reward

        done = self.done_in_ndarray_format  # 转换为numpy格式
        all_done = self._all_done()  # 是否所有都结束了
        self.episode_step += 1  # 更新当前的步数
        # return new_observation_img.transpose((2, 0, 1)), reward, done, all_done
        return new_observation_img, reward, done, all_done
    def calculate_reward(self):
        """在这里计算每一步的reward"""


        reward_all = self.reward_agent.get_rewards(
            dones=self.done_in_ndarray_format,
            positions=self._get_last_points(),
            parent_roads=list(self.agent_parent.values()),
            is_forwards=list(self.agents_forwards.values()),
            debug_dict=self.shared_data['reward_info']
        )
        return reward_all  # 单条路结束了，此时该路reward就为0

    def calculate_final_reward(self):
        """在这里计算最终的reward"""

        self.new_road_collection = Road

        if self.render_backend == RenderBackend.MATPLOTLIB:
            raise NotImplemented('尚未支持MATPLOTLIB的final reward的计算')
        else:
            agent_reward = self.reward_agent.get_final_rewards(
                self.road_agents,
                self.agent_intersect_with_road,
                self.agents_acute_count,
                debug_dict=self.shared_data['final_reward_info']
            )
            # print("is this two roads different?", self._print_road_collections_diff())
            roadnet_reward= RewardRoadNet(
                origin_road_collection=self.original_road_collection,
                new_road_collection=self.new_road_collection
            )
            roadnet_reward_vaule=roadnet_reward.get_roadnet_rewards()

            # print("agent_reward:",agent_reward)
            # print('roadnet_reward:',roadnet_reward_vaule)
            agent_reward += roadnet_reward_vaule
            return agent_reward

    def _print_road_collections_diff(self):
        print("[RoadNet Comparison]")
        print("- Original Road Collection:")
        print(f"  Type: {type(self.original_road_collection)}")
        print(f"  Total Roads: {len(self.original_road_collection.get_all_roads())}")
        print(f"  Sample: {self.original_road_collection.get_all_roads().head(1)}\n")

        print("- New Road Collection:")
        print(f"  Type: {type(self.new_road_collection)}")
        print(f"  Total Roads: {len(self.new_road_collection.get_all_roads())}")
        print(f"  Sample: {self.new_road_collection.get_all_roads().head(1)}\n")

        if self.original_road_collection.get_all_roads().equals(self.new_road_collection.get_all_roads()):
            print("=> These two road collections are IDENTICAL.")
        else:
            print("=> These two road collections are DIFFERENT.")
    def _get_last_points(self) -> np.ndarray:
        """获取所有agent道路的最后一个点，返回[n, 2]形状的np array"""
        last_points = []
        for i, road in enumerate(self.road_agents.values()):
            last_points.append(Road.get_road_last_point(road))
        return np.vstack(last_points)



    def _is_in_region(self, uid) -> bool:
        """判断uid编号的道路的是否在区域内。该函数仅对最后一个点有效，因此需要每步调用"""
        lst_pt = tuple(Road.get_road_last_point(self.road_agents[uid])[0])
        in_region = True
        in_region &= self.region_min[0] < lst_pt[0] < self.region_max[0]
        in_region &= self.region_min[1] < lst_pt[1] < self.region_max[1]
        return in_region

    def _is_way_forward(self, uid) -> bool:
        """判断uid编号的道路是否向前运动。需要每步调用"""
        coords = list(self.road_agents[uid]['geometry'].coords)
        if len(coords) < 3:
            return True
        vec1 = point_utils.vector_from_points(coords[-2], coords[-1])
        vec2 = point_utils.vector_from_points(coords[-3], coords[-2])
        return point_utils.vector_dot(vec1, vec2) > 0

    def _is_intersect_with_raw_roads(self, uid):
        """判断uid编号的道路是否与原始路网相交。该函数仅对最后一段线段有效，因此需要每步调用"""
        road = self.road_agents[uid]
        coords = list(self.road_agents[uid]['geometry'].coords)
        if len(coords) < 2:
            return False  # 如果线段数小于1， 即点数小于2，则不做判断
        last_element = Road.get_road_last_element(road).buffer(1e-5)
        intersects = self.raw_roads['geometry'].intersects(last_element)
        return intersects.sum() > 0  # 由于判断的是最后一段线段，因此只要大于0就是相交，无需考虑起点和原始路径的相交问题

    def _is_agent_done(self, uid) -> bool:
        """判断uid的道路是否完成"""
        if self.agents_done[uid]: return True  # 如果在agents_done中已经标记为完成，则直接返回完成
        if self.episode_step >= self.max_episode_step: return True  # 如果达到最大步数，则返回完成
        done = False
        done |= not self._is_in_region(uid)
        # # done |= not self._is_way_forward(uid)
        done |= self._is_intersect_with_raw_roads(uid)
        return done

    def _all_done(self):
        """是否所有的agent都完成了"""
        return all(self.agents_done.values())


mRoadNet: Union[RoadEnv, None] = None
mRewardSum = 0

mTargetOptimizedAgentNum = 0  # 仅限顺序模式
mCurrentOptimizedAgentNum = 0  # 仅限顺序模式


def from_gdf(edge_gdf: gpd.GeoDataFrame) -> RoadCollection:
    rc = RoadCollection()
    rc.add_roads(edge_gdf)
    return rc

def synchronous_mode_init(num_agents):
    """同步模式，若干agent同时跑"""
    global mRoadNet
    _ = io_utils.load_data('../data/VirtualEnv/0312_ty.bin')
    Building.data_to_buildings(_)
    Region.data_to_regions(_)
    Road.data_to_roads(_)

    mRoadNet = RoadEnv(num_agents)


def synchronous_mode_reset():
    global mRoadNet, mRewardSum
    mRoadNet.reset()
    mRoadNet.render()
    mRewardSum = 0
    print('road net reset')


def synchronous_mode_step(_) -> bool:
    global mRoadNet, mRewardSum
    try:
        print(f'当前轮次 {mRoadNet.episode_step}======================')
        action_list = []
        b = mRoadNet.action_space_bound
        c = mRoadNet.action_space_boundMove
        for i in range(len(mRoadNet.road_agents)):
            a = np.random.uniform(low=-1, high=1, size=(2,))
            _action = a * b + c
            action_list.append(_action)
        action = np.vstack(action_list)
        print(f'action {action}')
        next_state, reward, done, all_done = mRoadNet.step(action)
        mRewardSum += reward

        print(f'当前奖励 {reward}')
        print(f'当前累计奖励 {mRewardSum}')
        print(f'单路是否结束 {list(done)}')
        print(f'总体路网是否结束 {all_done}')
        print('==================================')
        mRoadNet.render()
        return all_done

    except Exception as e:
        print(e)
        traceback.print_exc()
        return True


def sequential_mode_init(num_agents):
    """顺序模式， agent一个一个跑"""
    global mRoadNet, mTargetOptimizedAgentNum
    _ = io_utils.load_data('../data/VirtualEnv/0312_ty.bin')
    Building.data_to_buildings(_)
    Region.data_to_regions(_)
    Road.data_to_roads(_)
    mRoadNet = RoadEnv(1)
    mTargetOptimizedAgentNum = num_agents


def sequential_mode_reset():
    global mRoadNet, mRewardSum, mCurrentOptimizedAgentNum

    mRoadNet.reset()
    mRoadNet.render()
    mRewardSum = 0
    mCurrentOptimizedAgentNum = 0
    print('road net reset')


def sequential_mode_step(_) -> bool:
    global mRoadNet, mRewardSum, mCurrentOptimizedAgentNum
    if mCurrentOptimizedAgentNum >= mTargetOptimizedAgentNum:
        return True

    print(f'当前轮次 {mRoadNet.episode_step}======================')

    b = mRoadNet.action_space_bound
    c = mRoadNet.action_space_boundMove
    a = np.random.uniform(low=-1, high=1, size=(2,))
    action = a * b + c
    print(f'action {action}')
    next_state, reward, done, all_done = mRoadNet.step(action)
    mRewardSum += reward

    print(f'当前奖励 {reward}')
    print(f'当前累计奖励 {mRewardSum}')
    print(f'单路是否结束 {list(done.values())}')
    print(f'总体路网是否结束 {all_done}')
    print('==================================')
    mRoadNet.render()

    if all_done:  # 这里一个单智能体的完成就会all done ，但不代表整体完成
        mRoadNet.clear_and_spawn_agents()
        mCurrentOptimizedAgentNum += 1

    return False


if __name__ == '__main__':
    pass
