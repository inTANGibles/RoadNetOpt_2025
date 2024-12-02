import uuid
from enum import Flag, auto

import geopandas as gpd
import networkx as nx
import numpy as np

from geo import RoadCollection
from utils import RoadLevel
from utils.road_utils import ratio_bound_by_road_level


class RoadNetAnalyzerFlags(Flag):
    """用于界定计算指标时， 采用哪些指标"""
    NONE = auto()
    RATIO = auto()
    DENSITY = auto()
    EFFICIENCY = auto()
    MEANDEPTH = auto()
    INTEGRATION = auto()
    CHOICE = auto()


class RoadNetValues:
    """记录RoadNet的指标以及计算指标用到的参数"""

    def __init__(self, values: dict, args: dict):
        self.values = values
        self.args = args


class RoadNetAnalyzer:
    """整体路网的评估体系，分为长度、密度、连通性三大方面"""

    # region 基本计算
    @staticmethod
    def _get_bound_area(road_collection: RoadCollection):
        """roads:roadcollection()"""
        region_min, region_max = road_collection.get_bbox()
        x1, y1 = region_min
        x2, y2 = region_max
        area = np.abs(x2 - x1) * np.abs(y2 - y1)
        return area

    @staticmethod
    def _get_total_length(roads: gpd.GeoDataFrame) -> float:
        """roads_gdf:geodataframe"""
        length = 0
        for uid, road in roads.iterrows():
            RoadCollection.get_road_length_using_coords(road)
        return length

    # endregion

    # region 指标计算
    """
    TODO：根据规范和框架计算评价指标
    12.6.2次干路主要起交通集散作用，其里程占城市总道路里程的比例宜为5%-15% ✔
    12.6.3根据街区尺度判断路网密度，路网的总长度除以社区的总面积 ✔
    路网运行效率（连通性）--平均通行时间来评估，通行时间越短，路网效率越高。（平均通行时间） ✔ 可以再想想

    特定需求：提升某地的连通性时，空间句法integration（优先）✔ /choice ✔/depth ✔的指标提升
    """

    @classmethod
    def roadnet_length_ratio(cls, road_collection: RoadCollection, target_level=RoadLevel.PRIMARY):
        """
        获取某一个道路等级在所有道路中的长度占比 （FOOTWAY除外）
        建议基于全域（大范围）计算-->self.MapRoad
        :param road_collection: 计算范围
        :param target_level: 目标道路等级
        :return: 目标道路在计算范围道路中的百分比
        """
        assert target_level != RoadLevel.FOOTWAY, "target level不能是RoadLevel.FOOTWAY"
        all_roads_gdf = road_collection.get_roads_except_attr_and_value('level', RoadLevel.FOOTWAY)
        target_roads_gdf = road_collection.get_roads_by_attr_and_value('level', target_level)
        length_total = cls._get_total_length(all_roads_gdf)
        assert length_total > 0
        length_target = cls._get_total_length(target_roads_gdf)  # 集合中目标道路的总长
        return length_target / length_total

    @classmethod
    def roadnet_density(cls, road_collection: RoadCollection) -> float:
        """
        可以基于社区尺度计算-->self.NeighbourRoad
        """
        area = cls._get_bound_area(road_collection)
        assert area > 0
        length = cls._get_total_length(road_collection.get_all_roads())
        density = length / area
        return density

    @staticmethod
    def roadnet_efficiency(G: nx.Graph, mode='car'):
        """
        roads为RoadCollection()类
        基于全域map进行图论计算

        TODO: 把车行与人行区分出来,以及看看有没有更科学的方法
        if mode == 'car':
            map = roads.get_roads_except_attr_and_value('level', RoadLevel.FOOTWAY)
        else:
            map = roads.get_all_roads() # geodataframe转过去graph
        """

        # average = nx.average_shortest_path_length(G,weight='length',method=‘dijkstra’) #但是G unconnected
        all_shortest_paths_times = dict(nx.all_pairs_dijkstra_path_length(G, weight='time'))
        num_starts = len(all_shortest_paths_times.values())
        num_paths = sum(len(times) for times in all_shortest_paths_times.values()) - num_starts
        total_shortest_path_time = sum(sum(times.values()) for times in all_shortest_paths_times.values())
        average_shortest_path_time = total_shortest_path_time / num_paths
        total_time = 0
        for edge_data in G.edges(data=True):
            total_time += edge_data[2]['time']
        eff = total_time / average_shortest_path_time / len(G.edges)
        return eff

    @staticmethod
    def roadnet_target_meandepth(G: nx.Graph, target_node_uid: uuid.UUID, r: float = 750.0) -> tuple[float, float]:
        """
        指定节点'geometry'的拓扑深度,某一空间到达其它空间所需经过的最小连接数,
        td:拓扑总深度指从起始节点，沿最少拓扑深度的路径，到达其他（一定筛选半径范围内的）所有节点的拓扑深度之和。
        """
        target_node = G.nodes[target_node_uid]
        subgraph: nx.Graph = nx.ego_graph(G, target_node, radius=r, distance='length')
        path_lengths = dict(nx.single_source_dijkstra_path_length(subgraph, target_node, cutoff=None, weight='length'))
        td = sum(lengths for lengths in path_lengths.values())
        md: float = td / (len(subgraph.nodes()) - 1)
        return md, len(subgraph.nodes())

    @classmethod
    def roadnet_target_integration(cls, G, target_node_uid: uuid.UUID, r: float = 750.0) -> float:
        """指定节点'geometry'的整合度"""

        md, n = cls.roadnet_target_meandepth(G, target_node_uid, r)
        RA = (md - 1) / (n / 2 - 1)
        RAoD = (n * (np.log2(n / 3) - 1) + 1) / ((n - 2) * (n - 1) / 2)
        RRA = RA / RAoD
        inte = 1 / RRA
        return inte

    @staticmethod
    def roadnet_target_choice(G: nx.Graph, target_node_uid: uuid.UUID, r: float = 750.0) -> float:
        """
        指定节点'geometry'的选择度,
        穿行度(Choice)，表示系统中某一空间被其他最短路径穿行的可能性(以下直接统计次数)。
        """
        target_node = G.nodes[target_node_uid]
        subgraph = nx.ego_graph(G, target_node, radius=r, distance='length')
        all_shortest_paths = dict(nx.all_pairs_shortest_path(subgraph))
        node_choice = sum(1 for path in all_shortest_paths.values() if target_node in path and len(path[target_node]) > 1)
        return node_choice

    @classmethod
    def roadnet_global_meandepth(cls, G: nx.Graph, r: float = 750.0) -> float:
        """全局的mean depth"""
        return sum((cls.roadnet_target_meandepth(G, node_uid, r)[0] for node_uid in G.nodes)) / len(G.nodes)

    @classmethod
    def roadnet_global_integration(cls, G: nx.Graph, r: float = 750.0) -> float:
        return sum((cls.roadnet_target_integration(G, node_uid, r) for node_uid in G.nodes)) / len(G.nodes)

    @classmethod
    def roadnet_global_choice(cls, G: nx.Graph, r: float = 750.0) -> float:
        return sum((cls.roadnet_target_choice(G, node_uid, r) for node_uid in G.nodes)) / len(G.nodes)

    # endregion

    # region 计算路网的所有指标或给定指标

    @classmethod
    def get_roadnet_values(cls,
                           road_collection: RoadCollection,
                           target_level=None,
                           r=750,
                           flags: "RoadNetAnalyzerFlags" = RoadNetAnalyzerFlags.RATIO | RoadNetAnalyzerFlags.DENSITY | RoadNetAnalyzerFlags.EFFICIENCY | RoadNetAnalyzerFlags.MEANDEPTH
                           ) -> RoadNetValues:
        """第一步：获取某个road_collection的各项指标"""
        if RoadNetAnalyzerFlags.EFFICIENCY in flags or RoadNetAnalyzerFlags.MEANDEPTH in flags or RoadNetAnalyzerFlags.INTEGRATION in flags or RoadNetAnalyzerFlags.CHOICE in flags:
            G = road_collection.to_graph()
        else:
            G = None
        values = {}
        if RoadNetAnalyzerFlags.RATIO in flags:
            assert target_level is not None
            values['RATIO'] = cls.roadnet_length_ratio(road_collection, target_level)
        if RoadNetAnalyzerFlags.DENSITY in flags:
            values['DENSITY'] = cls.roadnet_density(road_collection)
        if RoadNetAnalyzerFlags.EFFICIENCY in flags:
            values['EFFICIENCY'] = cls.roadnet_efficiency(G)
        if RoadNetAnalyzerFlags.MEANDEPTH in flags:
            values['MEANDEPTH'] = cls.roadnet_global_meandepth(G, r)
        if RoadNetAnalyzerFlags.INTEGRATION in flags:
            values['INTEGRATION'] = cls.roadnet_global_integration(G, r)
        if RoadNetAnalyzerFlags.CHOICE in flags:
            values['CHOICE'] = cls.roadnet_global_choice(G, r)
        args = {'road_collection': road_collection, 'target_level': target_level, 'r': r, 'flags': flags}
        return RoadNetValues(values, args)

    # endregion

    # region 指标转化为分数(raise)
    WEIGHT_RATIO = 1
    WEIGHT_DENSITY = 1
    WEIGHT_EFFICIENCY = 1
    WEIGHT_MEANDEPTH = 1
    WEIGHT_INTEGRATION = 1
    WEIGHT_CHOICE = 1

    @staticmethod
    def _value_to_score(value: float, baseline: float, bound=None, inverse=False) -> float:
        """
        定义一个value 转score的通用模板
        :param value:
        :param baseline: 基准值， 即原来的self.or_pri_ratio
        :param bound:
        :param inverse:
        :return:
        """
        if bound is not None:
            if value < bound[0] or value > bound[1]:
                return -1
        raise_ratio = (value - baseline) / baseline
        if inverse:
            raise_ratio *= -1
        return raise_ratio * 10

    @classmethod
    def get_roadnet_score(cls, values: RoadNetValues, baseline: RoadNetValues) -> dict[str:float]:
        """第二步：将指标值转化为得分"""
        values1 = values.values
        values2 = baseline.values
        args1 = values.args
        args2 = baseline.args
        assert set(values1.keys()) == set(values2.keys())
        assert args1['r'] == args2['r']
        assert args1['target_level'] == args2['target_level']
        flags = args1['flags']
        target_level = args1['target_level']

        score = {}
        if RoadNetAnalyzerFlags.RATIO in flags:
            score['RATIO'] = cls._value_to_score(values1['RATIO'], baseline=values2['RATIO'], bound=ratio_bound_by_road_level[target_level])
        if RoadNetAnalyzerFlags.DENSITY in flags:
            score['DENSITY'] = cls._value_to_score(values1['DENSITY'], baseline=values2['DENSITY'], bound=(0, 1000))
        if RoadNetAnalyzerFlags.EFFICIENCY in flags:
            score['EFFICIENCY'] = cls._value_to_score(values1['EFFICIENCY'], baseline=values2['EFFICIENCY'])
        if RoadNetAnalyzerFlags.MEANDEPTH in flags:
            score['MEANDEPTH'] = cls._value_to_score(values1['MEANDEPTH'], baseline=values2['MEANDEPTH'], inverse=True)
        if RoadNetAnalyzerFlags.INTEGRATION in flags:
            score['INTEGRATION'] = cls._value_to_score(values1['INTEGRATION'], baseline=values2['INTEGRATION'])
        if RoadNetAnalyzerFlags.CHOICE in flags:
            score['CHOICE'] = cls._value_to_score(values1['CHOICE'], baseline=values2['CHOICE'])

        return score
    # endregion
