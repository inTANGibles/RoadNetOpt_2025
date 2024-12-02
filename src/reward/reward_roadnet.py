import numpy as np
from geo import Road,RoadCollection
from utils import RoadLevel
import networkx as nx

class AnalysisRoadNet:
    """整体路网的评估体系，分为长度、密度、连通性三大方面"""

    def __init__(self,road_collection=None,map_collection=None,target_in_map=None):
        """
        target传入的是指定节点的'geometry'
        target_index:可选integration（优先）/choice/depth
        """
        if road_collection is None: road_collection = Road
        if map_collection is None: map_collection = road_collection

        self.NeighbourRoad = road_collection # 生成街区的路网,先基于这个做吧
        self.MapRoad = map_collection
        self.G = self.MapRoad.to_graph()
        self.target = target_in_map # uid

    # 基本计算
    def roadnet_bound_area(self,roads):
        """roads:roadcollection()"""
        region_min, region_max = roads.get_bbox()
        x1,y1 = region_min
        x2,y2 = region_max
        area = np.abs(x2 - x1) * np.abs(y2 - y1)
        return area

    def roadnet_total_length(self,roads_gdf):
        """roads_gdf:geodataframe"""
        length = 0
        for uid, road in roads_gdf.iterrows():
            length += self.MapRoad.get_road_length_using_coords(road)
        return length

    """
    TODO：根据规范和框架计算评价指标
    12.6.2次干路主要起交通集散作用，其里程占城市总道路里程的比例宜为5%-15% ✔
    12.6.3根据街区尺度判断路网密度，路网的总长度除以社区的总面积 ✔
    路网运行效率（连通性）--平均通行时间来评估，通行时间越短，路网效率越高。（平均通行时间） ✔ 可以再想想

    特定需求：提升某地的连通性时，空间句法integration（优先）✔ /choice ✔/depth ✔的指标提升
    """

    # 指标计算
    def roadnet_length_ratio(self,type = RoadLevel.PRIMARY,roads=None):
        """
        roads为RoadCollection()类
        建议基于全域（大范围）计算-->self.MapRoad
        """
        if roads is None: roads = self.MapRoad
        all_roads_gdf = roads.get_roads_except_attr_and_value('level', RoadLevel.FOOTWAY)
        primary_roads_gdf = roads.get_roads_by_attr_and_value('level', type)
        length_total = self.roadnet_total_length(all_roads_gdf)
        length_PRIMARY = self.roadnet_total_length(primary_roads_gdf) # 集合中次干道的总长
        return length_PRIMARY/length_total

    def roadnet_density(self,roads=None):
        """
        roads为RoadCollection()类
        可以基于社区尺度计算-->self.NeighbourRoad
        """
        if roads is None:roads = self.NeighbourRoad
        area = self.roadnet_bound_area(roads)
        length = self.roadnet_total_length(roads.get_all_roads())
        density = length/area
        return density

    def roadnet_efficiency(self,roads=None,mode='car'):
        """
        roads为RoadCollection()类
        基于全域map进行图论计算

        TODO把车行与人行区分出来,以及看看有没有更科学的方法
        if mode == 'car':
            map = roads.get_roads_except_attr_and_value('level', RoadLevel.FOOTWAY)
        else:
            map = roads.get_all_roads() # geodataframe转过去graph
        """
        if roads is not None:
            G = roads.to_graph()
        else:
            # roads = self.MapRoad
            G = self.G
        # average = nx.average_shortest_path_length(G,weight='length',method=‘dijkstra’) #但是G unconnected
        all_shortest_paths_times = dict(nx.all_pairs_dijkstra_path_length(G, weight='time'))
        num_starts = len(all_shortest_paths_times.values())
        num_paths = sum(len(times) for times in all_shortest_paths_times.values())-num_starts
        total_shortest_path_time = sum(sum(times.values()) for times in all_shortest_paths_times.values())
        average_shortest_path_time = total_shortest_path_time/ num_paths
        total_time = 0
        for edge_data in G.edges(data=True):
            total_time += edge_data[2]['time']
        eff = total_time / average_shortest_path_time/len(G.edges)
        return eff

    def roadnet_target_meandepth(self,target_geometry,r=750):
        """
        指定节点'geometry'的拓扑深度,某一空间到达其它空间所需经过的最小连接数,
        td:拓扑总深度指从起始节点，沿最少拓扑深度的路径，到达其他（一定筛选半径范围内的）所有节点的拓扑深度之和。
        """
        if target_geometry is None:
            return None,None
        else:
            G = self.G
            target_node = 0
            for node, geometry in nx.get_node_attributes(G, 'geometry').items():
                if geometry is target_geometry:
                    target_node = node
                    break
            subgraph = nx.ego_graph(G, target_node, radius=r,distance='length')
            path_lengths = dict(nx.single_source_dijkstra_path_length(subgraph,node,cutoff=None, weight='length'))
            td = sum(lengths for lengths in path_lengths.values())
            md = td/(len(subgraph.nodes())-1)
            return md,subgraph

    def roadnet_target_integration(self,target_geometry,r=750):
        """指定节点'geometry'的整合度"""
        if target_geometry is None:
            return None
        else:
            md,subgraph = self.roadnet_target_meandepth(target_geometry, r)
            n = len(subgraph.nodes())
            RA = (md-1)/(n/2-1)
            RAoD = (n*(np.log2(n/3)-1)+1)/((n-2)*(n-1)/2)
            RRA = RA/RAoD
            inte = 1/RRA
            return inte

    def roadnet_target_choice(self,target_geometry,r=750):
        """
        指定节点'geometry'的选择度,
        穿行度(Choice)，表示系统中某一空间被其他最短路径穿行的可能性(以下直接统计次数)。
        """
        if target_geometry is None:
            return None
        else:
            G = self.G
            target_node = 0
            for node, geometry in nx.get_node_attributes(G, 'geometry').items():
                if geometry is target_geometry:
                    target_node = node
                    break
            subgraph = nx.ego_graph(G, target_node, radius=r,distance='length')
            all_shortest_paths = dict(nx.all_pairs_shortest_path(subgraph))
            node_choice = sum(1 for path in all_shortest_paths.values() if target_node in path and len(path[target_node]) > 1)
            return node_choice

    def get_all_indexes(self,target_index='integration'):
        pri_ratio = self.roadnet_length_ratio(self.MapRoad)
        density = self.roadnet_density(self.NeighbourRoad)
        eff = self.roadnet_efficiency(self.MapRoad)
        if target_index == 'integration':
            target_score = self.roadnet_target_integration(self.target)
        elif target_index == 'choice':
            target_score = self.roadnet_target_choice(self.target)
        elif target_index == 'depth':
            target_score,_ = self.roadnet_target_meandepth(self.target)
        else:
            target_score = None
        return pri_ratio, density, eff, target_score


class RewardRoadNet:
    """结合路网的评估体系，分配权重（AHP），根据与原来优化的比值，得到一个综合得分"""
    def __init__(self,orgin_AnalysisRoadNet,target=False,target_index = 'integration'):
        """
        target_index:可选integration（优先）/choice/depth"""
        self.target_index = target_index
        self.target = target
        # 各部分的权重参数,现在是随便设置的
        self.fullscore = 100
        if self.target:
            self.target_weight = 0.25 * self.fullscore
            self.length_ratio_weight = 0.25 * self.fullscore
            self.density_weight = 0.25 * self.fullscore
            self.connetivity_weight = 0.25 * self.fullscore
        else:
            self.target_weight = 0 * self.fullscore
            self.length_ratio_weight = 0.3 * self.fullscore
            self.density_weight = 0.4 * self.fullscore
            self.connetivity_weight = 0.3 * self.fullscore

        # 规范中允许的区间
        self.orgin_AnalysisRoadNet = orgin_AnalysisRoadNet
        self.pri_ratio_bound = (0.05, 0.15)
        self.density_bound = (8,)  # 先默认按长宽均小于300的居住区进行判断
        self.or_pri_ratio, self.or_density, self.or_eff, self.or_target_sc = orgin_AnalysisRoadNet.get_all_indexes(self.target_index)


    # 优化计算(计算思路为在符合规范的条件下,提高占原来的百分比，再放大十倍)
    def pri_ratio_raise(self,new_pri_ratio):
        pri_ratio_min, pri_ratio_max = self.pri_ratio_bound
        if new_pri_ratio < pri_ratio_min or new_pri_ratio > pri_ratio_max:
            return -1
        else:
            raise_ratio = (new_pri_ratio - self.or_pri_ratio)/self.or_pri_ratio
            return raise_ratio * 10

    def density_raise(self,new_density):
        density_min,density_max = self.density_bound
        if new_density < density_min or new_density > density_max:
            return -1
        else:
            raise_ratio = (new_density - self.or_density)/self.or_density
            return raise_ratio * 10

    def eff_raise(self,new_eff):
        raise_ratio = (new_eff - self.or_eff) / self.or_eff
        return raise_ratio * 10

    def target_ratio(self,new_target_sc):
        if self.target_index == 'depth':
            raise_ratio = -(new_target_sc - self.or_target_sc) / self.or_target_sc
        else:
            raise_ratio = (new_target_sc - self.or_target_sc) / self.or_target_sc
        return raise_ratio * 10

    # 得分计算
    def get_roadnet_rewards(self,new_AnalysisRoadNet):
        """
        warning!!!以下代码没有测试过可能会有bug
        """
        new_pri_ratio, new_density, new_eff, new_target_sc = new_AnalysisRoadNet.get_all_indexes('self.target_index')
        if self.target:
            reward_dict = {
                'pri_ratio':self.pri_ratio_raise(new_pri_ratio)*self.length_ratio_weight,
                'density':self.density_raise(new_density)*self.density_weight,
                'eff':self.eff_raise(new_eff)*self.connetivity_weight,
                'target_sc':self.target_ratio(new_target_sc)*self.target_weight,
            }
        else:
            reward_dict = {
                'pri_ratio':self.pri_ratio_raise(new_pri_ratio)*self.length_ratio_weight,
                'density':self.density_raise(new_density)*self.density_weight,
                'eff':self.eff_raise(new_eff)*self.connetivity_weight,
                'target_sc':0,
            }
        reward = sum(list(reward_dict.values()))
        return reward