import logging
import time
import uuid
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import shapely.plotting
import pandas as pd
from utils.common_utils import timer
from utils import BuildingMovableType, BuildingStyle, BuildingQuality, BuildingCluster
from lib.accelerator import cAccelerator, arrs_addr_len


print('building loaded')


class Building:
    __building_attrs = ['uid', 'geometry', 'coords', 'enabled', 'movable', 'style', 'quality']
    __building_gdf = gpd.GeoDataFrame(columns=__building_attrs)
    __building_gdf.set_index('uid')

    __uid = uuid.uuid4()

    @staticmethod
    def uid():
        return Building.__uid

    # region 增加删除
    @staticmethod
    def _create_building_by_coords(coords: np.ndarray,
                                   movable: BuildingMovableType = BuildingMovableType.UNDEFINED,
                                   style: BuildingStyle = BuildingStyle.UNDEFINED,
                                   quality: BuildingQuality = BuildingQuality.UNDEFINED,
                                   enabled: bool = True):
        geometry = Polygon(coords)
        uid = uuid.uuid4()
        new_row = {
            'uid': [uid],
            'geometry': [geometry],
            'coords': [coords],
            'movable': [movable],
            'style': [style],
            'quality': [quality],
            'enabled': [enabled]
        }
        return gpd.GeoDataFrame(new_row, index=new_row['uid'])

    @staticmethod
    def _create_buildings_by_coords(coords_list: list[np.ndarray],
                                    movable_list: list[BuildingMovableType] = None,
                                    style_list: list[BuildingStyle] = None,
                                    quality_list: list[BuildingQuality] = None,
                                    enable_list: list[bool] = None):

        geometry_list = [Polygon(points) for points in coords_list]
        if enable_list is None:
            enable_list = [True for _ in geometry_list]
        if movable_list is None:
            movable_list = [BuildingMovableType.UNDEFINED for _ in geometry_list]
        if style_list is None:
            style_list = [BuildingStyle.UNDEFINED for _ in geometry_list]
        if quality_list is None:
            quality_list = [BuildingQuality.UNDEFINED for _ in geometry_list]

        assert len(geometry_list) == len(movable_list) == len(style_list) == len(quality_list)
        uid_list = [uuid.uuid4() for _ in geometry_list]
        new_data = {
            'uid': uid_list,
            'geometry': geometry_list,
            'coords': coords_list,
            'movable': movable_list,
            'style': style_list,
            'quality': quality_list,
            'enabled': enable_list
        }
        return gpd.GeoDataFrame(new_data, index=new_data['uid'])

    @staticmethod
    def add_building(building, return_uids=True):
        if not Building.__building_gdf.empty:
            Building.__building_gdf = gpd.pd.concat([Building.__building_gdf, building], ignore_index=False)
        else:
            Building.__building_gdf = building
        Building.__uid = uuid.uuid4()
        if return_uids:
            return building['uid']

    @staticmethod
    def add_buildings(buildings, return_uids=False):
        """默认不返回uid"""
        if not Building.__building_gdf.empty:
            Building.__building_gdf = gpd.pd.concat([Building.__building_gdf, buildings], ignore_index=False)
        else:
            Building.__building_gdf = buildings
        Building.__uid = uuid.uuid4()
        if return_uids:
            return buildings['uid'].values

    @staticmethod
    def add_building_by_coords(coords: np.ndarray,
                               movable: BuildingMovableType = BuildingMovableType.UNDEFINED,
                               style: BuildingStyle = BuildingStyle.UNDEFINED,
                               quality: BuildingQuality = BuildingQuality.UNDEFINED,
                               enabled: bool = True, return_uids=True):
        building = Building._create_building_by_coords(coords,
                                                       movable,
                                                       style,
                                                       quality,
                                                       enabled)
        return Building.add_building(building, return_uids)

    @staticmethod
    def add_buildings_by_coords(points_list: list[np.ndarray],
                                movable_list: list[BuildingMovableType] = None,
                                style_list: list[BuildingStyle] = None,
                                quality_list: list[BuildingQuality] = None,
                                enable_list: list[bool] = None,
                                return_uids=False):
        buildings = Building._create_buildings_by_coords(points_list,
                                                         movable_list,
                                                         style_list,
                                                         quality_list,
                                                         enable_list)
        return Building.add_buildings(buildings, return_uids)

    @staticmethod
    def delete_building(building):
        uid = building['uid']
        Building.__building_gdf.drop(uid, inplace=True)
        Building.__uid = uuid.uuid4()

    @staticmethod
    def delete_building_by_uid(uid):
        building = Building.get_building_by_uid(uid)
        Building.delete_building(building)

    @staticmethod
    def delete_all():
        Building.__building_gdf.drop(Building.__building_gdf['uid'], inplace=True)
        Building.__uid = uuid.uuid4()

    # endregion

    # region 获取查找
    @staticmethod
    def get_building_attrs():
        return Building.__building_attrs

    @staticmethod
    def get_building_by_uid(uid):
        building = Building.__building_gdf.loc[uid]
        return building

    @staticmethod
    def get_building_by_index(idx):
        building = Building.__building_gdf.iloc[idx]
        return building

    @staticmethod
    def get_buildings_by_attr_and_value(attr: str, value: any):
        assert attr in Building.__building_attrs, f'unexpected attr ({attr}), attr must be one of these: {Building.__building_attrs}'
        buildings = Building.__building_gdf.loc[Building.__building_gdf[attr] == value]
        return buildings

    @staticmethod
    def get_first_building():
        return Building.get_building_by_index(0)

    @staticmethod
    def get_last_building():
        return Building.get_building_by_index(-1)

    @staticmethod
    def get_all_buildings():
        return Building.__building_gdf

    @staticmethod
    def get_buildings_by_cluster(cluster: BuildingCluster):
        cluster = cluster.cluster
        uid_sets_by_attr = []
        for attr in cluster:
            gdfs = []
            if all(cluster[attr].values()):
                print(f'{attr} 全都是True, 跳过')
                continue
            for key in cluster[attr]:
                if cluster[attr][key]:
                    _gdfs = Building.get_buildings_by_attr_and_value(attr, key)
                    gdfs.append(_gdfs)
            if len(gdfs) == 0:
                return None
            gdf = pd.concat(gdfs, ignore_index=False)
            uid_sets_by_attr.append(set(gdf.index))
        if len(uid_sets_by_attr) == 0:
            print(f'全都为True, 直接返回所有')
            return Building.get_all_buildings()
        common_uid = list(set.intersection(*uid_sets_by_attr))
        return Building.get_all_buildings().loc[common_uid]

    # endregion

    # region 编辑修改
    @staticmethod
    def set_attr_value(buildings, attr, value):
        # TODO 这个可能有问题
        assert attr in Building.__building_attrs, f'unexpected attr ({attr}), attr must be one of these: {Building.__building_attrs}'
        buildings[attr] = value
        Building.__uid = uuid.uuid4()

    # endregion

    # region 绘图相关
    @staticmethod
    def plot_buildings(buildings, *args, **kwargs):
        if buildings is None:
            return
        buildings.plot(*args, **kwargs)

    @staticmethod
    def plot_all(*args, **kwargs):
        Building.__building_gdf.plot(*args, **kwargs)

    @staticmethod
    def plot_using_style_factory(buildings, style_factory, *args, **kwargs):
        if buildings is None:
            return
        colors, face_color, edge_color, line_width = style_factory(buildings)
        buildings_copy = buildings.copy()
        buildings_copy['colors'] = colors
        buildings_copy['edge_color'] = edge_color
        buildings_copy['line_width'] = line_width
        buildings_copy.plot(color=buildings_copy['colors'],
                            edgecolor=buildings_copy['edge_color'],
                            linewidth=buildings_copy['line_width'],
                            *args, **kwargs)

    @staticmethod
    def _process_geometry(geometry: shapely.Polygon) -> np.ndarray:
        coords = np.array(geometry.exterior.coords, dtype=np.float32)[:-1, :]
        return coords

    @staticmethod
    @timer
    def get_vertices_data(buildings, style_factory):
        # time logger
        start_time = time.time()
        # colors and width
        params = style_factory(buildings)  # tuple( colors and widths)
        colors = np.array(params[0], dtype=np.float32)  # float32
        if colors.shape[1] == 3:
            colors = np.concatenate((colors, np.ones((len(colors), 1), dtype=np.float32)), axis=1)
        in_r, in_g, in_b, in_a = colors[:, 0].copy(), colors[:, 1].copy(), colors[:, 2].copy(), colors[:, 3].copy()

        # coords, first and num
        vertex_coords = buildings['coords']  # pandas
        in_first = np.concatenate(([0], np.cumsum(vertex_coords.apply(len).to_numpy()[:-1])), dtype=np.int32)  # int32
        in_num = vertex_coords.apply(len).to_numpy().astype(np.int32)  # int32
        vertex_coords = np.concatenate(vertex_coords.values, axis=0).astype(np.float32)  # (n, 2) float32
        in_x, in_y = vertex_coords[:, 0].copy(), vertex_coords[:, 1].copy()

        # time logger
        logging.debug(f'prepare bytes 消耗时间 = {time.time() - start_time}s')
        start_time = time.time()

        # c# code
        c_params = arrs_addr_len(in_x, in_y, in_first, in_num, in_r, in_g, in_b, in_a)
        buffer = cAccelerator.TriangulatePolygons(*c_params)

        # time logger
        logging.debug(f'c# 代码消耗时间（含数据传输） = {time.time() - start_time}s')
        start_time = time.time()

        # convert to numpy
        vertices = np.frombuffer(bytes(buffer), np.float32).reshape(-1, 6)
        if np.any(np.isnan(vertices)):
            logging.warning("There are NaN values in the vertices array.")

        # time logger
        logging.debug(f'转换为numpy消耗时间 = {time.time() - start_time}s')
        return vertices

    # endregion

    # region 类型转换

    @staticmethod
    @timer
    def data_to_buildings(data: dict):
        assert 'buildings' in data, 'invalid data'
        Building.delete_all()

        buildings_data = data['buildings']
        assert isinstance(buildings_data, list)
        # print(f"共有{len(buildings_data)}条建筑数据")
        points_list = []
        movable_list = []
        style_list = []
        quality_list = []
        for i in range(len(buildings_data)):
            bd = buildings_data[i]
            if len(bd['points']) < 4:
                continue
            points_list.append(np.array(bd['points']))
            movable_list.append(bd['movable'])
            style_list.append(bd['style'])
            quality_list.append(bd['quality'])
        Building.add_buildings_by_coords(points_list, movable_list, style_list, quality_list, None)

    @staticmethod
    def buildings_to_data(out_data: dict):

        if 'buildings' not in out_data:
            out_data['buildings'] = []
        for uid, building in Building.get_all_buildings().iterrows():
            building_data = {
                'points': building['coords'],
                'style': building['style'],
                'movable': building['movable'],
                'quality': building['quality']
            }
            out_data['buildings'].append(building_data)


    # endregion

    # region 其他

    # endregion


if __name__ == "__main__":
    pass
