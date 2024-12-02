import logging
import time
import uuid

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from lib.accelerator import cAccelerator, arrs_addr_len
from utils import RegionAccessibleType, RegionType, RegionCluster


class RegionCollection:
    __region_attrs = ['uid', 'geometry', 'coords', 'enabled', 'accessible', 'region_type', 'quality']
    __region_gdf = gpd.GeoDataFrame(columns=__region_attrs)
    __region_gdf.set_index('uid')

    __uid = uuid.uuid4()

    def uid(self):
        return self.__uid

    # region 增加删除

    @staticmethod
    def _create_region_by_coords(coords: np.ndarray,
                                 accessible: RegionAccessibleType = RegionAccessibleType.UNDEFINED,
                                 region_type: RegionType = RegionType.UNDEFINED,
                                 enabled: bool = True):
        geometry = Polygon(coords)
        uid = uuid.uuid4()
        new_row = {
            'uid': [uid],
            'geometry': [geometry],
            'coords': [coords],
            'accessible': [accessible],
            'region_type': [region_type],
            'enabled': [enabled]
        }
        return gpd.GeoDataFrame(new_row, index=new_row['uid'])

    @staticmethod
    def create_regions_by_coords(coords_list: list[np.ndarray],
                                 accessible_list: list[RegionAccessibleType] = None,
                                 region_type_list: list[RegionType] = None,
                                 enable_list: list[bool] = None):
        geometry_list = [Polygon(points) for points in coords_list]
        if enable_list is None:
            enable_list = [True for _ in geometry_list]
        if accessible_list is None:
            accessible_list = [RegionAccessibleType.UNDEFINED for _ in geometry_list]
        if region_type_list is None:
            region_type_list = [RegionType.UNDEFINED for _ in geometry_list]
        assert len(geometry_list) == len(accessible_list) == len(region_type_list)
        uid_list = [uuid.uuid4() for _ in geometry_list]
        new_data = {
            'uid': uid_list,
            'geometry': geometry_list,
            'coords': coords_list,
            'accessible': accessible_list,
            'region_type': region_type_list,
            'enabled': enable_list
        }
        return gpd.GeoDataFrame(new_data, index=new_data['uid'])

    def add_region(self, region, return_uid=True):
        if not self.__region_gdf.empty:
            self.__region_gdf = gpd.pd.concat([self.__region_gdf, region], ignore_index=False)
        else:
            self.__region_gdf = region
        self.__uid = uuid.uuid4()
        if return_uid:
            return region['uid']

    def add_regions(self, regions, return_uid=False):
        if not self.__region_gdf.empty:
            self.__region_gdf = gpd.pd.concat([self.__region_gdf, regions], ignore_index=False)
        else:
            self.__region_gdf = regions
        self.__uid = uuid.uuid4()
        if return_uid:
            return list(regions['uid'])

    def add_region_by_coords(self, coords: np.ndarray,
                             accessible: RegionAccessibleType = RegionAccessibleType.UNDEFINED,
                             region_type: RegionType = RegionType.UNDEFINED,
                             enabled: bool = True):
        region = self._create_region_by_coords(coords,
                                               accessible,
                                               region_type,
                                               enabled)
        return self.add_region(region)

    def add_regions_by_coords(self, coords_list: list[np.ndarray],
                              accessible_list: list[RegionAccessibleType] = None,
                              region_type_list: list[RegionType] = None,
                              enable_list: list[bool] = None):
        regions = self.create_regions_by_coords(coords_list,
                                                accessible_list,
                                                region_type_list,
                                                enable_list)
        return self.add_regions(regions)

    def delete_region(self, region):
        uid = region['uid']
        self.__region_gdf.drop(uid, inplace=True)
        self.__uid = uuid.uuid4()

    def delete_region_by_uid(self, uid):
        region = self.get_region_by_uid(uid)
        self.delete_region(region)

    def delete_all(self):
        self.__region_gdf.drop(self.__region_gdf['uid'], inplace=True)
        self.__uid = uuid.uuid4()

    def create_region_by_min_max(self, region_min, region_max) -> gpd.GeoDataFrame:
        top_left = region_min
        top_right = np.array([region_max[0], region_min[1]])
        bottom_right = region_max
        bottom_left = np.array([region_min[0], region_max[1]])
        coords = np.array([top_left, top_right, bottom_right, bottom_left])
        bound_region_gdf = self.create_regions_by_coords(
            coords_list=[coords],
            accessible_list=[RegionAccessibleType.UNDEFINED],
            region_type_list=[RegionType.UNDEFINED],
            enable_list=[True]
        )
        return bound_region_gdf

    # endregion

    # region 获取查找

    def get_region_attrs(self):
        return self.__region_attrs

    def get_region_by_uid(self, uid):
        region = self.__region_gdf.loc[uid]
        return region

    def get_region_by_index(self, idx):
        region = self.__region_gdf.iloc[idx]
        return region

    def get_regions_by_attr_and_value(self, attr: str, value: any):
        assert attr in self.__region_attrs, f'unexpected attr ({attr}), attr must be one of these: {self.__region_attrs}'
        regions = self.__region_gdf.loc[self.__region_gdf[attr] == value]
        return regions

    def get_first_region(self):
        return self.get_region_by_index(0)

    def get_last_region(self):
        return self.get_region_by_index(-1)

    def get_all_regions(self):
        return self.__region_gdf

    def get_regions_by_cluster(self, cluster: RegionCluster):
        cluster = cluster.cluster
        uid_sets_by_attr = []
        for attr in cluster:
            gdfs = []
            if all(cluster[attr].values()):
                print(f'{attr} 全都是True, 跳过')
                continue
            for key in cluster[attr]:
                if cluster[attr][key]:
                    _gdfs = self.get_regions_by_attr_and_value(attr, key)
                    gdfs.append(_gdfs)
            if len(gdfs) == 0:
                return None
            gdf = pd.concat(gdfs, ignore_index=False)
            uid_sets_by_attr.append(set(gdf.index))
        if len(uid_sets_by_attr) == 0:
            print(f'全都为True, 直接返回所有')
            return self.get_all_regions()
        common_uid = list(set.intersection(*uid_sets_by_attr))
        return self.get_all_regions().loc[common_uid]

    # endregion

    # region 编辑修改

    def set_attr_value(self, regions, attr, value):
        # TODO: 这里可能有问题
        assert attr in self.__region_attrs, f'unexpected attr ({attr}), attr must be one of these: {self.__region_attrs}'
        regions[attr] = value
        self.__uid = uuid.uuid4()

    # endregion

    # region 绘图相关

    @staticmethod
    def plot_regions(regions, *args, **kwargs):
        if regions is None:
            return
        regions.plot(*args, **kwargs)

    def plot_all(self, *args, **kwargs):
        self.__region_gdf.plot(*args, **kwargs)

    @staticmethod
    def plot_using_style_factory(regions, style_factory, *args, **kwargs):
        if regions is None:
            return
        colors, face_color, edge_color, line_width = style_factory(regions)
        regions_copy = regions.copy()
        regions_copy['colors'] = colors
        regions_copy['edge_color'] = edge_color
        regions_copy['line_width'] = line_width
        regions_copy.plot(color=regions_copy['colors'],
                          edgecolor=regions_copy['edge_color'],
                          linewidth=regions_copy['line_width'],
                          *args, **kwargs)

    @staticmethod
    def get_vertices_data(regions, style_factory, debug=False):
        _ = debug
        # time logger
        start_time = time.time()
        # colors and width
        params = style_factory(regions)  # tuple( colors and widths)
        colors = np.array(params[0], dtype=np.float32)  # float32
        if colors.shape[1] == 3:
            colors = np.concatenate((colors, np.ones((len(colors), 1), dtype=np.float32)), axis=1)
        in_r, in_g, in_b, in_a = colors[:, 0].copy(), colors[:, 1].copy(), colors[:, 2].copy(), colors[:, 3].copy()

        # coords, first and num
        vertex_coords = regions['coords']  # pandas
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

    def data_to_regions(self, data: dict):
        assert 'regions' in data, 'invalid data'
        self.delete_all()

        regions_data = data['regions']
        assert isinstance(regions_data, list)
        coords_list = []
        accessible_list = []
        region_type_list = []

        for bd in regions_data:
            if len(bd['points']) < 4:
                continue
            coords_list.append(np.array(bd['points']))
            accessible_list.append(bd['accessible'])
            if 'region_type' in bd:  # 向前兼容
                region_type_list.append(bd['region_type'])
            else:
                region_type_list.append(bd['type'])
        self.add_regions_by_coords(coords_list, accessible_list, region_type_list, None)

    def regions_to_data(self, out_data: dict):
        if 'regions' not in out_data:
            out_data['regions'] = []
        for uid, region in self.get_all_regions().iterrows():
            region_data = {
                'points': region['coords'],
                'region_type': region['region_type'],
                'accessible': region['accessible'],
                'quality': region['quality']
            }
            out_data['regions'].append(region_data)

    # endregion

    # region 其他

    # endregion
