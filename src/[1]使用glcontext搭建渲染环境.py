import numpy as np
from PIL import Image

from geo import RoadCollection,BuildingCollection,RegionCollection
from graphic_module import RoadObserver,BuildingObserver,RegionObserver
from style_module import StyleManager as sm
from utils import headless_utils
from utils import io_utils

"""
    
备注： 程序使用了moderngl的headless模式运行，其需要在python运行时加入--window headless参数， 
因此无法在jupyter notebook中运行
"""


# region 方法一: 继承headless_utils.GLContext类
class MyGLContext(headless_utils.GLContext):
    """
    headless_utils.GLContext是本项目中搭建好的一个支持使用openGL 和modernGL进行渲染的环境类

    此代码使用该类的基础模板
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self):
        """
        main方法是一个抽象类， 需要被覆写， 其中内容定义了程序执行的主要内容
        """
        # 导入data
        data = io_utils.load_data("../data/VirtualEnv/0322_data.bin")
        road_collection = RoadCollection()
        road_collection.data_to_roads(data)
        building_collection = BuildingCollection()
        building_collection.data_to_buildings(data)
        region_collection = RegionCollection()
        region_collection.data_to_regions(data)

        # 计算视图范围
        min_coord, max_coord = road_collection.get_bbox()
        observation_center = (min_coord + max_coord) / 2
        observation_size = np.max(max_coord - min_coord)

        # 创建观察者（渲染器）
        road_obs = RoadObserver(
            name="road_observer",  # 名字不能重复
            width=1024,  # 图像宽度
            height=1024,  # 图像高度
            observation_size=(observation_size, observation_size),  # 设置观察范围
            initial_gdf=road_collection.get_all_roads(),  # 设置初始渲染内容
            sf=sm.I.env.road_level_style_factory  # 设置着色方案
        )

        road_obs.update_observation_center(observation_center)  # 设置渲染中心点

        # 创建建筑观察者
        building_obs = BuildingObserver(
            name="building_observer",
            width=1024,
            height=1024,
            observation_size=(observation_size, observation_size),
            initial_gdf=building_collection.get_all_buildings(),
            sf=sm.I.env.building_movable_style_factory
        )
        building_obs.update_observation_center(observation_center)

        # 创建区域观察者
        region_obs = RegionObserver(
            name="region_observer",
            width=1024,
            height=1024,
            observation_size=(observation_size, observation_size),
            initial_gdf=region_collection.get_all_regions(),
            sf=sm.I.env.region_accessible_style_factory
        )
        region_obs.update_observation_center(observation_center)

        # 渲染图像
        road_obs.render()  # 使用OpenGL渲染图像至Observer的texture中（GPU）
        road_img_arr = road_obs.get_render_img()  # 从Texture读取图像为numpy.array（CPU）
        # road_img = Image.fromarray(road_img_arr)  # 转换为pillow Image
        # road_img.show()  # 显示图像

        building_obs.render()  # 使用OpenGL渲染图像至Observer的texture中（GPU）
        building_img_arr = building_obs.get_render_img()  # 从Texture读取图像为numpy.array（CPU）
        # building_img = Image.fromarray(building_img_arr)  # 转换为pillow Image
        # building_img.show()  # 显示图像
        region_obs.render()  # 使用OpenGL渲染图像至Observer的texture中（GPU）
        region_img_arr = region_obs.get_render_img()

        combined_img_arr = np.maximum(np.maximum(road_img_arr, building_img_arr), region_img_arr)
        combined_img = Image.fromarray(combined_img_arr)
        combined_img.show()

if __name__ == '__main__':
    MyGLContext.run()


# endregion


# region 方法二: 使headless_utils.glcontext装饰器将需要使用OpenGL渲染的内容包裹起来
@headless_utils.glcontext
def my_func():
    pass


if __name__ == '__main__':
    my_func()

# endregion
