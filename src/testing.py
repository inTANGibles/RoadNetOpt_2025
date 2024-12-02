import numpy as np
from PIL import Image

from geo import RoadCollection, BuildingCollection, RegionCollection
from graphic_module import RoadObserver
from style_module import StyleManager as sm
from utils import RoadState, RoadLevel
from utils import headless_utils
from utils import io_utils


@headless_utils.glcontext
def main():
    Road = RoadCollection()
    Building = BuildingCollection()
    Region = RegionCollection()

    data = io_utils.load_data("../data/和县/data.bin")
    Road.data_to_roads(data)

    raw_roads = Road.get_all_roads()
    random_road = Road.get_all_roads().sample().iloc[0]

    spawn_point = Road.interpolate_road_by_random_position(random_road)
    assert spawn_point is not None
    spawn_point = spawn_point.reshape(-1, 2)
    Road.split_road_by_coord(random_road, spawn_point)
    uid = Road.add_road_by_coords(spawn_point, RoadLevel.TERTIARY, RoadState.OPTIMIZING)
    road_agent = Road.get_road_by_uid(uid)

    STRTree_key = Road.build_STRTree(raw_roads)

    vec = np.array([[20.0, 0]])
    for i in range(100):
        print(i)
        lst_pt = Road.get_road_last_point(road_agent)
        road_agent = Road.add_point_to_road(road_agent, point=lst_pt + vec)
        intersect = Road.detect_intersection_fast(STRTree_key, road_agent)
        print(intersect)
        if intersect:
            break
    # 计算视图范围
    min_coord, max_coord = Road.get_bbox()
    observation_center = (min_coord + max_coord) / 2
    observation_size = np.max(max_coord - min_coord)

    # 创建观察者（渲染器）
    road_obs = RoadObserver(
        name="road_observer",
        width=512,
        height=512,
        observation_size=(observation_size, observation_size),
        initial_gdf=Road.get_all_roads(),
        sf=sm.I.dis.road_state_style_factory
    )
    road_obs.update_observation_center(observation_center)

    # 渲染图象
    road_obs.render()
    img_arr = road_obs.get_render_img()
    img = Image.fromarray(img_arr)
    img.show()


main()
