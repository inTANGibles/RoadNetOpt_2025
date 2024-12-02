from reward.reward_roadnet import RewardRoadNet,AnalysisRoadNet
from geo import RoadCollection
from utils import io_utils,RoadLevel

if __name__ == '__main__':
    data = io_utils.load_data('../data/和县/data.bin')
    road_collection = RoadCollection()
    road_collection.data_to_roads(data)

    node = road_collection.get_all_nodes()['geometry'].values[100]
    print(node)

    ob1 = AnalysisRoadNet(road_collection,road_collection,node)
    pri_ratio, density, eff, target_score = ob1.get_all_indexes('choice')
    print(target_score)
    ob1.roadnet_target_meandepth(node)