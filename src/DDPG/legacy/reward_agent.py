import cv2
import numpy as np #目前仍然以numpy对接，后期改成tensor
from graphic_module import GraphicManager
class RewardAgent():
    def __init__(self,last_points,buildings_img, boundpts, headless=True):
        self.agent_position = last_points
        self.env_buildings = buildings_img 
        self.bound = boundpts
        self.headless = headless
        '''后续补充其他评价来源
        self.graph = graph
        self.road = roads
        self.env_terrain = terrain
        self.env_water = water_img
        '''
        self.building_weight = 300
        self.graph_weight = 1
        self.road_weight = 1
        self.terrain_weight = 1
        self.water_weight = 300
        self.bound_weight = 700
        #后续补充调整其他评价的权重关系

    def building_rewards(self):
        '''限制不能进入的区域,传入当前建筑图像与智能体位置
        返回规则: 得分区间[-1,0],越靠近建筑越接近于-1'''

        imgGray = self.env_buildings
        # imgGray = cv2.blur_image(imgGray[:, :, :3])
        imgGray = cv2.GaussianBlur(imgGray, (3,3), 0)
        imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)
        if not self.headless:
            GraphicManager.I.bilt_to_queue('building_rewards', imgGray)
        inds = self.agent_position.astype(int)
        pixel_values = imgGray[inds[:, 1], inds[:, 0]]
        min1 = 126 # np.min(imgGray)
        max1 = np.max(imgGray)
        reward = -1 + (1 / np.log(max1-min1+1) * np.log(np.abs(pixel_values- min1+1))) # 非线性变化
        # reward = (pixel_values-min1)/(max1-min1)-1 #线性变化

        # 监听reward对应关系
        # for ind in inds:
        #     print((ind[0],ind[1]))
        #     cv2.circle(imgGray,(ind[0],ind[1]),3, (0, 0, 0), 3)
        # cv2.imshow('gray', imgGray)
        # cv2.waitKey()
        # print(pixel_values)
        # print(reward)
        return reward.reshape((-1, 1))


    def graph_rewards(self):
        return 0
    
    def road_rewards(self):
        return 0
    
    def terrain_rewards(self):
        return 0
    
    # def water_rewards(self):
    #     '''限制不能进入的区域,传入当前水体图像与智能体位置
    #     返回规则: 得分区间[-1,0],越靠近建筑越接近于-1'''

    #     imgGray = self.env_water
    #     # imgGray = cv2.blur_image(imgGray[:, :, :3])
    #     imgGray = cv2.GaussianBlur(imgGray, (5,5), 0)
    #     imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)
    #     inds = self.agent_position.astype(int)
    #     pixel_values = imgGray[inds[:, 1], inds[:, 0]]
    #     min1 = 126 # np.min(imgGray)
    #     max1 = np.max(imgGray)
    #     reward = -1 + (1 / np.log(max1-min1+1) * np.log(np.abs(pixel_values- min1+1))) # 非线性变化
    #     return reward.reshape((-1, 1))

    def bound_rewards(self):
        height, width = self.env_buildings.shape[:2]
        img = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(self.bound, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img,[pts], color=255)
        img = cv2.GaussianBlur(img, (55, 55), 0)
        if not self.headless:
            GraphicManager.I.bilt_to_queue('bound_rewards', img)
        inds = self.agent_position.astype(int)
        pixel_values = img[inds[:, 1], inds[:, 0]]
        min1 = 0.1
        max1 = np.max(img)
        reward = -1 + (1 / np.log(max1 - min1 + 1) * np.log(np.abs(pixel_values - min1 + 1)))  # 非线性变化

        # cv2.imshow('Result', img)
        # cv2.waitKey(0)
        # print(pixel_values)
        # print(reward)
        return reward.reshape((-1, 1))

    def agent_reward(self):
        reward = self.building_rewards()*self.building_weight + self.graph_rewards()*self.graph_weight + self.road_rewards() * self.road_weight + self.terrain_rewards() * self.terrain_weight + self.bound_rewards() * self.bound_weight #+ self.water_rewards() * self.water_weight
        return reward