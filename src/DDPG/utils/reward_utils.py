# reward_consts.py

# -------------------------------
# Step-level reward constants
# -------------------------------
BUILDING_REGION_REWARD = 0b00000  # 0 建筑 / 区域碰撞惩罚
ROAD_REWARD = 0b00001  # 1 靠近出生道路惩罚
BOUND_REWARD = 0b00010  # 2 边界惩罚
BACKWARD_PENALTY = 0b00100  # 4 回头惩罚
DEAD_NODE_PENALTY = 0b00101  # 5 靠近断头路奖励
CROSS_NODE_PENALTY = 0b00110  # 6 靠近交叉口惩罚
EXPLORATION_REWARD = 0b00111  # 7 角度顺滑奖励
REWARD_SUM = 0b01000  # 8 探索区域稀疏奖励

# -------------------------------
# Final-level reward constants
# -------------------------------
FINAL_DEFAULT = 0b01000  # 8
FINAL_ENDNODE_REWARD = 0b01001  # 9
FINAL_DISTANCE_REWARD = 0b01010  # 10
FINAL_ANGLE_REWARD = 0b01011  # 11
FINAL_REWARD_SHIFT = 0b01100  # 12
FINAL_REWARD_WEIGHT = 0b01101  # 13

# -------------------------------
# Global roadnet reward constants
# -------------------------------
FINAL_EFFICIENCY_REWARD = 0b11000  # 24
FINAL_DENSITY_REWARD = 0b11001  # 25
FINAL_CONTINUITY_REWARD = 0b11010  # 26
FINAL_BEARING_REWARD = 0b11011  # 27

REWARD_KEYS = {
    BUILDING_REGION_REWARD,
    ROAD_REWARD,
    BOUND_REWARD,
    BACKWARD_PENALTY,
    DEAD_NODE_PENALTY,
    CROSS_NODE_PENALTY,
    EXPLORATION_REWARD
}

FINAL_REWARD_KEYS = {
    FINAL_DEFAULT,
    FINAL_ENDNODE_REWARD,
    FINAL_DISTANCE_REWARD,
    FINAL_ANGLE_REWARD,
    FINAL_EFFICIENCY_REWARD,
    FINAL_DENSITY_REWARD,
    FINAL_CONTINUITY_REWARD,
    FINAL_BEARING_REWARD
}

REWARD_DISPLAY_NAMES = {
    BUILDING_REGION_REWARD: 'building_region_reward',
    ROAD_REWARD: 'road_reward',
    BOUND_REWARD: 'bound_reward',
    BACKWARD_PENALTY: 'backward_penalty',
    DEAD_NODE_PENALTY: 'dead_node_penalty',
    CROSS_NODE_PENALTY: 'cross_node_penalty',
    REWARD_SUM: 'sum',

    FINAL_DEFAULT: 'final_default',
    FINAL_ENDNODE_REWARD: 'final_endnode_reward',
    FINAL_DISTANCE_REWARD: 'final_distance_ratio_reward',
    FINAL_ANGLE_REWARD: 'final_acute_angle_reward',
    FINAL_EFFICIENCY_REWARD: 'final_efficiency_reward',
    FINAL_DENSITY_REWARD: 'final_density_reward',
    FINAL_CONTINUITY_REWARD: 'final_continuity_reward',
    FINAL_BEARING_REWARD: 'final_bearing_reward',
    FINAL_REWARD_SHIFT: 'final_reward_shift',
}
