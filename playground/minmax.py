import numpy as np
import pandas as pd

# 示例 DataFrame
df = pd.DataFrame({'coords': [np.array([(1, 2), (3, 4), (5, 6)]),
                              np.array([(7, 8), (9, 10)])]})

# 将所有坐标展开为一维数组
all_coords = np.concatenate(df['coords'].values)
print(all_coords.shape)

# 获取坐标的最小值和最大值
min_coords = np.amin(all_coords, axis=0)
max_coords = np.amax(all_coords, axis=0)

print("Min Coords:", min_coords)
print("Max Coords:", max_coords)