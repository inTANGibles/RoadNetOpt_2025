import numpy as np


for pixel_value in range(0, 255, 10):
    min1 = 0.1  # np.min(imgGray)
    max1 = 255
    reward = - (1 / np.log(max1 - min1 + 1) * np.log(np.abs(pixel_value - min1 + 1)))
    print(pixel_value, reward)