import torch.nn as nn
from DDPG import conv, pool

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(  # 4 * 128 * 128
            nn.Conv2d(4, 64, 4, 2),  # 64 * 63 * 63
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),  # 64 * 31 * 31

            nn.Conv2d(64, 128, 4, 2),  # 64 * 14 * 14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),  # 128 * 7 * 7

            nn.Conv2d(128, 128, 3, 1),  # 128 * 5 * 5
            nn.BatchNorm2d(128),
            nn.Sigmoid(),  # 128 * 5 * 5
        )


    def forward(self, x):
        return self.CNN(x)

    @staticmethod
    def calculate_cnn_output_shape(input_shape: tuple[int]):
        shape = input_shape
        shape = conv(shape, 64, 4, 2)
        shape = pool(shape, 2)
        shape = conv(shape, 64, 4, 2)
        shape = pool(shape, 2)
        shape = conv(shape, 64, 3, 1)
        shape = pool(shape, 2)
        return shape
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# A = CNN().to(device)
# print(A)
# summary(A, input_size=(4, 512, 512))