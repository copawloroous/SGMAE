from einops import rearrange
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'),
            nn.ReLU(),
            # 移除问题池化层
            # nn.MaxPool3d(kernel_size=(2, 2, 2))  # 会导致特征图尺寸过小
        )
        # 添加自适应池化层替代固定池化
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3, 1, 1))  # 确保稳定输出尺寸
        self.linear = nn.Linear(128 * 3 * 1 * 1, 30)  # 根据新尺寸调整输入特征数

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.adaptive_pool(X)  # 应用自适应池化
        X = rearrange(X, 'b c h w d -> b (c h w d)')
        X = self.linear(X)
        return X