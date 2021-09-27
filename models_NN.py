import numpy as np
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F

# 1層目の出力を16384にしたNN
class Net_deeper(nn.Module):
    def __init__(self):
        super(Net_deeper, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6670,16384), nn.ReLU(), nn.BatchNorm1d(16384),
            nn.Linear(16384, 8192), nn.ReLU(), nn.BatchNorm1d(8192),
            nn.Linear(8192, 4096), nn.ReLU(),nn.BatchNorm1d(4096),
            nn.Linear(4096, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024,512),nn.ReLU(),nn.BatchNorm1d(512),
            nn.Linear(512,256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256,64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
    # フィードフォワード(順伝搬)
    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim = 1)
        return x

# lienar -> batch norm  -> ReLU -> dropoutに変更
class Net_deeper2(nn.Module):
    def __init__(self):
        super(Net_deeper2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6670,16384), nn.BatchNorm1d(16384), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(16384, 8192), nn.BatchNorm1d(8192), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(8192, 4096), nn.BatchNorm1d(4096),  nn.ReLU(),nn.Dropout(p=0.2),
            nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(256,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        )
    # フィードフォワード(順伝搬)
    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim = 1) # ソフトマックス関数で2クラスの予測(実質シグモイドを使うのと同じ)
        return x

# ELU + batch norm

class Net_deeper_elu(nn.Module):
    def __init__(self):
        super(Net_deeper_elu, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6670,16384), nn.ELU(), nn.BatchNorm1d(16384),
            nn.Linear(16384, 8192), nn.ELU(), nn.BatchNorm1d(8192),
            nn.Linear(8192, 4096), nn.ELU(),nn.BatchNorm1d(4096),
            nn.Linear(4096, 1024), nn.ELU(), nn.BatchNorm1d(1024),
            nn.Linear(1024,512),nn.ELU(),nn.BatchNorm1d(512),
            nn.Linear(512,256), nn.ELU(), nn.BatchNorm1d(256),
            nn.Linear(256,64), nn.ELU(), nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
    # フィードフォワード(順伝搬)
    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim = 1)
        return x

# ELU + batch norm + dropout
class Net_deeper_elu2(nn.Module):
    def __init__(self):
        super(Net_deeper_elu2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6670,16384), nn.BatchNorm1d(16384), nn.ELU(), nn.Dropout(p=0.2),
            nn.Linear(16384, 8192), nn.BatchNorm1d(8192), nn.ELU(), nn.Dropout(p=0.2),
            nn.Linear(8192, 4096), nn.BatchNorm1d(4096),  nn.ELU(),nn.Dropout(p=0.2),
            nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ELU(), nn.Dropout(p=0.2),
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ELU(), nn.Dropout(p=0.2),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(p=0.2),
            nn.Linear(256,64), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        )
    # フィードフォワード(順伝搬)
    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim = 1)
        return x

# mish
class Net_deeper_mish(nn.Module):
    def __init__(self):
        super(Net_deeper_mish, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6670,16384), nn.Mish(), nn.BatchNorm1d(16384),
            nn.Linear(16384, 8192), nn.Mish(), nn.BatchNorm1d(8192),
            nn.Linear(8192, 4096), nn.Mish(),nn.BatchNorm1d(4096),
            nn.Linear(4096, 1024), nn.Mish(), nn.BatchNorm1d(1024),
            nn.Linear(1024,512),nn.Mish(),nn.BatchNorm1d(512),
            nn.Linear(512,256), nn.Mish(), nn.BatchNorm1d(256),
            nn.Linear(256,64), nn.Mish(), nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
    # フィードフォワード(順伝搬)
    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim = 1)
        return x


# nakayaくんからもらったmodelはどういうわけかエラーで動かない


# # なかやくんのモデル
# class Net_nky(nn.Module):
#     def __init__(self):
#         super(Net_nky, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(6670,3000), nn.ReLU(), nn.BatchNorm1d(3000),nn.Dropout(p=0.2),
#             nn.Linear(3000, 1500), nn.ReLU(), nn.BatchNorm1d(1500),nn.Dropout(p=0.2),
#             nn.Linear(1500, 750), nn.ReLU(),nn.BatchNorm1d(750),nn.Dropout(p=0.2),
#             nn.Linear(750, 375), nn.ReLU(), nn.BatchNorm1d(375),nn.Dropout(p=0.2),
#             nn.Linear(375,100),nn.ReLU(),nn.BatchNorm1d(100),nn.Dropout(p=0.2),
#             nn.Linear(100,20), nn.ReLU(), nn.BatchNorm1d(20),nn.Dropout(p=0.2),
#             nn.Linear(20, 2)
#         )
#     # フィードフォワード(順伝搬)
#     def forward(self, x):
#         x = self.net(x)
#         x = nn.functional.softmax(x, dim = 1)
#         return x

# # なかやくんのモデルを
# # linear -> batch norm -> ReLU -> dropoutの順番にした
# # batch norm とdropoutを併用するときはこうするらしい
# class Net_nky2(nn.Module):
#     def __init__(self):
#         super(Net_nky2, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(6670,3000), nn.BatchNorm1d(3000), nn.ReLU(), nn.Dropout(p=0.2),
#             nn.Linear(3000, 1500), nn.BatchNorm1d(1500), nn.ReLU(), nn.Dropout(p=0.2),
#             nn.Linear(1500, 750), nn.BatchNorm1d(750), nn.ReLU(),nn.Dropout(p=0.2),
#             nn.Linear(750, 375), nn.BatchNorm1d(375), nn.ReLU(), nn.Dropout(p=0.2),
#             nn.Linear(375,100), nn.BatchNorm1d(100),nn.ReLU(), nn.Dropout(p=0.2),
#             nn.Linear(100,20), nn.BatchNorm1d(20), nn.ReLU(), nn.Dropout(p=0.2),
#             nn.Linear(20, 2)
#         )
#     # フィードフォワード(順伝搬)
#     def forward(self, x):
#         x = self.net(x)
#         x = nn.functional.softmax(x, dim = 1)
#         return x



