import numpy as np
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F # これはいらないかも
# torch.nn.functionalでは用いずに基本的にnnで良いらしい
# nn.Softmaxで動いてくれなかったのでF.softmaxにした
from torch.utils import data

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
        x = F.softmax(x, dim = 1) # ソフトマックス関数で2クラスの予測(実質シグモイドを使うのと同じ)
        return x