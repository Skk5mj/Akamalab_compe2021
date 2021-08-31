import numpy as np
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F # これはいらないかも
# torch.nn.functionalでは用いずに基本的にnnで良いらしい
# nn.Softmaxで動いてくれなかったのでF.softmaxにした
from torch.utils import data
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6670,3000), nn.ReLU(),
            nn.Linear(3000, 1500), nn.ReLU(),
            nn.Linear(1500, 750), nn.ReLU(),
            nn.Linear(750, 100), nn.ReLU(),
            nn.Linear(100,20), nn.ReLU(),
            nn.Linear(20, 2)
        )
    # フィードフォワード(順伝搬)
    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim = 1) # ソフトマックス関数で2クラスの予測(実質シグモイドを使うのと同じ)
        return x