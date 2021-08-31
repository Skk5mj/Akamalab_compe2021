# 実装したいこと
"""
stratified 5-fold CV
"""

import torch
from torch import nn, optim
from torch.utils import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()

# def train_loop(model_NN, num_epoch, learning_rate):
#     model_NN = model_NN().to(device)
#     optimizer = optim.SGD(model_NN.parameters(), lr = learning_rate,momentum = 0.9, weight_decay= 5e-3)
    
# nnを引数にする意味はあるのか
def cross_val(model_NN, num_epoch, train_data, train_label):
    for train_idx, valid_idx in skf.split(train_data, train_label):
        model = model_NN().to(device) # 毎回新規のインスタンスにしてモデルを更新したい
        optimizer = optim.SGD(model.paramiters(), lr = 1e-3, momentum= 0.9, weight_decay = 5e-3)

        _train_data = train_data.iloc[train_idx]
        _train_label = train_label.iloc[train_idx]
        _valid_data = train_data.iloc[valid_idx]
        _valid_label = train_label.iloc[valid_idx]

        train_data_tensor = torch.tensor(np.array(_train_data.astype('f')))
        train_label_tesnor = torch.tensor(_train_label)
        train_tensor = data.TesnsorDataset(train_data_tensor, train_label_tensor)
        trainloader = data.Dataloader(train_tensor, batch_size = batch_size)

        valid_data_tensor  = torch.tensor(np.array(_valid_data.astype('f')))
        valid_label_tensor = torch.tensor(_valid_label)
        valid_tensor = data.TensorDataset(valid_data_tensor, valid_label_tensor)
        validloader = data.DataLoader(valid_tensor, batch_size = batch_size)

        for epoch in range(num_epoch):
            train_loss = 0
            train_acc = 0
            valid_loss = 0
            valid_acc = 0
            
            # 訓練モード
            model.train()
            for xt, yt in trainloader:
                xt = xt.to(device)
                yt = yt.to(device)

                y_pred = model.forward(xt)
                loss = loss_fn(y_pred, yt)

                train_loss += loss.item() * xt.size(0)
                train_acc += (y_pred.max(1)[1] == yt).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = train_loss / len(trainloader.dataset)
            avg_train_acc = train_acc / len(trainloader.dataset)

            # 検証モード
            model.eval()
            with torch.no_grad():
                for xv, yv in validloader:
                    xv = xv.to(device)
                    yv = yv.to(device)
                    y_pred = model(xv)
                    loss = loss_fn(y_pred, yv)
                    valid_loss += loss.item() * xv.size(0)
                    valid_acc += (y_pred.max(1)[1] == yv).sum().item()

                avg_valid_loss = valid_loss / len(validloader.dataset)
                avg_valid_acc = valid_acc / len(validloader.dataset)
            
            if epoch == 0 or (epoch + 1) % 100 == 0:
                print(f"epoch:{epoch + 1},train_loss:{avg_train_loss:.5f}, train_acc:{avg_train_acc:.5f}, val_loss:{avg_valid_loss:.5f},val_acc:{avg_valid_acc:.5f}")
        

