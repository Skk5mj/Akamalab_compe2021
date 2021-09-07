import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()
def train_valid_loop(
    train_loader, valid_loader, valid_data_tensor, valid_label_tensor, model,
    n_epoch, optimizer):
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []
    auc_score_list = []

    for epoch in range(n_epoch):
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        # auc_score = 0

        model.train()
        for xt, yt in train_loader:
            xt = xt.to(device)
            yt = yt.to(device)

            y_pred = model.forward(xt)
            loss = loss_fn(y_pred, yt)

            train_loss += loss.item() * xt.size(0)
            train_acc += (y_pred.max(1)[1] == yt).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for xv, yv in valid_loader:
                xv = xv.to(device)
                yv = yv.to(device)

                y_pred = model(xv)
                loss = loss_fn(y_pred, yv)

                valid_loss += loss.item() * xv.size(0)
                valid_acc += (y_pred.max(1)[1] == yv).sum().item()

            avg_valid_loss = valid_loss / len(valid_loader.dataset)
            avg_valid_acc = valid_acc / len(valid_loader.dataset)

        if epoch == 0 or (epoch + 1 ) % 100 == 0:
            train_acc_list.append(avg_train_acc)
            train_loss_list.append(avg_train_loss)
            valid_acc_list.append(avg_valid_acc)
            valid_loss_list.append(avg_valid_loss)
        # 出来たモデルでauc scoreを計算
        _,prediction = torch.max(
            model.forward(valid_data_tensor.to(device)),dim=1)#fold全体の予測値
        # tensor配列からnumpy配列に戻すときはdetach()を挟む必要アリ
        auc_score = roc_auc_score(valid_label_tensor.detach().numpy().copy(),prediction.to('cpu').detach().numpy().copy())
        auc_score_list.append(auc_score)
    return train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, auc_score_list