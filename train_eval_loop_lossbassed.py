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

def train_valid_loop_lossbased(
    train_loader, valid_loader, valid_data_tensor, valid_label_tensor, model,
    n_epoch, optimizer):
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []
    auc_score_list = []
    patience = 0
    best_auc_score = 0
    smllst_train_loss = 1000

    for epoch in range(n_epoch):
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0

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

                y_pred = model.forward(xv)
                loss = loss_fn(y_pred, yv)

                valid_loss += loss.item() * xv.size(0)
                valid_acc += (y_pred.max(1)[1] == yv).sum().item()

            avg_valid_loss = valid_loss / len(valid_loader.dataset)
            avg_valid_acc = valid_acc / len(valid_loader.dataset)

        if epoch == 0 or (epoch + 1 ) % 10 == 0:
            train_acc_list.append(avg_train_acc)
            train_loss_list.append(avg_train_loss)
            valid_acc_list.append(avg_valid_acc)
            valid_loss_list.append(avg_valid_loss)
        # 出来たモデルでauc scoreを計算
        # _,prediction = torch.max(
        #     model.forward(valid_data_tensor.to(device)),dim=1)#fold全体の予測値
        # 確率値をラベルに置き換えてしまうとaucスコアは下がってしまう
        # 従って確率値のまま出力するように変更
        prediction = model.forward(valid_data_tensor.to(device))

        # tensor配列からnumpy配列に戻すときはdetach()を挟む必要アリ
        auc_score = roc_auc_score(valid_label_tensor.detach().numpy().copy(),prediction[:,1].to('cpu').detach().numpy().copy())
        auc_score_list.append(auc_score)
        if auc_score > best_auc_score:
            best_auc_score = auc_score
            # 更新されたときだけモデルを保存しておく
            # torch.save(model.state_dict(), "model.pth")
            # earlystoppingの基準を訓練誤差に変える
        if avg_train_loss < smllst_train_loss:
            smllst_train_loss = avg_train_loss
            torch.save(model.state_dict(), "model.pth")
        else:
            patience += 1 # 訓練誤差が10回下がらなかったら学習をストップ
        # Early Stopping
        if patience == 10:
            print(f"Early Stopping is working in {epoch}th epoch.")
            print(f"Best auc score in validation is {best_auc_score}")
            break
    return train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, auc_score_list