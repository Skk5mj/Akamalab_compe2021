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
    """[訓練/検証ループを実行。分割されたデータと対応するラベル、訓練したいモデルなどを
    入力して10epochごとの結果を出力。10回AUCスコアが更新されなければ学習停止し、最良のモデルを保存。
    ]

    Args:
        train_loader ([dataloader]]): [訓練データローダ]
        valid_loader ([dataloader]): [検証データローダ]
        valid_data_tensor ([torch.tensor]): [AUCスコア計算時に用いるための検証データテンソル]
        valid_label_tensor ([torch.tensor]): [同上の目的のラベル]
        model ([obj?]): [訓練したいモデル]
        n_epoch ([Int]): [epoch数]
        optimizer ([obj?]): [用いる最適化アルゴリズム]

    Returns:
        [dict]: [訓練と検証における正解率と誤差、各epochの検証におけるAUCスコアのリストを
        まとめた辞書を返す]
    """
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []
    auc_score_list = []
    patience = 0 # Earlystoppingに用いる変数
    best_auc_score = 0 

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
        # _,prediction = torch.max(
        #     model.forward(valid_data_tensor.to(device)),dim=1)#fold全体の予測値
        # AUCスコアで競う場合には予測ラベルを提出するのではなく、確率値で出力したほうが
        # スコアが出やすい。ただし提出方式が確率値でもよい場合のみなので注意
        prediction = model.forward(valid_data_tensor.to(device))

        # tensor配列からnumpy配列に戻すときはdetach()を挟む必要アリ
        auc_score = roc_auc_score(valid_label_tensor.detach().numpy().copy(),prediction.to('cpu').detach().numpy().copy())
        auc_score_list.append(auc_score)

        if auc_score > best_auc_score:
            best_auc_score = auc_score
            # 更新されたときだけモデルを保存しておく
            torch.save(model.state_dict(), "model.pth")
        else:
            # AUCスコアの向上がなければpatienceを1増やす
            patience += 1
        # Early Stopping
        if patience == 10:
            # 10epochにわたってAUCスコアの更新がなければ学習を終了させる。
            print(f"Early Stopping is working in {epoch}th epoch.")
            print(f"Best auc score in validation is {best_auc_score}")
            break
    results = {
        "train_acc_list": train_acc_list, "train_loss_list": train_loss_list,
        "valid_acc_list": valid_acc_list, "valid_loss_list": valid_loss_list,
        "auc_score_list": auc_score_list
        }
    return results