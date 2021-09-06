import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def df2tensor(x_train,y_train, x_val, y_val, batch_size):
    train_data_tensor = torch.tensor(np.array(x_train.astype('f')))
    train_label_tensor = torch.tensor(y_train)
    train_tensor = data.TensorDataset(train_data_tensor, train_label_tensor)
    trainloader = data.DataLoader(train_tensor, batch_size = batch_size)

    valid_data_tensor = torch.tensor(np.array(x_val.astype('f')))
    valid_label_tensor = torch.tensor(y_val)
    valid_tensor = data.TensorDataset(valid_data_tensor, valid_label_tensor)
    validloader = data.DataLoader(valid_tensor, batch_size = batch_size)

    return trainloader, validloader

def train_valid_loop(train_loader, valid_loader, model, optimizer, loss_function, n_epoch):
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
            loss = loss_function(y_pred, yt)

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
                loss = loss_function(y_pred, yv)

                valid_loss += loss.item() + xv.size(0)
                valid_acc += (y_pred.max(1)[1] == yv).sum().item()

            avg_valid_loss = valid_loss / len(valid_loader.dataset)
            avg_valid_acc = valid_acc / len(valid_loader.dataset)

        if epoch == 0 or (epoch + 1 ) % 100 == 0:
            print("--------------------")
            print(f"epoch:{epoch + 1},\n train_loss:{avg_train_loss:.5f}, train_acc:{avg_train_acc:.5f}\n")
            print(f"valid_loss:{valid_loss:.5f}, valid_acc:{valid_acc:.5f}")
            print("--------------------")




