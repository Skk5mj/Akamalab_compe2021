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






