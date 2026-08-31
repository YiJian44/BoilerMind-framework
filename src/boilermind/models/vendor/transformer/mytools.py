import pandas as pd
import numpy as np
import torch
import math

def DataFrame(input):
    # 输入数据格式预处理，就是加一下变量名字
    temp_for_label = 0
    series_list = []
    for col_name, col_data in input.items():
        series_name = 'V_' + str(temp_for_label)
        temp_for_label = temp_for_label + 1
        series = pd.Series(col_data, name=series_name)
        series_list.append(series)
    df = pd.concat(series_list, axis=1)
    return df

def standardization(data):
    mu = np.min(data)
    sigma = np.max(data) - np.min(data)
    return (data - mu) / sigma

def reverse_standardization(x, data):
    mu = np.min(data)
    sigma = np.max(data) - np.min(data)
    return x * sigma + mu

def my_FFT_for_Period(x, k=5):
    xf = np.fft.rfft(x.T)
    frequency_list = abs(xf).mean(0)
    frequency_list[0] = 0
    sorted_indices = np.argsort(-frequency_list)
    period = x.shape[0] // (sorted_indices[:k]+1)
    return period

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # elif val_loss <=0.001:
        #     self.early_stop = True
        elif math.isnan(val_loss):
            self.early_stop = True
            print('loss: nan')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss