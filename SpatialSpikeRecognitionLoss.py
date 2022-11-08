import os
import pickle as pkl
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
import numpy as np
from scipy.linalg import norm

from mne.filter import filter_data, notch_filter
from sklearn.metrics import f1_score

def accuracy_check(pred, label):
    prediction = np.argmax(pred, axis=1)
    lb = np.argmax(label, axis=1)

    compare = np.equal(lb, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())

    f1acc = f1_score(lb, prediction, average='micro')

    return accuracy, f1acc

def dotprod_sample(sample):
    length = sample.shape[0]
    sam = nn.functional.normalize(sample.reshape([length,-1]), dim=1)
    result = torch.matmul(sam, sam.T)

    return result


TRAIN = 0
VALIDATION = 1
TEST = 2

node_loc = {"AF3": (-2, 3), "F7": (-4, 2), "F3": (-2, 2), "FC5": (-3, 1), "T7": (-4, 3), "P7": (-4, -2),
                 "O1": (-1, -4), "O2": (1, -4), "P8": (4, -2),
                 "T8": (4, 0), "FC6": (3, 1), "F4": (2, 2), "F8": (4, 2), "AF4": (2, 4)}
selected_node = ["F7", "P7", "AF4", "P8"]
loc_x = [i[0] for i in node_loc.values()]
loc_y = [i[1] for i in node_loc.values()]
node_num = len(selected_node)
indexes = list(node_loc.keys())
scale=1
beta=2

def basisfunc( c, d):
    return scale * np.exp(-beta * norm(c - d) ** 2)

def RBF():
    # calculate activations of RBFs
    result = [0 for i in range(len(node_loc))]
    for c in selected_node:
        sum = 0
        for x in node_loc:
            sum += basisfunc(torch.tensor(node_loc[c]), torch.tensor(node_loc[x]))
        result[indexes.index(c)] = sum
    return result

rbfi = RBF()

class Cal_Loss(torch.nn.modules.loss._Loss):
    def __init__(self,bands, labels, device = None):
        super().__init__()
        self.device = device
        self.bands = bands
        self.labels = labels

    def forward(self, batch, encoder, train):
        index = indexes.index(selected_node[0])
        temp = batch[0]
        temp[index] += rbfi[index]
        x_data = temp
        x_data = x_data.unsqueeze(0)
        y_data = [self.labels[selected_node[0]]]
        for i in range(1, batch.shape[0] * node_num):
            index = indexes.index(selected_node[i //  batch.shape[0]])
            temp = batch[i % batch.shape[0]]
            temp = temp.unsqueeze(0)
            x_data = torch.cat((x_data, temp), dim=0)
            x_data[-1][index] += rbfi[index]
            y_data.append(self.labels[selected_node[i //  batch.shape[0]]])

        y_data = torch.Tensor(np.array(y_data))

        CrossEL = torch.nn.CrossEntropyLoss()
        pred = encoder.forward(x_data.type(torch.float32))
        loss = CrossEL(pred, y_data)

        if train == TRAIN:
            loss.backward(retain_graph=True)
        elif train == VALIDATION or train == TEST:
            acc = accuracy_check(pred.detach(), y_data)
            return loss, acc

        return loss
