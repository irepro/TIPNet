import numpy as np
import torch
import torch.nn as nn
from mne.filter import filter_data, notch_filter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.linalg import norm

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


'''
def accuracy_check(pred, label):
    prediction = np.argmax(pred, axis=1)
    lb = np.argmax(label, axis=1)

    compare = np.equal(lb, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())

    f1acc = f1_score(lb, prediction, average='micro')

    return accuracy, f1acc
'''

TRAIN = 0
VALIDATION = 1
TEST = 2

class PredTaskLoss(torch.nn.modules.loss._Loss):
    def __init__(self, beta, scale, device = None):
        super().__init__()
        self.device = device

        '''
        self.node_loc = {"AF3": (-2, 3), "F7": (-4, 2), "F3": (-2, 2), "FC5": (-3, 1), "T7": (-4, 3), "P7": (-4, -2),
                    "O1": (-1, -4), "O2": (1, -4), "P8": (4, -2),
                    "T8": (4, 0), "FC6": (3, 1), "F4": (2, 2), "F8": (4, 2), "AF4": (2, 4)}
        self.selected_node = ["F7", "P7", "AF4", "P8"]
        '''
        self.node_loc = ['FP1-F7',
 'F7-T7',
 'T7-P7',
 'P7-O1',
 'FP1-F3',
 'F3-C3',
 'C3-P3',
 'P3-O1',
 'FP2-F4',
 'F4-C4',
 'C4-P4',
 'P4-O2',
 'FP2-F8',
 'F8-T8',   #
 'T8-P8',
 'P8-O2',
 'FZ-CZ',   #
 'CZ-PZ',   #
 'P7-T7',  #
 'T7-FT9',
 'FT9-FT10',
 'FT10-T8',
 'T8-P8']
        self.selected_node=['F8-T8', 'FZ-CZ', 'CZ-PZ', 'P7-T7']
        self.labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

        #self.indexes = list(self.node_loc.keys())
        self.indexes = self.node_loc

        #self.loc_x = [i[0] for i in self.node_loc.values()]
        #self.loc_y = [i[1] for i in self.node_loc.values()]


        #data=self.preprocessing(data)
        self.beta = beta
        self.scale= scale
        #self.rbf= self.RBF()
        self.rbf= self.RBF()

    def basisfunc(self, c, d):
        return self.scale*np.exp(-self.beta * norm(c - d) ** 2)

    def RBF(self):
        # calculate activations of RBFs
        result=np.zeros(len(self.node_loc))
        for c in self.selected_node :
            sum=0
            for x in self.node_loc :
                #sum += self.basisfunc(np.array(self.node_loc[c]),np.array(self.node_loc[x]))
                sum=1*self.scale
            result[self.indexes.index(c)] = sum
        return result



    def forward(self, batch, encoder, train):
        node_num = len(self.selected_node)
        batch_aug= batch.repeat(node_num, 1,1)
        batch_num=batch.shape[0]

        batch_label = []
        for i in range(batch_aug.shape[0]):
            index = self.indexes.index(self.selected_node[i//batch_num])
            batch_aug[i][index] += self.rbf[index]
            batch_label.append(self.labels[i//batch_num])
        batch_label = torch.Tensor(np.array(batch_label))
        #batch_label =np.array([i//batch_num for i in range(batch_aug.shape[0])])

        CrossEL = torch.nn.CrossEntropyLoss()
        pred = encoder.forward(batch_aug.type(torch.float32))
        loss = CrossEL(pred, torch.Tensor(batch_label))
        if train == TRAIN:
            loss.backward(retain_graph=True)
        elif train == TEST:
            '''
            rept = encoder.getRep(torch.Tensor(batch_aug))
            ori_rept = encoder.getRep(batch)

            xplot = np.concatenate((ori_rept.detach().numpy(), rept.detach().numpy()))
            split = int(xplot.shape[0] / 6)
            yplot = [0]*split + [1]*split + [2]*split + [3]*split + [4]*split + [5]*split

            hyp.plot(xplot, '.', hue=yplot, ndims=2)'''
            pass
        if train == VALIDATION or train == TEST:
            acc = accuracy_check(pred.detach(), torch.Tensor(batch_label))
            return loss, acc
        return loss