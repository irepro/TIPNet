import numpy as np
import torch
import torch.nn as nn
from mne.filter import filter_data, notch_filter
import matplotlib.pyplot as plt
import hypertools as hyp
from sklearn.metrics import f1_score

def accuracy_check(pred, label):
    prediction = np.argmax(pred, axis=1)
    lb = np.argmax(label, axis=1)

    compare = np.equal(lb, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())

    f1acc = f1_score(lb, prediction, average='micro')

    return accuracy, f1acc

TRAIN = 0
VALIDATION = 1
TEST = 2

class StoppedBandPredTaskLoss(torch.nn.modules.loss._Loss):
    def __init__(self, bands, labels, device = None):
        super().__init__()
        self.device = device
        self.BAND = bands
        self.LABEL = labels

    def forward(self, CrossEL, batch, encoder, sfreq, train):
        batch_aug = []
        batch_label = []
        for idx,band in enumerate(self.BAND):
            lfreq, rfreq = band
            data = filter_data(batch.numpy().astype(np.float64), sfreq=sfreq, l_freq=lfreq, h_freq=rfreq, verbose=False)
            batch_aug.append(data)
            batch_label.append(data.shape[0] * [self.LABEL[idx]])

        '''
        plt.subplot(2, 1, 1)
        plt.plot(range(len(batch[0,0,:])), batch[0,0,:], label='Loss', color='red')
        plt.subplot(2, 1, 2)
        plt.plot(range(len(data[0,0,:])), data[0,0,:], label='Loss', color='blue')
        plt.show()'''

        b,a,c,l = np.array(batch_aug).shape
        batch_aug = np.array(batch_aug).reshape([b*a,c,l])
        batch_label = np.array(batch_label).reshape([b*a,5])

        pred = encoder.forward(torch.Tensor(batch_aug))
        loss = CrossEL(pred, torch.Tensor(batch_label))

        if train == TRAIN:
            loss.backward(retain_graph=True)

            del batch_aug, batch_label
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
