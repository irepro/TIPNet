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

def dotprod_sample(sample):
    length = sample.shape[0]
    sam = nn.functional.normalize(sample.reshape([length,-1]), dim=1)
    result = torch.matmul(sam, sam.T)

    return result


import numpy as np

def MAF(x, window):
    new_signals = []
    b,c,t = x.shape
    x = x.view(b*c, t)

    for i in range(int(b*c)):
        noise = np.convolve(x[i], np.ones(window), 'same') / window
        new_signals.append((x[i] - noise).numpy())

    new_signals = torch.Tensor(np.array(new_signals)).view(b,c,t)

    return np.array(new_signals)

def asln(x, A, B):
    b,c,t = x.shape
    x = x.view(b*c, t)

    noise = A * np.linspace(0, 1, t) - B * np.ones(t)
    noise = noise * np.ones((b*c, 1))
    n = np.random.randint(2)
    if n == 0:
        new_signals = x + noise
    else:
        new_signals = x - noise

    new_signals = new_signals.view(b,c,t)

    return np.array(new_signals)


def apn(x, C, D):
    b,c,t = x.shape
    x = x.view(b*c, t)

    noise = np.linspace(-np.pi + D, np.pi + D, t) - D * np.ones(t)
    noise = C * np.sin(noise) * np.ones((b*c, 1))
    new_signals = x + noise

    new_signals = new_signals.view(b,c,t)

    return np.array(new_signals)

TRAIN = 0
VALIDATION = 1
TEST = 2


class TemporalTrendIdentification_TaskLoss(torch.nn.modules.loss._Loss):
    def __init__(self, bands, labels, device=None):
        super().__init__()
        self.device = device
        self.BAND = bands
        self.LABEL = labels

    def forward(self, batch, encoder, train, window=50, A=50, B=0.75, C=50, D=0.75):
        batch_aug = []
        batch_label = []

        for idx,band in enumerate(self.BAND):
            if band == "Original" :
                temp = np.array(batch.clone())
            elif band == "MAF" :
                temp = MAF(batch, window)
            elif band == "asln" :
                temp = asln(batch,A,B)
            elif band == "apn":
                temp = apn(batch,C,D)

            batch_aug.append(temp)
            batch_label.append(temp.shape[0]*batch.shape[1] * [self.LABEL[idx]])

        # print('y:',y.shape)
        # print('rept:',rept.shape)
        b, a, c, l = np.array(batch_aug).shape
        batch_aug = np.array(batch_aug).reshape([b * a, c, l])
        batch_label = np.array(batch_label).reshape([b * a * c, len(self.LABEL)])

        CrossEL = torch.nn.CrossEntropyLoss()
        pred = encoder.forward(torch.Tensor(batch_aug))
        loss = CrossEL(pred, torch.Tensor(batch_label))

        if train == TRAIN:
            loss.backward(retain_graph=True)
        elif train == VALIDATION:
            acc = accuracy_check(pred.detach(), torch.Tensor(batch_label))
            return loss, acc
        elif train == TEST:
            #Calculate Similarity Matrix of Feature Vectors of original and augmentations
            sample_ori = batch[0,0,:].reshape([1,1,-1])

            for idx, band in enumerate(self.BAND):
                if band == "Original":
                    sample = sample_ori.clone()
                elif band == "MAF":
                    temp = MAF(sample_ori, window)
                    sample = torch.cat([sample, torch.tensor(temp)], dim=0)
                elif band == "asln":
                    temp = asln(sample_ori, A, B)
                    sample = torch.cat([sample, torch.tensor(temp)], dim=0)
                elif band == "apn":
                    temp = apn(sample_ori, C, D)
                    sample = torch.cat([sample, torch.tensor(temp)], dim=0)

            sample_Rept = encoder.getRep(sample.type(torch.float32))

            dotprod = dotprod_sample(sample_Rept.detach())
            acc = accuracy_check(pred.detach(), torch.Tensor(batch_label))
            return loss, acc, sample, np.array(dotprod)
        return loss