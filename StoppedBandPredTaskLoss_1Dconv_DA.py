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


TRAIN = 0
VALIDATION = 1
TEST = 2

class StoppedBandPredTaskLoss(torch.nn.modules.loss._Loss):
    def __init__(self, bands, labels, device = None):
        super().__init__()
        self.device = device
        self.BAND = bands
        self.LABEL = labels

    def forward(self, batch, label, encoder, sfreq, train, max_idx):
        batch_aug = []
        batch_label = []
        #Band has freqeuncy range, filter_data plays the role of Band Stop filter
        #each Labels are composed of one-hot vectors and are copied ( Bands size * Batch size )
        for idx,band in enumerate(self.BAND):
            lfreq, rfreq = band
            data = filter_data(batch.numpy().astype(np.float64), sfreq=sfreq, l_freq=rfreq, h_freq=lfreq, verbose=False)
            batch_aug.append(data)
            batch_label.append(data.shape[0]*batch.shape[1] * [self.LABEL[idx]])

        #[batch, augmentation, channel, time length] -> [batch*augmentation, channel, time length]
        b,a,c,l = np.array(batch_aug).shape
        batch_aug = np.array(batch_aug).reshape([b*a,c,l])
        batch_label = np.array(batch_label).reshape([b*a*c,len(self.LABEL)])

        #Self-supervised Learning Loss is CrossEntropy
        CrossEL = torch.nn.CrossEntropyLoss()
        pred = encoder.forward(torch.Tensor(batch_aug))
        dom_pred = encoder.getDomainfeature(torch.Tensor(batch.numpy().astype(np.float64)))
        ssl_loss = CrossEL(pred, torch.Tensor(batch_label))

        la = label.flatten().squeeze()
        on = nn.functional.one_hot(la, num_classes = max_idx+1)

        #da_loss = - CrossEL(dom_pred, nn.functional.one_hot(label.flatten(), num_classes = max_idx+1))
        loss = ssl_loss #+ da_loss

        if train == TRAIN:
            loss.backward(retain_graph=True)
        elif train == VALIDATION:
            acc = accuracy_check(pred.detach(), torch.Tensor(batch_label))
            return loss, acc
        elif train == TEST:
            #Calculate Similarity Matrix of Feature Vectors of original and augmentations
            sample_ori = np.array(batch[0,0,:].reshape([1,1,-1]))
            sample = sample_ori
            for idx, band in enumerate(self.BAND):
                lfreq, rfreq = band
                data = filter_data(sample_ori.astype(np.float64), sfreq=sfreq, l_freq=rfreq, h_freq=lfreq,
                                   verbose=False)
                sample = np.concatenate((sample, data), axis=0)
            sample_Rept = encoder.getRep(torch.Tensor(sample))

            dotprod = dotprod_sample(sample_Rept.detach())
            acc = accuracy_check(pred.detach(), torch.Tensor(batch_label))
            return loss, acc, sample, np.array(dotprod)
        return loss
