import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import StoppedBandPathway_1Dconv

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import separableconv.nn as nn
from Dataset import ConcatDataset
import TIPNet
from scipy import signal
from torch.utils.data import Dataset
import torch
import numpy as np


from sklearn.metrics import f1_score

def accuracy_check(pred, label):
    prediction = np.argmax(pred, axis=1)
    lb = np.argmax(label, axis=1)

    compare = np.equal(lb, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())

    f1acc = f1_score(lb, prediction, average='micro')

    return accuracy, f1acc

class CHB_F_Dataset(Dataset):
    def __init__(self, idxs, ssl=True, sfreq=200, ratio=None):
        # self.path = os.getcwd()
        self.path = '../TIPNetPrac/data/CHB-MIT/'

        # temporal
        self.temporal_rept = np.load("../TIPNetPrac/save_model/rept_100Hz207575.npy")
        # data = torch.tensor(data)
        # data = torch.mean(data, 1) # channel
        # temporal_rept = torch.mean(data, 2) # time

        # sepctral
        self.spectral_rept = np.load("../TIPNetPrac/save_model/spec_f_64_noGP.npy")
        # data = torch.tensor(data)
        # spectral_rept = torch.mean(data, 1) # channel

        # spatial
        self.spatial_rept = np.load("../TIPNetPrac/save_model/spatial_feature.npy")

        labels = []
        for idx in idxs:
            y = np.load(self.path + f'Data_Label{idx:02d}.npy')
            labels.append(y)

        del y
        labels = np.array(np.concatenate(labels, axis=0))
        labels = torch.tensor(labels).long()
        labels = nn.functional.one_hot(labels, num_classes=2)

        # data = np.concatenate(data, axis=0)

        # data = signal.resample(data, sfreq)
        self.labels = labels
        self.ssl = ssl
        self.sfreq = sfreq

    def __len__(self):
        return self.temporal_rept.shape[0]

    def __getitem__(self, item):
        return {'temporal': torch.tensor(self.temporal_rept[item]),
                'spectral': torch.tensor(self.spectral_rept[item]),
                'spatial': torch.tensor(self.spatial_rept[item]),
                'label': self.labels[item]
                }


import numpy as np
import torch
import torch.nn as nn
import separableconv.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()

        #self.spec_GAP = nn.AdaptiveAvgPool1d(1)
        self.spec_GAP = nn.AdaptiveAvgPool2d((1,1))
        self.temp_GAP = nn.AdaptiveAvgPool2d((1,1))
        self.spec_spatconv = nn.Conv1d(64, 64, 23)
        self.temp_spatconv = nn.Conv1d(64, 64, 23)

    def spectral_GAP(self, x):
        #  x.shape = batch,  channel, feature, time -->
        #batch, channel, feature = x.shape
        #x = x.view(batch,feature,channel) # channel

        batch, channel, feature, time = x.shape
        x = x.view(batch,feature,channel,time) # channel
        #x = self.spec_spatconv(x).squeeze()
        x = self.spec_GAP(x).squeeze()  # time
        #x = self.spec_GAP(x).squeeze()  # time

        '''
        batch, channel, feature = x.shape
        x = x.view(batch, feature, channel)  # channel
        x = self.spec_spatconv(x).squeeze()
        '''
        return x

    def spectral_GVP(self, x):
        #  x.shape = batch,  channel, feature, time -->
        #batch, channel, feature = x.shape
        #x = x.view(batch,feature,channel) # channel

        batch, channel, feature, time = x.shape
        x = x.view(batch,feature,channel,time) # channel

        #x = self.spec_spatconv(x).squeeze()
        x = torch.var(x, -1).squeeze()
        x = torch.var(x, -1).squeeze()  # time
        '''
        batch, channel, feature = x.shape
        x = x.view(batch, feature, channel)  # channel
        x = self.spec_spatconv(x).squeeze()

        #x = torch.var(x, -1) # channel
        '''
        return x


    def Temporal_GAP(self, x):
        #  x.shape = batch,  channel, feature, time -->
        batch, channel, feature, time = x.shape
        x = x.view(batch,feature,channel,time) # channel
        x = self.temp_GAP(x)  # time
        return x.squeeze()

    def Temporal_GVP(self, x):
        #  x.shape = batch,  channel, feature, time -->
        #batch, v channel, feature, v time
        #2D
        #batch, feature, time,
        batch, channel, feature, time = x.shape
        x = x.view(batch,feature,channel,time) # channel

        x = torch.var(x, -1).squeeze()
        x = torch.var(x, -1).squeeze()
        #x = self.temp_spatconv(x).squeeze() # time
        #x = torch.var(x, -1)  # channel
        return x

    def forward(self, x):
        f_1 = x['spectral']
        #f_1 = self.spectral_encoder.getRep(x)
        f_2 = x['temporal']

        f_GAP = torch.cat((self.spectral_GAP(f_1), self.Temporal_GAP(f_2), x['spatial']), axis=1)
        f_GVP = torch.cat((self.spectral_GVP(f_1), self.Temporal_GVP(f_2), x['spatial']), axis=1)

        #f_GAP = self.spectral_GAP(f_1)
        #f_GVP = self.spectral_GVP(f_1)

        return f_GAP, f_GVP


class StatisticianModule(nn.Module):
    def __init__(self, dense, classes):
        super(StatisticianModule, self).__init__()
        self.classes = classes

        self.softmax = torch.nn.Softmax(dim=1)
        self.c_dense = nn.Linear(int(64 * 2 * 2 + 60*2), dense)  # 64 * 2 + 64 * 2 + 60 * 2 = 376 = sum(i=3) represntation*2

        self.gap_pwconv = nn.Conv1d(int(64*2 + 60), dense, 1)
        self.gvp_pwconv = nn.Conv1d(int(64*2 + 60), dense, 1)

        self.fullconnect = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(int(64*2 + 60), self.classes)
        )

    def forward(self, f_GAP, f_GVP):
        # [batch, gap+gvp] -> [batch,dense]

        c = self.softmax(self.c_dense(torch.cat((f_GAP, f_GVP), axis=1)))

        # [batch, gap, 1] -> [batch, 1, dense] -> [batch, dense]
        f_GAP_d = self.gap_pwconv(f_GAP.unsqueeze(dim=-1)).squeeze()
        f_GVP_d = self.gvp_pwconv(f_GVP.unsqueeze(dim=-1)).squeeze()

        f_GAP_dd = torch.sum(c * f_GAP_d, dim=1)
        f_GVP_dd = torch.sum(c * f_GVP_d, dim=1)

        ALN = torch.div(torch.sub(f_GAP.T, f_GAP_dd), f_GAP_dd).T

        y_hat = self.softmax(self.fullconnect(ALN))
        # print('y_hat: ',y_hat.shape)
        return y_hat

class TIPNet(nn.Module):
    def __init__(self, FeatureEncoder, StatisticianModule):
        super(TIPNet, self).__init__()
        self.FeatureEncoder = FeatureEncoder
        self.StatisticianModule = StatisticianModule

    def forward(self, x):
        f_GAP, f_GVP = self.FeatureEncoder(x)
        y_hat = self.StatisticianModule(f_GAP, f_GVP)
        return y_hat

idx = list(range(1,21))
CHBdataset = CHB_F_Dataset(idx)

train_size = int(0.8 * len(CHBdataset))
val_size = int(0.1 * len(CHBdataset))
test_size = len(CHBdataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(CHBdataset, [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(16))

from torch.utils.data import DataLoader

epochs = 30
learning_rate = 0.00001
batch_size = 64

trainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
valLoader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
testLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

#encode_info = [(8, 16, 30, "same"), (16, 32, 15, "same"),(32, 64, 5, "same")]
#specPath = StoppedBandPathway_1Dconv.StoppedBandPathway(sfreq=100,electrode=23,Unsupervise=False,encode_info = encode_info,BANDS=None).to(device)

CrossEL = torch.nn.CrossEntropyLoss()

featureencoder = FeatureEncoder()
statistic = StatisticianModule(376,2)
model = TIPNet(featureencoder, statistic)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

loss_tr = []
loss_val = []
acc_tr = []
acc_val = []
f1_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    acc_ep = 0
    for batch_idx, batch in enumerate(trainLoader):
        optimizer.zero_grad()
        pred = model.forward(batch).to(device)
        label = batch['label'].type(torch.float64).to(device)
        # label = batch['label'].to(device)
        # print(pred.shape)
        loss = CrossEL(pred, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        acc, f1 = accuracy_check(pred.detach().numpy(), label)
        #acc = acc / batch_size  # acc/(batch*channels*4(augmented))
        loss_ep += loss.item()
        # print('acc:', acc)
        acc_ep += acc

    loss_tr.append((loss_ep) / (batch_idx+1))
    acc_tr.append((acc_ep) / (batch_idx+1))

    loss_ep_val = 0
    acc_ep_val = 0

    for batch_idx, batch in enumerate(valLoader):
        pred = model.forward(batch).to(device)
        label = batch['label'].type(torch.float64).to(device)
        loss = CrossEL(pred, label)

        acc, f1 = accuracy_check(pred.detach().numpy(), label)

        loss_ep_val += loss.item()
        acc_ep_val += acc

    loss_val.append((loss_ep_val) / (batch_idx+1))
    acc_val.append((acc_ep_val) / (batch_idx+1))
    print("epoch : ", epoch, "  train loss : ", loss_tr[epoch], 'train acc : ', acc_tr[epoch], "    val loss : ",
          loss_val[epoch], 'val acc : ', acc_val[epoch])


with torch.no_grad():
    loss_te = 0
    acc_te = 0
    f1_te = 0

    for batch_idx, batch in enumerate(testLoader):
        pred = model.forward(batch).to(device)
        label = batch['label'].type(torch.float64).to(device)
        loss = CrossEL(pred, label)

        acc, f1 = accuracy_check(pred.detach().numpy(), label)
        loss_te += loss.item()
        acc_te += acc
        f1_te += f1

    print("    test loss : ", loss_te/(batch_idx+1), 'test acc : ', acc_te/(batch_idx+1), 'test f1 : ', f1_te/(batch_idx+1))

import matplotlib.pyplot as plt

# plt.figure(figsize =(15, 10))
plt.plot(range(epochs), loss_tr, color='red')
plt.plot(range(epochs), loss_val, color='blue')
plt.title('Model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('CHB_100Hz_loss.png', bbox_inches='tight')
plt.show()

# plt.figure(figsize =(15, 10))
plt.plot(range(epochs), acc_tr, color='red')
plt.plot(range(epochs), acc_val, color='blue')
plt.title('Model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('CHB_100Hz_accuracy.png',bbox_inches = 'tight')
plt.show()