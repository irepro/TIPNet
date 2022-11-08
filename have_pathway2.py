import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import StoppedBandPathway_1Dconv
import models2

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

class CHB_MIT_Dataset(Dataset):
    def __init__(self, idxs, ssl=True, sfreq=200, ratio = None):
        #self.path = os.getcwd()
        self.path = '../TIPNetPrac/data/CHB-MIT/'

        data = []
        labels = []

        for idx in idxs:
            tmp = np.load(self.path + f'Data_Sample{idx:02d}.npy')
            #y = np.load(self.path + f'Data_Label{idx:02d}.npy')
            y = np.load(self.path + f'Data_Label{idx:02d}.npy')

            if ratio != None:
                length = int(tmp.shape[0] * ratio)
                data.append(tmp[:length, :, :])
                labels.append(y[:length])
            else:
                data.append(tmp)
                labels.append(y)

        del tmp, y

        data = np.concatenate(data, axis=0)

        data = np.moveaxis(data, -1, 0)
        data = signal.resample(data, sfreq)

        self.data = np.moveaxis(data, 0, -1)

        labels = np.array(np.concatenate(labels, axis=0))
        labels = torch.tensor(labels).long()
        labels = nn.functional.one_hot(labels, num_classes=2)
        self.labels = labels
        self.ssl = ssl
        self.sfreq = sfreq

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

import numpy as np
import torch
import torch.nn as nn
import separableconv.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

idx = list(range(1,21))
CHBdataset = CHB_MIT_Dataset(idx)

train_size = int(0.8 * len(CHBdataset))
val_size = int(0.1 * len(CHBdataset))
test_size = len(CHBdataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(CHBdataset, [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(16))

from torch.utils.data import DataLoader

epochs = 20
learning_rate = 0.00001
batch_size = 64

trainLoader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
valLoader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
testLoader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

BANDS = [(0.5,4), (4,8), (8,15), (15,30), (30,49.9)]
encode_info = [(8, 16, 30, "same"), (16, 32, 15, "same"),(32, 64, 5, "same")]
FL = [8,16,32]
KL= [30,15,5]
fs = [10,20,34]

#specPath = TIPNet.StoppedBandPathway_1Dconv(fs=200,encode_info = encode_info).to(device)
specPath = torch.load("../TIPNetPrac/save_model/DIP/98b64tr1_6st2specPath_noGP.pth")
#tempPath = TIPNet.TrendPredictPathway_GP(fs, bands=BANDS, dense=64)
tempPath = torch.load("../TIPNetPrac/save_model/DIP/93b64tr1_6st2tempPath_GP.pth")
#spatiPath = torch.nn.DataParallel(models2.model_loader("SpatialNetwork",200, 23, 23,FL,KL))
#spatiPath = torch.load("../TIPNetPrac/save_model/all Pathway/spatial_model.pth")

CrossEL = torch.nn.CrossEntropyLoss()

featureencoder = TIPNet.FeatureEncoder_noSpatial(specPath, tempPath)
statistic = TIPNet.StatisticianModule_noSpatial(128, 2, [128,128])
model = TIPNet.DIPNet(featureencoder, statistic).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

loss_tr = []
loss_val = []
acc_tr = []
acc_val = []
f1_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    acc_ep = 0
    for batch_idx, (batch, label) in enumerate(trainLoader):
        optimizer.zero_grad()
        pred = model.forward(batch.type(torch.float32)).to(device)
        labels = label.type(torch.float64).to(device)
        # label = batch['label'].to(device)
        # print(pred.shape)
        loss = CrossEL(pred, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        acc, f1 = accuracy_check(pred.detach().numpy(), labels)
        #acc = acc / batch_size  # acc/(batch*channels*4(augmented))
        loss_ep += loss.item()
        # print('acc:', acc)
        acc_ep += acc

    loss_tr.append((loss_ep) / (batch_idx+1))
    acc_tr.append((acc_ep) / (batch_idx+1))

    loss_ep_val = 0
    acc_ep_val = 0

    for batch_idx, (batch, label) in enumerate(valLoader):
        pred = model.forward(batch.type(torch.float32)).to(device)
        labels = label.type(torch.float64).to(device)
        loss = CrossEL(pred, labels)

        acc, f1 = accuracy_check(pred.detach().numpy(), labels)

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

    for batch_idx, (batch, label) in enumerate(testLoader):
        pred = model.forward(batch.type(torch.float32)).to(device)
        labels = label.type(torch.float64).to(device)
        loss = CrossEL(pred, labels)

        acc, f1 = accuracy_check(pred.detach().numpy(), labels)
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