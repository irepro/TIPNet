import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from Dataset import ConcatDataset
import StoppedBandPathway_1C, Temporal_Dynamics_pathway, SpatialSpikeRecognitionModel
import TIPNet

import MSNN

import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# batch size
learning_rate = 0.01
epochs = 30
sfreq = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cpu"
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#"cpu"

#dataset 몇개를 사용할 것인지 결정 ex)1~4

idx = list(range(1,21))
CHBdataset = ConcatDataset.CHB_MIT_Dataset(idx)

train_size = int(0.8 * len(CHBdataset))
val_size = int(0.1 * len(CHBdataset))
test_size = len(CHBdataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(CHBdataset, [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(123456))

batch_size = 32
tr_dataload = DataLoader(dataset=train_dataset,
                             batch_size=batch_size, shuffle=False)
val_dataload = DataLoader(dataset=val_dataset,
                              batch_size=batch_size, shuffle=False)
te_dataload = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)

spectralPath = torch.load("../TIPNetPrac/save_model/96specPath.pth").to(device)
temporalPath = torch.load("../TIPNetPrac/save_model/Temporal207575_100Hz.pt").to(device)
#spatialPath = torch.load("").to(device)
#spatialPath =SpatialSpikeRecognitionModel.SpatialNetwork4(1000, in_channels=1, out_channels=64, ).to(device)
#model = MSNN.feature_extractor3(sfreq).to(device)

featureEncoder = TIPNet.FeatureEncoder_noSpatial(spectralPath, temporalPath)
statis = TIPNet.StatisticianModule(100, 2)
model = TIPNet.TIPNet(featureEncoder, statis)

CrossEL = torch.nn.BCELoss()
# use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.FeatureEncoder.spectral_path.requires_grad = False
model.FeatureEncoder.temporal_path.requires_grad = False

loss_tr = []
loss_val = []
acc_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    #concatdata.getTrain()
    for batch_idx, (inputs, labels) in enumerate(tr_dataload):
        optimizer.zero_grad()

        pred = model.forward(inputs)
        loss_batch = CrossEL(pred, labels)
        loss_batch.backward()

        optimizer.step()
        loss_ep += loss_batch.item()

        del loss_batch

    loss_tr.append(loss_ep)

    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        #concatdata.getVal()
        for batch_idx, (inputs, labels) in enumerate(val_dataload):
            pred = model.forward(inputs)
            loss_batch = CrossEL(pred, labels)

            acc_batch, _ = accuracy_check(pred, labels)
            loss_v += loss_batch.item()
            acc_v += acc_batch

        loss_val.append(loss_v)
        acc_v = acc_v/(batch_idx+1)

        acc_val.append(acc_v)
        # scheduler.step()
        print("epoch : ", epoch, "   train loss : ", str(loss_ep), "    val loss : ", str(loss_v), "    val acc : ", str(acc_v))

with torch.no_grad():
    loss_te = 0
    acc_te = 0
    #concatdata.getTrain()
    for batch_idx, batch in enumerate(te_dataload):
        loss_batch, acc_batch, sample, dot_prd = criterion.forward(batch, model, sfreq, TEST)
        loss_te += loss_batch.item()
        acc_te += acc_batch[0]
    print("test loss : ", str(loss_te), "      test acc,f1 : ", str(acc_te/(batch_idx+1)))
    print("original  (0.5,4)   (4,8)   (8,15)   (15,30)   (30,49.9)")
    print(dot_prd)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.subplot(2, 1, 2)
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

torch.save(model, "../TIPNetPrac/save_model/"+ str(int(100*acc_te/(batch_idx+1))) + "specPath.pth")
