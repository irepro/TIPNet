import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from Dataset import ConcatDataset
import TrendPredictTaskLoss_1Dconv
import TIPNet

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


TRAIN = 0
VALIDATION = 1
TEST = 2

# batch size
learning_rate = 0.01
epochs = 20
sfreq = 200

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = "cpu"
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#"cpu"

#dataset 몇개를 사용할 것인지 결정 ex)1~4

'''
ids = { "train":
    {
        "CHB-MIT":[list(range(1,2)),0.98],
        "DEAP":[list(range(1,2)),0.27],
        "SEED":[list(range(1,2)),None],
        "SEED-IV":[list(range(1,2)),None],
        "batch":5
    }, "validation" :
    {
        "CHB-MIT":[list(range(2,3)),0.3],
        "batch":8
    }, "test" :
    {
        "CHB-MIT":[list(range(2,3)),0.5],
        "batch":8
    }
}



concatdata = ConcatDataset.ConcatDataInit(ids)

tr_dataload = concatdata.getTrain()
val_dataload = concatdata.getVal()
te_dataload = concatdata.getTest()

'''

BANDS = ["Original", "MAF", "asln", "apn"]
LABEL = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

#concatdata = ConcatDataset.ConcatDataInit(ids)

idx = list(range(1,6))
CHBdataset = ConcatDataset.CHB_MIT_Dataset(idx)

train_size = int(0.8 * len(CHBdataset))
val_size = int(0.1 * len(CHBdataset))
test_size = len(CHBdataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(CHBdataset, [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(123456))

batch_size = 64
tr_dataload = DataLoader(dataset=train_dataset,
                             batch_size=batch_size, shuffle=False)
val_dataload = DataLoader(dataset=val_dataset,
                              batch_size=batch_size, shuffle=False)
te_dataload = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)


fs = [10,20,34]
#encode_info = [(16, 64, (1,5), 0)]
#model = TIPNet.TrendPredictPathway_GP(fs, bands=BANDS, dense=64).to(device)
encode_info = [(128, 128, 30, 0), (128, 128, 15, 0),(128, 128, 5, 0)]
model = TIPNet.StoppedBandPathway_1Dconv(sfreq, encode_info, Unsupervise=True, bands=BANDS, dense=128).to(device)
#model = MSNN.feature_extractor3(sfreq).to(device)

# Custom Tripletloss
criterion = TrendPredictTaskLoss_1Dconv.TemporalTrendIdentification_TaskLoss(BANDS, LABEL, device=device)

# use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# save epoch loss
loss_tr = []
loss_val = []
acc_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    #concatdata.getTrain()
    for batch_idx, (batch,label) in enumerate(tr_dataload):
        optimizer.zero_grad()

        loss_batch = criterion.forward(batch, model, TRAIN)
        optimizer.step()
        loss_ep += loss_batch.item()

        del loss_batch

    loss_tr.append(loss_ep)

    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        #concatdata.getVal()
        for batch_idx, (batch,label) in enumerate(val_dataload):
            loss_batch, acc_batch = criterion.forward(batch, model, VALIDATION)
            loss_v += loss_batch.item()
            acc_v += acc_batch[0]

        loss_val.append(loss_v)
        acc_v = acc_v/(batch_idx+1)

        acc_val.append(acc_v)
        # scheduler.step()
        print("epoch : ", epoch, "   train loss : ", str(loss_ep), "    val loss : ", str(loss_v), "    val acc : ", str(acc_v))

with torch.no_grad():
    loss_te = 0
    acc_te = 0
    #concatdata.getTrain()
    for batch_idx, (batch,label) in enumerate(te_dataload):
        loss_batch, acc_batch, sample, dot_prd = criterion.forward(batch, model, TEST)
        loss_te += loss_batch.item()
        acc_te += acc_batch[0]

    print("test loss : ", str(loss_te), "      test acc,f1 : ", str(acc_te/(batch_idx+1)))
    print("original  MAF    asln    apn")
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

torch.save(model, "../TIPNetPrac/save_model/"+ str(int(100*acc_te/(batch_idx+1))) + "tempPath_GP.pth")
