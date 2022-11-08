import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from Dataset import ConcatDataset
import StoppedBandPredTaskLoss, StoppedBandPredTaskLoss_1C
import StoppedBandPathway, StoppedBandPathway_1C
import MSNN

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
ids = { "train":
    {
        "CHB-MIT":[list(range(1,2)),0.98],
        "DEAP":[list(range(1,2)),0.27],
        "SEED":[list(range(1,2)),None],
        "SEED-IV":[list(range(1,2)),None],
        "batch":5
    }, "validation" :
    {
        "CHB-MIT":[list(range(2,3)),0.15],
        "batch":8
    }, "test" :
    {
        "DEAP":[list(range(2,3)),0.3],
        "batch":8
    }
}

BANDS = [(0.5,4), (4,8), (8,15), (15,30), (30,49.9)]
#BANDS = [(0.5,4), (4,8)]
LABEL = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
#LABEL = [[1,0],[0,1]]

concatdata = ConcatDataset.ConcatDataInit(ids)

electrode = 32
encode_info = [(8, 16, (1,30), 0), (16, 32, (1,15), 0),(32, 64, (1,5), 0)]
#encode_info = [(16, 64, (1,5), 0)]
model = StoppedBandPathway_1C.StoppedBandPathway(sfreq,electrode,True,encode_info,BANDS).to(device)
#model = MSNN.feature_extractor3(sfreq).to(device)

# Custom Tripletloss
criterion = StoppedBandPredTaskLoss_1C.StoppedBandPredTaskLoss(BANDS, LABEL, device=device)


# use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# save epoch loss
loss_tr = []
loss_val = []
acc_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    for batch_idx, batch in enumerate(concatdata.getTrain()):
        optimizer.zero_grad()

        loss_batch = criterion.forward(batch, model, sfreq, TRAIN)
        optimizer.step()
        loss_ep += loss_batch.item()

        del loss_batch

    loss_tr.append(loss_ep)

    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        for batch_idx, batch in enumerate(concatdata.getVal()):
            loss_batch, acc_batch = criterion.forward(batch, model, sfreq, VALIDATION)
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
    for batch_idx, batch in enumerate(concatdata.getTrain()):
        loss_batch, acc_batch, sample, dot_prd = criterion.forward(batch, model, sfreq, TEST)
        loss_te += loss_batch.item()
        acc_te += acc_batch[0]
    '''
    timelength = sample.shape[-1]

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(range(timelength), sample[0,0,:])
    plt.title('original')
    plt.subplot(3, 2, 2)
    plt.plot(range(timelength), sample[1,0,:])
    plt.title('(0.5,4)')
    plt.subplot(3, 2, 3)
    plt.plot(range(timelength), sample[2,0,:])
    plt.title('(4,8)')
    plt.subplot(3, 2, 4)
    plt.plot(range(timelength), sample[3,0,:])
    plt.title('(8,15)')
    plt.subplot(3, 2, 5)
    plt.plot(range(timelength), sample[4,0,:])
    plt.title('(15,30)')
    plt.subplot(3, 2, 6)
    plt.plot(range(timelength), sample[5,0,:])
    plt.title('(30,49.9)')
    '''
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
