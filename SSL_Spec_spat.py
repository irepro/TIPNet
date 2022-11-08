import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from Dataset import ConcatDataset
import StoppedBandPredTaskLoss, StoppedBandPredTaskLoss_1Dconv
import TIPNet

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


TRAIN = 0
VALIDATION = 1
TEST = 2


BANDS = [(0.5,4), (4,8), (8,15), (15,30), (30,49.9)]
#BANDS = [(0.5,4), (4,8)]
LABEL = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
#LABEL = [[1,0],[0,1]]

# batch size
learning_rate = 0.1
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
        "CHB-MIT":[list(range(1,2)),0.25],
        #"DEAP":[list(range(1,2)),0.39],
        "SEED":[list(range(1,2)),None],
        "SEED-IV":[list(range(1,2)),None],
        "batch":5
    }, "validation" :
    {
        "CHB-MIT":[list(range(2,3)),0.24],
        "batch":20
    }, "test" :
    {
        "CHB-MIT":[list(range(3,4)),0.25],
        "batch": 15
    }
}

concatdata = ConcatDataset.ConcatDataInit(ids)


tr_dataload = concatdata.getTrain()
val_dataload = concatdata.getVal()
te_dataload = concatdata.getTest()
'''

idx = list(range(1,6))
CHBdataset = ConcatDataset.CHB_MIT_Dataset(idx)

train_size = int(0.8 * len(CHBdataset))
val_size = int(0.1 * len(CHBdataset))
test_size = len(CHBdataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(CHBdataset, [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(123456))
                                                                         
batch_size = 64
tr_dataload = DataLoader(dataset=train_dataset,
                             batch_size=batch_size, shuffle=True)
val_dataload = DataLoader(dataset=val_dataset,
                              batch_size=batch_size, shuffle=False)
te_dataload = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)

'''
ids = { "train":
    {
        "CHB-MIT":[list(range(1,2)),0.98],
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
#encode_info = [(128, 128, 30, "same"), (128, 128, 15, "same"),(128, 128, 5, "same")]
#encode_info = [(32, 32, 30, 0), (32, 32, 15, 0), (32, 32, 5, 0)]
encode_info = [(256, 256, 30, 0), (256, 256, 15, 0),(256, 256, 5, 0)]
#encode_info = [(16, 64, (1,5), 0)]
model = TIPNet.StoppedBandPathway_spaticonv(sfreq, encode_info, Unsupervise=True, bands=BANDS, dense=256).to(device)
#model = MSNN.feature_extractor3(sfreq).to(device)

# Custom Tripletloss
criterion = StoppedBandPredTaskLoss.StoppedBandPredTaskLoss(BANDS, LABEL, device=device)

# use SGD optimizer
CrossEL = torch.nn.CrossEntropyLoss()
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

        loss_batch = criterion.forward(CrossEL, batch, model, sfreq, TRAIN)
        optimizer.step()
        loss_ep += loss_batch.item()#/batch.shape[0]

        del loss_batch

    loss_tr.append(loss_ep)

    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        #concatdata.getVal()
        for batch_idx, (batch,label) in enumerate(val_dataload):
            loss_batch, acc_batch = criterion.forward(CrossEL, batch, model, sfreq, VALIDATION)
            loss_v += loss_batch.item()#/batch.shape[0]
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
        loss_batch, acc_batch = criterion.forward(CrossEL,batch, model, sfreq, TEST)
        loss_te += loss_batch.item()
        acc_te += acc_batch[0]
    '''
    timelength = sample.shape[-1]

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(range(timelength), sample[0])
    plt.title('original')
    plt.subplot(3, 2, 2)
    plt.plot(range(timelength), sample[1])
    plt.title('(0.5,4)')
    plt.subplot(3, 2, 3)
    plt.plot(range(timelength), sample[2])
    plt.title('(4,8)')
    plt.subplot(3, 2, 4)
    plt.plot(range(timelength), sample[3])
    plt.title('(8,15)')
    plt.subplot(3, 2, 5)
    plt.plot(range(timelength), sample[4])
    plt.title('(15,30)')
    plt.subplot(3, 2, 6)
    plt.plot(range(timelength), sample[5])
    plt.title('(30,49.9)')
    '''
    print("test loss : ", str(loss_te), "      test acc,f1 : ", str(acc_te/(batch_idx+1)))
    print("original  (0.5,4)   (4,8)   (8,15)   (15,30)   (30,49.9)")
    #print(dot_prd)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.subplot(2, 1, 2)
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

torch.save(model, "../TIPNetPrac/save_model/"+ str(int(100*acc_te/(batch_idx+1))) + "specspatPath.pth")
