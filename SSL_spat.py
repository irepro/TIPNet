import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from Dataset import ConcatDataset
import TIPNet
import SpatialTaskLoss

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


TRAIN = 0
VALIDATION = 1
TEST = 2

# batch size
batch_size = 32
learning_rate = 0.001
epochs = 20

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#"cpu"


idx = list(range(1,21))
CHBdataset = ConcatDataset.CHB_MIT_Dataset(idx)

train_size = int(0.8 * len(CHBdataset))
val_size = int(0.1 * len(CHBdataset))
test_size = len(CHBdataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(CHBdataset, [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(123456))
tr_dataload = DataLoader(dataset=train_dataset,
                             batch_size=batch_size, shuffle=False)
val_dataload = DataLoader(dataset=val_dataset,
                              batch_size=batch_size, shuffle=False)
te_dataload = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)


FL = [8,16,32]
KL= [30,15,5]

model= TIPNet.SpikeRecognitionPathway_GP_ver2(200, 23, 23,FL,KL).to(device)

# Custom Tripletloss
criterion =SpatialTaskLoss.PredTaskLoss(2, 1)

# use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# save epoch loss
loss_tr = []
loss_val = []
acc_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    for batch_idx, (batch, label) in enumerate(tr_dataload):
        optimizer.zero_grad()

        loss_batch = criterion.forward(batch, model,TRAIN)
        optimizer.step()
        loss_ep += loss_batch.item()

        del loss_batch

    loss_tr.append(loss_ep)

    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        for batch_idx, (batch, label) in enumerate(val_dataload):
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
    for batch_idx, (batch, label) in enumerate(te_dataload):
        loss_batch, acc_batch = criterion.forward(batch, model, TEST)
        loss_te += loss_batch.item()
        acc_te += acc_batch[0]
    print("test loss : ", str(loss_te), "      test acc,f1 : ", str(acc_te/(batch_idx+1)))

plt.subplot(2, 1, 1)
plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.subplot(2, 1, 2)
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


torch.save(model, "../TIPNetPrac/save_model/"+ str(int(100*acc_te/(batch_idx+1))) + "spatPath_GP.pth")
