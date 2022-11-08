import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import ConcatDataset
import models2
import SpatialTaskLoss

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


TRAIN = 0
VALIDATION = 1
TEST = 2

# batch size
batch_size = 3
learning_rate = 0.001
epochs = 20

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#"cpu"


#dataset 몇개를 사용할 것인지 결정 ex)1~4
ids = { "train":{
        "CHB-MIT":[list(range(1,19)),1.0],
        "batch":3
    }, "validation" : {
        "CHB-MIT":[list(range(19,20)),1.0],
        "batch":3
    }, "test" : {
        "CHB-MIT":[list(range(20,21)),1.0],
        "batch":3
    }
}
'''

ids = { "train":{
        "DREAMER":[3,1.0],
    }, "validation" : {
        "DREAMER":[4,1.0],
    }, "test" : {
        "DREAMER":[5,1.0],
    }
}
'''
concatdata = ConcatDataset.ConcatDataInit(ids)


FL = [8,10,12,15]
KL= [45,30,15,5]



model= models2.model_loader("SpatialNetwork",200, 23, 23,FL,KL).to(device)

# Custom Tripletloss
criterion =SpatialTaskLoss.PredTaskLoss(2, 4, device=device)


# use SGD optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# save epoch loss
loss_tr = []
loss_val = []
acc_val = []
for epoch in range(epochs):
    loss_ep = 0  # add batch loss in epoch
    for batch_idx, batch in enumerate(concatdata.getTrain()):
        optimizer.zero_grad()
        loss_batch = criterion.forward(batch, model,TRAIN)
        optimizer.step()
        loss_ep += loss_batch.item()

        del loss_batch

    loss_tr.append(loss_ep)

    with torch.no_grad():
        loss_v = 0
        acc_v = 0
        for batch_idx, batch in enumerate(concatdata.getVal()):
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
    for batch_idx, batch in enumerate(concatdata.getTest()):
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


torch.save(model,"/home/wypark/bci_project/model_save")
