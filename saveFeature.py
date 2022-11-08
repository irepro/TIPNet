import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Dataset import ConcatDataset

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from utils import utilLoader
from utils import DEAPutils, SEEDutils, SEEDIVutils, Trackutils
import StoppedBandPredTaskLoss, StoppedBandPredTaskLoss_1C
import StoppedBandPathway, StoppedBandPathway_1C

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



idx = list(range(1,21))
CHBdataset = ConcatDataset.CHB_MIT_Dataset(idx)

batch_size = 64
tr_dataload = DataLoader(dataset=CHBdataset,
                             batch_size=batch_size, shuffle=False)
model = torch.load("../TIPNetPrac/save_model/95specPath_noGP.pth").to(device)

for batch_idx, (batch, label) in enumerate(tr_dataload):
    with torch.no_grad():
        feature_batch = model.getRep(batch.type(torch.float32))
        if batch_idx == 0:
            feature = np.array(feature_batch)
        else:
            feature = np.concatenate((feature, feature_batch.detach().numpy()), axis=0)

print(feature.shape)
np.save("../TIPNetPrac/save_model/spec_f_64_noGP", feature)
