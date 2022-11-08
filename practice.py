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

x = torch.Tensor(np.random.randn(32,23,128,200))

b ,c ,f, t = x.shape

spatial_layer = nn.Conv2d(128, 128, (1, 1))

out = torch.matmul(x.view(b,f,c,t), x.view(b,f,t,c))
out = spatial_layer(out)

print(out.shape)