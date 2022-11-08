import numpy as np
import torch
import torch.nn as nn
import separableconv.nn as nn

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, nn.SeparableConv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)

    if isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)

class SBP_raw_Encoder(nn.Module):
    def __init__(self, fs, electrode, encode_info):
        super(SBP_raw_Encoder, self).__init__()
        self.fs = fs
        self.elu = nn.ELU()
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.spectral_layer = nn.Conv2d(1, 4, (1, int(self.fs/2)))
        self.spatial_layer = nn.Conv2d(4, 16, (electrode, 1))

        self.bac_layer = nn.Sequential()
        for i, arg in enumerate(encode_info):
            input_dim, output_dim, kernel_size, padding = arg
            self.bac_layer.add_module("temporal_conv_"+str(i),
                                  nn.SeparableConv2d(input_dim, output_dim, kernel_size, padding=padding,
                                                     normalization_dw="bn", normalization_pw="bn"))
            self.bac_layer.add_module("ELU",nn.ELU())

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)
        torch.nn.init.xavier_uniform_(self.spatial_layer.weight)
        #self.bac_layer.apply(weight_init_xavier_uniform)

    def forward(self, x):
        x = x.unsqueeze(dim=1) #[batch, 1, electrode, length]
        x = self.elu(self.spectral_layer(x)) #[batch, 4, electrode, length']
        x = self.elu(self.spatial_layer(x))
        x = self.bac_layer(x)
        out = self.maxpool(x).squeeze()

        return out

class SBP_raw_Head_NN(nn.Module):
    def __init__(self, classes):
        super(SBP_raw_Head_NN, self).__init__()
        self.classes = classes
        self.layer = nn.Sequential(
            torch.nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, self.classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.layer.apply(weight_init_xavier_uniform)

    def forward(self, x):
        x = self.layer(x) #[batch, 4, electrode, length']
        out = self.softmax(x)

        return out

class StoppedBandPathway_raw(nn.Module):
    def __init__(self, fs, electrode, Unsupervise, encode_info):
        super(StoppedBandPathway_raw, self).__init__()
        self.encoder = SBP_raw_Encoder(fs, electrode, encode_info)
        self.pretrain = SBP_raw_Head_NN(5)
        self.Unsupervise = Unsupervise
        self.bn = nn.BatchNorm1d(electrode)

    def forward(self, x):
        x = self.bn(x)
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x