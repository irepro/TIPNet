import numpy as np
import torch
import torch.nn as nn
import separableconv.nn as nn

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, nn.SeparableConv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)

    if isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)

class SBP_1C_Encoder(nn.Module):
    def __init__(self, fs, electrode, encode_info):
        super(SBP_1C_Encoder, self).__init__()
        #spectral layer means spectral convolution
        #self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        #convolution layer are initiated by xavier_uniform initization
        #Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        #[batch, electrode, length] -> [batch, electrode, Feature]
        self.fs = fs
        self.elu = nn.ELU()
        self.maxpool = nn.AdaptiveMaxPool2d((None,1))
        self.bn = nn.BatchNorm2d(1)

        self.spectral_layer = nn.Conv2d(1, 8, (1,int(self.fs/2)))

        self.bac_layer = nn.Sequential()
        for i, arg in enumerate(encode_info):
            input_dim, output_dim, kernel_size, padding = arg
            self.bac_layer.add_module("temporal_conv_"+str(i),
                                  nn.SeparableConv2d(input_dim, output_dim, kernel_size, padding=padding,
                                                     normalization_dw="bn", normalization_pw="bn"))
            self.bac_layer.add_module("ELU",nn.ELU())

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)
        #self.bac_layer.apply(weight_init_xavier_uniform)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.bn(x)
        x = self.elu(self.spectral_layer(x))
        x = self.bac_layer(x)
        x = x.permute(0, 2, 1, 3)
        x = self.maxpool(x).squeeze()

        return x

#Linear layer for SSL classification
class SBP_1C_Head_NN(nn.Module):
    def __init__(self, classes):
        super(SBP_1C_Head_NN, self).__init__()
        self.classes = classes
        self.layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, self.classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.layer.apply(weight_init_xavier_uniform)
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.view(-1,64)
        x= self.bn(x)
        x = self.layer(x)
        x = self.softmax(x)

        return x

class StoppedBandPathway_1C(nn.Module):
    def __init__(self, fs, electrode, Unsupervise, encode_info, bands):
        super(StoppedBandPathway_1C, self).__init__()
        self.encoder = SBP_1C_Encoder(fs, electrode, encode_info)
        self.pretrain = SBP_1C_Head_NN(len(bands))
        self.Unsupervise = Unsupervise

    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x