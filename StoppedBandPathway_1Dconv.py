import numpy as np
import torch
import torch.nn as nn
import separableconv.nn as nn

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, nn.SeparableConv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)

    if isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)

class Encoder(nn.Module):
    def __init__(self, fs, electrode, encode_info):
        super(Encoder, self).__init__()
        #spectral layer means spectral convolution
        #self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        #convolution layer are initiated by xavier_uniform initization
        #Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        #[batch, electrode, length] -> [batch, electrode, Feature]
        self.fs = fs
        self.elu = nn.ELU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(1)

        self.spectral_layer = nn.Conv1d(1, 8, int(self.fs/2), padding="same")

        self.bac_layer = nn.Sequential()
        for i, arg in enumerate(encode_info):
            input_dim, output_dim, kernel_size, padding = arg
            self.bac_layer.add_module("temporal_conv_"+str(i),
                                  nn.SeparableConv1d(input_dim, output_dim, kernel_size, padding=padding,
                                                     normalization_dw="bn", normalization_pw="bn"))
            self.bac_layer.add_module("ELU",nn.ELU())

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)
        #self.bac_layer.apply(weight_init_xavier_uniform)

    def forward(self, x):
        b,c,l = x.shape
        x = x.view(-1,l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)
        x = self.elu(self.spectral_layer(x))
        x = self.bac_layer(x)
        x = self.maxpool(x).squeeze()

        x = x.view(b,c,-1)

        return x

#Linear layer for SSL classification
class Head_NN(nn.Module):
    def __init__(self, classes, feature):
        super(Head_NN, self).__init__()
        self.classes = classes
        self.feature = feature
        self.layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature, self.classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.layer.apply(weight_init_xavier_uniform)
        self.bn = nn.BatchNorm1d(feature)

    def forward(self, x):
        b,c,l = x.shape
        x = x.view(-1,l)

        x= self.bn(x)
        x = self.layer(x)
        x = self.softmax(x)

        return x

class StoppedBandPathway_1Dconv(nn.Module):
    def __init__(self, fs, electrode, Unsupervise, encode_info, bands):
        super(StoppedBandPathway_1Dconv, self).__init__()
        self.encoder = Encoder(fs, electrode, encode_info)
        self.pretrain = Head_NN(len(bands), 128)
        self.Unsupervise = Unsupervise


    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x