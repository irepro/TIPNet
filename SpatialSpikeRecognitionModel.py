import torch
from torch import nn
import separableconv.nn as snn
import os


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

def model_loader(model_name:str,temporal_len, in_channels, out_channels,FL,KL):
    if model_name == "SpatialNetwork":
        return SpatialNetwork(temporal_len, in_channels, out_channels,FL)
    if model_name == "SpatialNetwork3":
        return SpatialNetwork3(temporal_len,in_channels, out_channels,FL,KL)
    if model_name == "SpatialNetwork4":
        return SpatialNetwork4(temporal_len, in_channels, out_channels, FL, KL)
    if model_name == "SpatialNetwork5":
        return SpatialNetwork5(temporal_len,in_channels, out_channels, FL)
    if model_name == "SpatialNetwork6":
        return SpatialNetwork6(temporal_len,in_channels, out_channels,FL,KL)
    if model_name == "SpatialNetwork7":
        return SpatialNetwork7(temporal_len, in_channels, out_channels, FL, KL)


class SpatialNetwork4(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels,FL,KS):
        super(SpatialNetwork4, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.FL= FL

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GAP2 = nn.AdaptiveAvgPool1d(20)
        self.dropout=nn.Dropout(0.3)

        self.sep1 = nn.Sequential(
            nn.Conv2d(1,FL[0], kernel_size=(1,KS[0]), padding=(0,1), groups=1),
            nn.Conv2d(FL[0], FL[0], kernel_size=1)
        )
        self.sep2 = nn.Sequential(
            nn.Conv2d(FL[0], FL[0] * FL[1], kernel_size=(1,KS[1]), padding=(0,1), groups=FL[0]),
            nn.Conv2d(FL[0]*FL[1], FL[1], kernel_size=1)
        )
        self.sep3 = nn.Sequential(
            nn.Conv2d(FL[1], FL[1] * FL[2], kernel_size=(1,KS[2]), padding=(0,1), groups=FL[1]),
            nn.Conv2d(FL[1]*FL[2], FL[2], kernel_size=1)
        )

        self.bn1 = nn.BatchNorm2d(FL[0])
        self.bn2 = nn.BatchNorm2d(FL[1])

        self.sep1.apply(weight_init_xavier_uniform)
        self.sep2.apply(weight_init_xavier_uniform)
        self.sep3.apply(weight_init_xavier_uniform)

        self.fc = nn.Linear(20 * 3, 4)
        self.elu = nn.ELU()
        self.classifier = nn.Softmax(dim=1)
        self.maxpool=nn.MaxPool2d((1,2), stride=(1,2))
        # xavier initialization

    #   X =(B,C,X,Y)  X =(B,14,T)
    def forward(self, x):


        # X=(B,1,C,T)
        x=x.unsqueeze(1)
        out = self.sep1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.maxpool(out)

        # X=(B,F,C,T)
        out = self.sep2(out)
        out= self.bn2(out)
        out = self.elu(out)
        out= self.maxpool(out)

        # X=(B,F,C,T)
        out = self.sep3(out)
        out = self.elu(out)

        # X=(B,C,F,T)
        out=torch.transpose(out,1,2)
        # X=(B,C,T,F)
        out=torch.transpose(out,2,3)
        b,c,t,f= out.shape
        # X=(B*C,T,F)
        out=out.reshape([b*c,t,-1])
        # X=(B,C,T,1)
        out=self.GAP(out)
        # X=(B,C,T)        
        out=out.view([b,c,t])

        # X=(B,C,20)
        out=self.GAP2(out)

        # X=(B,1,60)
        out = torch.cat([out.sum(dim=1),out.max(dim=1)[0], out.sum(dim=1) / self.in_channels], dim=1)

        # X=(B,4)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

