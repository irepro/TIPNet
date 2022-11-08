import torch
from torch import nn
from torchsummary import summary as summary_
import separableconv.nn as snn
import os

def model_loader(model_name:str,temporal_len, in_channels, out_channels,FL,KL):
    if model_name == "SpatialNetwork":
        return SpatialNetwork(temporal_len, in_channels, out_channels,FL,KL)
    if model_name == "SpatialNetwork2":
        return SpatialNetwork2(temporal_len, in_channels, out_channels,FL,KL)

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class SSL_extractor(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels, FL, KL):
        super(SSL_extractor,self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.flat=nn.Flatten()
        self.fc = nn.Linear(temporal_len * 3, 4)
        self.classifier = nn.Softmax(dim=1)

    def forward(self,x):
        out= self.flat(x)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


class Endcoder1(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels, FL, KL):
        super(Endcoder1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.FL = FL

        self.dropout=nn.Dropout(0.3)
        self.conv0 = nn.Conv1d(in_channels, in_channels, padding="same", kernel_size=45, groups=in_channels)
        self.conv1 = nn.Conv1d(in_channels, in_channels, padding="same", kernel_size=30, groups=in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, padding="same", kernel_size=15, groups=in_channels)
        self.conv3 = nn.Conv1d(in_channels, in_channels, padding="same", kernel_size=5, groups=in_channels)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(0.5)

        self.sep1 = nn.Sequential(
            nn.Conv1d(1, FL[0], kernel_size=30, padding=1, groups=1),
            nn.Conv1d(FL[0], FL[0], kernel_size=1)
        )
        self.sep2 = nn.Sequential(
            nn.Conv1d(FL[0], FL[0] * FL[1], kernel_size=15, padding=1, groups=FL[0]),
            nn.Conv1d(FL[0] * FL[1], FL[1], kernel_size=1)
        )
        self.sep3 = nn.Sequential(
            nn.Conv1d(FL[1], FL[1] * FL[2], kernel_size=5, padding=1, groups=FL[1]),
            nn.Conv1d(FL[1] * FL[2], FL[2], kernel_size=1)
        )

        # self.fc = nn.Linear(FL[-1] * 3, 4)
        self.fc = nn.Linear(temporal_len * 3, 4)
        self.elu = nn.ELU()
        self.classifier = nn.Softmax(dim=1)
        # xavier initialization
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

    #   X =(B,C,X,Y)  X =(B,14,T)
    def forward(self, x):
        # X=(B,C,T)

        # X=(B*C,1,T)
        #out = self.conv0(x)
        #out=self.bn1(x)
        #out = self.elu(out)
        # X=(B*C,1,T)
        x = self.conv1(x)
        x = self.elu(x)

        # X=(B*C,F,T2)
        x = self.conv2(x)
        x = self.elu(x)

        # X=(B*C,F,T3)
        x = self.conv3(x)
        x = self.elu(x)

        out = torch.cat([x.sum(dim=1), x.max(dim=1)[0], x.sum(dim=1) / self.in_channels], dim=1)

        return out

class SpatialNetwork2(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels,FL,KS):
        super(SpatialNetwork2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.FL= FL

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GAP2 = nn.AdaptiveAvgPool1d(20)
        self.dropout=nn.Dropout(0.3)

        self.sep1 = snn.SeparableConv1d(1, FL[0], KS[0], padding="same", normalization_dw="bn", normalization_pw="bn")
        self.sep2 = snn.SeparableConv1d(FL[0], FL[1], KS[1], padding="same", normalization_dw="bn", normalization_pw="bn")
        self.sep3 = snn.SeparableConv1d(FL[1], FL[2], KS[2], padding="same", normalization_dw="bn", normalization_pw="bn")

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(FL[1])

        self.sep1.apply(weight_init_xavier_uniform)
        self.sep2.apply(weight_init_xavier_uniform)
        self.sep3.apply(weight_init_xavier_uniform)

        self.fc = nn.Linear(FL[-1]*3, 4)
        self.elu = nn.ELU()
        self.classifier = nn.Softmax(dim=1)
        self.maxpool=nn.MaxPool1d(2, stride=2)
        # xavier initialization

    #   X =(B,C,X,Y)  X =(B,14,T)
    def forward(self, x):

        #X=(B,C,T)
        b_s,c_s,t_s=x.shape
        x=x.view([b_s*c_s,1,t_s])
        out = self.bn1(x)

        # X=(B*C,1,T)
        out = self.sep1(out)
        out = self.elu(out)


        # X=(B*C,F,T2)
        out = self.sep2(out)
        #out= self.bn2(out)
        out = self.elu(out)

        # X=(B*C,F,T3)
        out = self.sep3(out)
        out = self.elu(out)


        # B*C ,F,T

        #B*C , F
        out=self.GAP(out)
        out = out.view([b_s, c_s,-1])

        #B,C,F

        #B,3*F
        out = torch.cat([out.sum(dim=1),out.max(dim=1)[0], out.sum(dim=1) / self.in_channels], dim=1)
        # fc layer out -> 10
        # X= (B,1,10)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

class SpatialNetwork(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels, FL, KL):
        super(SpatialNetwork, self).__init__()
        self.encoder = Endcoder1(temporal_len, in_channels, out_channels, FL, KL)
        self.classifier = SSL_extractor(temporal_len, in_channels, out_channels,FL,KL)

    def forward(self,x):
        out = self.encoder(x)
        out= self.classifier(out)
        return out
    def get_Rep(self,x):
        out = self.encoder(x)
        return out


if __name__ == "__main__":

    model = SpatialNetwork(200,23,23,[10,12,15],[30,15,5])
    #model = SpatialNetwork6(7680,14,14,[8,10,12,20])
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    summary_(model, (23,200), batch_size=322)


