import numpy as np
import torch
import torch.nn as nn
import separableconv.nn as nn

'''Appendix'''
def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, nn.Conv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)

    elif isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)

    elif isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()
    '''elif isinstance(submodule, nn.SeparableConv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)'''


def ExchangeFnC(x):
    batch, channel, feature = x.shape
    x = x.view(batch, feature, channel)
    return x

'''
StoppedBandPathway classes
 - raw Pathway
 - 1C Pathway
 - 1Dconv Pathway
 - 1Dconv noGP Pathway
'''
###raw
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

###1C
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

###1Dconv
class SBP_1Dconv_Encoder(nn.Module):
    def __init__(self, fs, encode_info):
        super(SBP_1Dconv_Encoder, self).__init__()
        #spectral layer means spectral convolution
        #self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        #convolution layer are initiated by xavier_uniform initization
        #Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        #[batch, electrode, length] -> [batch, electrode, Feature]
        self.fs = fs
        self.elu = nn.ELU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(1)

        self.spectral_layer = nn.Conv1d(1, 128, int(self.fs/2), padding="same")

        self.bac_layer = nn.Sequential()
        for i, arg in enumerate(encode_info):
            input_dim, output_dim, kernel_size, padding = arg
            self.bac_layer.add_module("temporal_conv_"+str(i),
                                  nn.SeparableConv1d(input_dim, output_dim, kernel_size, padding=padding,
                                                     normalization_dw="bn", normalization_pw="bn"))
            self.bac_layer.add_module("ELU",nn.ELU())
            #self.bac_layer.add_module("maxpool", nn.MaxPool1d(3,stride=2))

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)

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

    def getDomainfeature(self, x):
        b, c, l = x.shape
        x = x.view(-1, l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)

        x = self.elu(self.spectral_layer(x))

        return x

####cutdown
class SBP_1Dconv_Cut_Encoder(nn.Module):
    def __init__(self, fs, encode_info):
        super(SBP_1Dconv_Cut_Encoder, self).__init__()
        #spectral layer means spectral convolution
        #self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        #convolution layer are initiated by xavier_uniform initization
        #Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        #[batch, electrode, length] -> [batch, electrode, Feature]
        self.fs = fs
        self.elu = nn.ELU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(1)

        self.spectral_layer = nn.Conv1d(1, 256, int(self.fs/2), padding="same")

        self.bac_layer = nn.Sequential()
        for i, arg in enumerate(encode_info):
            input_dim, output_dim, kernel_size, padding = arg
            self.bac_layer.add_module("temporal_conv_"+str(i),
                                  nn.SeparableConv1d(input_dim, output_dim, kernel_size, padding=padding,
                                                     normalization_dw="bn", normalization_pw="bn",stride=2**i))
            self.bac_layer.add_module("ELU",nn.ELU())
            #self.bac_layer.add_module("maxpool", nn.MaxPool1d(3,stride=2))

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)

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

    def getDomainfeature(self, x):
        b, c, l = x.shape
        x = x.view(-1, l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)

        x = self.elu(self.spectral_layer(x))

        return x

class SBP_spaticonv_Encoder(nn.Module):
    def __init__(self, fs, encode_info):
        super(SBP_spaticonv_Encoder, self).__init__()
        #spectral layer means spectral convolution
        #self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        #convolution layer are initiated by xavier_uniform initization
        #Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        #[batch, electrode, length] -> [batch, electrode, Feature]
        self.fs = fs
        self.elu = nn.ELU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(1)

        self.spectral_layer = nn.Conv1d(1, 256, int(self.fs/2), padding="same")

        self.bac_layer = nn.Sequential()
        for i, arg in enumerate(encode_info):
            input_dim, output_dim, kernel_size, padding = arg
            self.bac_layer.add_module("temporal_conv_"+str(i),
                                  nn.SeparableConv1d(input_dim, output_dim, kernel_size, padding=padding,
                                                     normalization_dw="bn", normalization_pw="bn",stride=2**i))
            self.bac_layer.add_module("ELU",nn.ELU())
            self.bac_layer.add_module("maxpool", nn.MaxPool1d(3,stride=2))

        self.spatial_layer = nn.Conv2d(output_dim, output_dim, (1,1))

        torch.nn.init.xavier_uniform_(self.spectral_layer.weight)

    def forward(self, x):
        b,c,l = x.shape
        x = x.view(-1,l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)

        x = self.elu(self.spectral_layer(x))
        x = self.bac_layer(x)
        x = self.maxpool(x).squeeze()

        _, feature = x.shape
        x = torch.matmul(x.view(b,feature,c,1), x.view(b,feature,1,c))
        x = self.spatial_layer(x)

        #x = x.view(b,c,-1)

        return x


#Linear layer for SSL classification
class SBP_spaticonv_Head_NN(nn.Module):
    def __init__(self, classes, feature):
        super(SBP_spaticonv_Head_NN, self).__init__()
        self.classes = classes
        self.feature = feature
        self.layer = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(self.feature, self.classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.layer.apply(weight_init_xavier_uniform)
        self.bn = nn.BatchNorm1d(feature)

    def forward(self, x):
        #x= self.bn(x)
        x = self.maxpool(x).squeeze()
        x = self.layer(x)
        x = self.softmax(x)

        return x

class StoppedBandPathway_spaticonv(nn.Module):
    def __init__(self, fs, encode_info, Unsupervise=None, bands=None, dense = 128, dom_len = 4):
        super(StoppedBandPathway_spaticonv, self).__init__()
        if bands == None:
            bands = [0,1,2,3,4]

        self.encoder = SBP_spaticonv_Encoder(fs, encode_info)
        self.pretrain = SBP_spaticonv_Head_NN(len(bands), dense)
        self.domain_pred = SBP_1Dconv_Head_NN(dom_len, dense)
        self.Unsupervise = Unsupervise

    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x

    def getDomainfeature(self, x):
        x = self.encoder(x)
        x = self.domain_pred(x)
        return x


#Linear layer for SSL classification
class SBP_1Dconv_Head_NN(nn.Module):
    def __init__(self, classes, feature):
        super(SBP_1Dconv_Head_NN, self).__init__()
        self.classes = classes
        self.feature = feature
        self.layer = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(self.feature, self.classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.layer.apply(weight_init_xavier_uniform)
        self.bn = nn.BatchNorm1d(feature)

    def forward(self, x):
        b,c,l = x.shape
        x = x.view(-1,l)

        #x= self.bn(x)
        x = self.layer(x)
        x = self.softmax(x)

        return x

class StoppedBandPathway_1Dconv(nn.Module):
    def __init__(self, fs, encode_info, Unsupervise=None, bands=None, dense = 128, dom_len = 4):
        super(StoppedBandPathway_1Dconv, self).__init__()
        if bands == None:
            bands = [0,1,2,3,4]

        self.encoder = SBP_1Dconv_Cut_Encoder(fs, encode_info)
        self.pretrain = SBP_1Dconv_Head_NN(len(bands), dense)
        self.domain_pred = SBP_1Dconv_Head_NN(dom_len, dense)
        self.Unsupervise = Unsupervise

    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x

    def getDomainfeature(self, x):
        x = self.encoder(x)
        x = self.domain_pred(x)
        return x

####1Dconv_no Global pooling
class SBP_1DnGP_Encoder(nn.Module):
    def __init__(self, fs, encode_info):
        super(SBP_1DnGP_Encoder, self).__init__()
        #spectral layer means spectral convolution
        #self.bac_layer is consist of several SeparableConv2d, which plays the role of temporal separable convolution
        #convolution layer are initiated by xavier_uniform initization
        #Input are Normalized by self.bn(=torch.nn.BatchNorm2d)
        #[batch, electrode, length] -> [batch, electrode, Feature]
        self.fs = fs
        self.elu = nn.ELU()
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

        x = x.view(b,c,-1,l)

        return x

#Linear layer for SSL classification
class SBP_1DnGP_Head_NN(nn.Module):
    def __init__(self, classes, feature):
        super(SBP_1DnGP_Head_NN, self).__init__()
        self.classes = classes
        self.feature = feature
        self.layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature, self.classes)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.layer.apply(weight_init_xavier_uniform)
        self.bn = nn.BatchNorm1d(feature)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        b,c,f,l = x.shape
        x = x.view(-1,f,l)

        x = self.maxpool(x).squeeze()
        x= self.bn(x)
        x = self.layer(x)
        x = self.softmax(x)

        return x

class StoppedBandPathway_1Dconv_noGP(nn.Module):
    def __init__(self, fs, encode_info, Unsupervise=None, bands=None, dense = 128):
        super(StoppedBandPathway_1Dconv_noGP, self).__init__()
        if bands == None:
            bands = [0,1,2,3,4]

        self.encoder = SBP_1DnGP_Encoder(fs, encode_info)
        self.pretrain = SBP_1DnGP_Head_NN(len(bands), dense)
        self.Unsupervise = Unsupervise


    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x

'''
Spatial Pathway

Using : 
if __name__ == "__main__":

    model = SpatialNetwork(7680,14,14,[10,12,15],[30,15,5])
    #model = SpatialNetwork6(7680,14,14,[8,10,12,20])
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    summary_(model, (23,200), batch_size=322)
'''
class SRP_GP_Encoder(nn.Module):
    def __init__(self, FL):
        super(SRP_GP_Encoder, self).__init__()
        self.FL = FL
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.sep1 = nn.Sequential(
            nn.Conv1d(1, FL[0], kernel_size=30, padding="same", groups=1),
            nn.Conv1d(FL[0], FL[0], kernel_size=1)
        )
        self.sep2 = nn.Sequential(
            nn.Conv1d(FL[0], FL[0] * FL[1], kernel_size=15, padding="same", groups=FL[0]),
            nn.Conv1d(FL[0] * FL[1], FL[1], kernel_size=1)
        )
        self.sep3 = nn.Sequential(
            nn.Conv1d(FL[1], FL[1] * FL[2], kernel_size=5, padding="same", groups=FL[1]),
            nn.Conv1d(FL[1] * FL[2], FL[2], kernel_size=1)
        )

    #   X =(B,C,X,Y)  X =(B,14,T)
    def forward(self, x):
        # X=(B,C,T)
        b,c,l = x.shape
        x = x.view(-1,l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)

        # X=(B*C,1,T)
        out = self.sep1(x)
        out = self.elu(out)

        # X=(B*C,F,T2)
        out = self.sep2(out)
        out = self.elu(out)

        # X=(B*C,F,T3)
        out = self.sep3(out)
        out = self.elu(out)
        # print(out.shape)
        # norm = torch.norm(out)
        ##READ OUT 만들기 , fc 길이 설정

        # Read out X = (B,1,53)
        out= out.view(b,c,self.FL[-1],-1)
        # max_vec, _ = out.max(dim=1)
        out = torch.cat([out.sum(dim=1), out.max(dim=1)[0], out.sum(dim=1) / c], dim=-2)

        return self.maxpool(out).squeeze()

class SRP_GP_Head_NN(nn.Module):
    def __init__(self, classes, feature):
        super(SRP_GP_Head_NN, self).__init__()
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
        x = self.bn(x)
        x = self.layer(x)
        x = self.softmax(x)

        return x

class SpikeRecognitionPathway_GP(nn.Module):
    def __init__(self, FL, Unsupervise = None, dense = 64, bands = None):
        super(SpikeRecognitionPathway_GP, self).__init__()
        if bands == None:
            bands = [0,1,2,3]

        self.encoder = SRP_GP_Encoder(FL)
        self.pretrain = SRP_GP_Head_NN(len(bands), dense)
        self.Unsupervise = Unsupervise

    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x

####SpikeNet ver2
class SRP_GP_extractor(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels, FL, KL):
        super(SRP_GP_extractor,self).__init__()
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


class SRP_GP_Endcoder1(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels, FL, KL):
        super(SRP_GP_Endcoder1, self).__init__()

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
        out=self.bn1(x)
        '''out = self.conv0(out)
        #out=self.bn1(out)
        out = self.elu(out)'''
        # X=(B*C,1,T)
        out = self.conv1(out)
        out = self.elu(out)

        # X=(B*C,F,T2)
        out = self.conv2(out)
        out = self.elu(out)

        # X=(B*C,F,T3)
        out = self.conv3(out)
        out = self.elu(out)

        out = torch.cat([out.sum(dim=1), out.max(dim=1)[0], out.sum(dim=1) / self.in_channels], dim=1)

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

        self.sep1 = nn.SeparableConv1d(1, FL[0], KS[0], padding="same", normalization_dw="bn", normalization_pw="bn")
        self.sep2 = nn.SeparableConv1d(FL[0], FL[1], KS[1], padding="same", normalization_dw="bn", normalization_pw="bn")
        self.sep3 = nn.SeparableConv1d(FL[1], FL[2], KS[2], padding="same", normalization_dw="bn", normalization_pw="bn")

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

class SpikeRecognitionPathway_GP_ver2(nn.Module):
    def __init__(self, temporal_len, in_channels, out_channels, FL, KL):
        super(SpikeRecognitionPathway_GP_ver2, self).__init__()
        self.encoder = SRP_GP_Endcoder1(temporal_len, in_channels, out_channels, FL, KL)
        self.classifier = SRP_GP_extractor(temporal_len, in_channels, out_channels,FL,KL)

    def forward(self,x):
        feature = self.encoder(x)
        out= self.classifier(feature)
        return out

    def get_Rep(self,x):
        x = self.encoder(x)
        return x
'''
Temporal Pathway
 - no Global Pooling
 - Global Pooling
'''
class TemporalPathway_noGP(nn.Module):
    def __init__(self):
        super(TemporalPathway_noGP, self).__init__()

        self.channels = 1  # we use only single channel

        # Activation functions
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1)

        # self.conv2t = nns.SeparableConv1d(16,32,10,padding ='same') (in_channels, out_channels, kernel size,,,)

        self.softmax = nn.Softmax()
        self.conv1t = nn.Conv1d(1, 10, 30, padding='same')  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv1d(10, 10, self.channels)
        self.conv2t = nn.Conv1d(10, 20, 15, padding='same')
        self.conv2s = nn.Conv1d(20, 20, self.channels)
        self.conv3t = nn.Conv1d(20, 34, 5, padding='same')
        self.conv3s = nn.Conv1d(34, 34, self.channels)

        # Flatteninig
        self.flatten = nn.Flatten()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Decision making
        self.Linear = nn.Linear(12800, 4)  #

    def embedding(self, x):
        x = self.bn(x)
        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = torch.cat([f1, f2, f3], dim=1)

        return feature

    def classifier(self, feature):
        # Flattening, dropout, mapping into the decision nodes
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.Linear(feature))
        return y_hat

    def forward(self, x):
        feature = self.embedding(x)
        y_hat = self.classifier(feature)
        return y_hat

    def getRep(self, x):
        feature = self.embedding(x)
        return feature

###GP Encoder
class TPP_GP_Encoder(nn.Module):
    def __init__(self, Fs):
        super(TPP_GP_Encoder, self).__init__()

        self.channels = 1  # we use only single channel

        # Activation functions
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # self.conv2t = nns.SeparableConv1d(16,32,10,padding ='same') (in_channels, out_channels, kernel size,,,)

        self.conv1t = nn.Conv1d(1, Fs[0], 30, padding='same')  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv1d(Fs[0], Fs[0], self.channels)
        self.conv2t = nn.Conv1d(Fs[0], Fs[1], 15, padding='same')
        self.conv2s = nn.Conv1d(Fs[1], Fs[1], self.channels)
        self.conv3t = nn.Conv1d(Fs[1], Fs[2], 5, padding='same')
        self.conv3s = nn.Conv1d(Fs[2], Fs[2], self.channels)

    def forward(self, x):
        b,c,l = x.shape
        x = x.view(-1,l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)

        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = torch.cat([f1, f2, f3], dim=1)
        feature = self.maxpool(feature).squeeze()

        feature =feature.view(b,c,-1)

        return feature


###GP Encoder stride
class TPP_GP_Cut_Encoder(nn.Module):
    def __init__(self, Fs):
        super(TPP_GP_Cut_Encoder, self).__init__()

        self.channels = 1  # we use only single channel

        # Activation functions
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # self.conv2t = nns.SeparableConv1d(16,32,10,padding ='same') (in_channels, out_channels, kernel size,,,)

        self.conv1t = nn.Conv1d(1, Fs[0], 30, )  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv1d(Fs[0], Fs[0], self.channels)
        self.conv2t = nn.Conv1d(Fs[0], Fs[1], 15, stride=2)
        self.conv2s = nn.Conv1d(Fs[1], Fs[1], self.channels)
        self.conv3t = nn.Conv1d(Fs[1], Fs[2], 5, padding=4)
        self.conv3s = nn.Conv1d(Fs[2], Fs[2], self.channels)

    def forward(self, x):
        b,c,l = x.shape
        x = x.view(-1,l)

        x = x.unsqueeze(dim=1)
        x = self.bn(x)

        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = torch.cat([f1, f2, f3], dim=1)
        feature = self.maxpool(feature).squeeze()

        feature =feature.view(b,c,-1)

        return feature

class TPP_GP_Head_NN(nn.Module):
    def __init__(self, classes, feature):
        super(TPP_GP_Head_NN, self).__init__()
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

        x = self.bn(x)
        x = self.layer(x)
        x = self.softmax(x)

        return x

class TrendPredictPathway_GP(nn.Module):
    def __init__(self, fs, Unsupervise=None, bands=None, dense = 64):
        super(TrendPredictPathway_GP, self).__init__()
        if bands == None:
            bands = [0,1,2,3]

        self.encoder = TPP_GP_Cut_Encoder(fs)
        self.pretrain = TPP_GP_Head_NN(len(bands), dense)
        self.Unsupervise = Unsupervise

    def forward(self, x):
        x = self.encoder(x)
        x = self.pretrain(x)
        return x

    def getRep(self, x):
        x = self.encoder(x)
        return x
'''
Feature Encoder & StatisticianModule
 - Using Feature
 - only 1 Pathway
  1) spectral
  2) spatial
  3) temporal
 - have 2 Pathway
  1) spectral & spatial
  2) spatial & temporal
  3) temporal & spectral
 - have all Pathway
'''

###have all Pathway
class FeatureEncoder(nn.Module):
    def __init__(self, spectral_path, spatial_path, temporal_path):
        super(FeatureEncoder, self).__init__()
        self.spectral_path = spectral_path
        self.spatial_path = spatial_path
        self.temporal_path = temporal_path

        self.spec_GAP = nn.AdaptiveAvgPool1d(1)
        self.temp_GAP = nn.AdaptiveAvgPool1d(1)
        #self.GVP = torch.var(dim=-1)

    def forward(self, x):
        f_spec = ExchangeFnC(self.spectral_path.getRep(x))
        f_spat = self.spatial_path.module.get_Rep(x)
        f_temp = ExchangeFnC(self.temporal_path.getRep(x))

        f_GAP = torch.cat((self.spec_GAP(f_spec).squeeze(), f_spat, self.temp_GAP(f_temp).squeeze()), axis=1)
        f_GVP = torch.cat((torch.var(f_spec, dim=-1), f_spat, torch.var(f_temp, dim=-1)), axis=1)

        x = [f_GAP, f_GVP]

        return x

class StatisticianModule(nn.Module):
    def __init__(self, dense, classes, param):
        super(StatisticianModule, self).__init__()
        self.classes = classes

        self.softmax = torch.nn.Softmax(dim=1)
        self.c_dense = nn.Linear(int(param[0]+param[1]+param[2])*2, dense)

        self.gap_pwconv = nn.Conv1d(int(param[0]+param[1]+param[2]), dense, 1)
        self.gvp_pwconv = nn.Conv1d(int(param[0]+param[1]+param[2]), dense, 1)

        self.fullconnect = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dense, self.classes)
        )


    def forward(self, f_GAP, f_GVP):
        #[batch, gap+gvp] -> [batch,dense]
        c = self.softmax(self.c_dense(torch.cat((f_GAP, f_GVP),axis=1)))

        #[batch, gap, 1] -> [batch, 1, dense] -> [batch, dense]
        f_GAP_d = self.gap_pwconv(f_GAP.unsqueeze(dim=-1)).squeeze()
        f_GVP_d = self.gvp_pwconv(f_GVP.unsqueeze(dim=-1)).squeeze()

        f_GAP_dd = torch.sum(c*f_GAP_d,dim=1)
        f_GVP_dd = torch.sum(c*f_GVP_d,dim=1)

        ALN = torch.div(torch.sub(f_GAP_d.T,f_GAP_dd),f_GAP_dd).T

        y_hat = self.softmax(self.fullconnect(ALN))

        return y_hat


###have Spec&Temp
class FeatureEncoder_noSpatial(nn.Module):
    def __init__(self, spectral_path, temporal_path):
        super(FeatureEncoder_noSpatial, self).__init__()
        self.spectral_path = spectral_path
        self.temporal_path = temporal_path

        self.spec_GAP = nn.AdaptiveAvgPool1d(1)
        self.spec_spatial = nn.Conv1d(128,128,23)
        self.temp_GAP = nn.AdaptiveAvgPool1d(1)
        self.temp_spatial = nn.Conv1d(128,128,23)
        #self.GVP = torch.var(dim=-1)

    def forward(self, x):
        f_spec = self.spec_spatial(ExchangeFnC(self.spectral_path.getRep(x)))
        f_temp = self.temp_spatial(ExchangeFnC(self.temporal_path.getRep(x)))

        #f_GAP = torch.cat((self.spec_GAP(f_spec).squeeze(), self.temp_GAP(f_temp).squeeze()), axis=1)
        #f_GVP = torch.cat((torch.var(f_spec, dim=-1), torch.var(f_temp, dim=-1)), axis=1)

        #f_GAP = torch.cat((f_spec.squeeze(), f_temp.squeeze()), axis=1)
        #f_GVP = torch.cat((torch.var(f_spec, dim=-1), torch.var(f_temp, dim=-1)), axis=1)

        x = [f_spec.squeeze(), f_temp.squeeze()]

        return x

class StatisticianModule_noSpatial(nn.Module):
    def __init__(self, dense, classes, param):
        super(StatisticianModule_noSpatial, self).__init__()
        self.classes = classes

        self.softmax = torch.nn.Softmax(dim=1)
        self.c_dense = nn.Linear(int(param[0]+param[1]), dense)

        self.gap_pwconv = nn.Conv1d(int(param[0]), dense, 1)
        self.gvp_pwconv = nn.Conv1d(int(param[1]), dense, 1)

        self.fullconnect = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dense, self.classes)
        )


    def forward(self, f_GAP, f_GVP):
        #[batch, gap+gvp] -> [batch,dense]
        c = self.softmax(self.c_dense(torch.cat((f_GAP, f_GVP),axis=1)))

        #[batch, gap, 1] -> [batch, 1, dense] -> [batch, dense]
        f_GAP_d = self.gap_pwconv(f_GAP.unsqueeze(dim=-1)).squeeze()
        f_GVP_d = self.gvp_pwconv(f_GVP.unsqueeze(dim=-1)).squeeze()

        f_GAP_dd = torch.sum(c*f_GAP_d,dim=1)
        f_GVP_dd = torch.sum(c*f_GVP_d,dim=1)

        ALN = torch.div(torch.sub(f_GAP_d.T,f_GAP_dd),f_GAP_dd).T

        y_hat = self.softmax(self.fullconnect(ALN))

        return y_hat

'''
TIPNet
'''

class TIPNet(nn.Module):
    def __init__(self, FeatureEncoder, StatisticianModule):
        super(TIPNet, self).__init__()
        self.FeatureEncoder = FeatureEncoder
        '''
        for param in self.FeatureEncoder.parameters():
            param.requires_grad = False
        '''
        self.StatisticianModule = StatisticianModule

    def forward(self, x):
        f_GAP, f_GVP = self.FeatureEncoder(x)
        x = self.StatisticianModule(f_GAP, f_GVP)
        return x


class DIPNet(nn.Module):
    def __init__(self, FeatureEncoder, StatisticianModule):
        super(DIPNet, self).__init__()
        self.FeatureEncoder = FeatureEncoder
        '''
        for param in self.FeatureEncoder.parameters():
            param.requires_grad = False
        '''
        self.StatisticianModule = StatisticianModule

    def forward(self, x):
        f_GAP, f_GVP = self.FeatureEncoder(x)
        x = self.StatisticianModule(f_GAP, f_GVP)
        return x