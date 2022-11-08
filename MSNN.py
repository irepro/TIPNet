import torch
import torch.nn as nn
import separableconv.nn as nn


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, nn.SeparableConv1d):
        torch.nn.init.xavier_uniform_(submodule.weight)

    if isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)


class feature_extractor3(nn.Module):
    def __init__(self, fs):
        super(feature_extractor3, self).__init__()

        self.channels = 1  # we use only single channel
        self.fs = fs

        # Activation functions
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(1)

        # self.conv2t = nns.SeparableConv1d(16,32,10,padding ='same') (in_channels, out_channels, kernel size,,,)

        self.softmax = nn.Softmax()
        self.spectral_layer = nn.Conv2d(1, 4, (1,int(self.fs/2)))

        self.conv1t = nn.SeparableConv2d(4, 8, (1,30), padding='same')  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv2d(8, 8, self.channels)
        self.conv2t = nn.SeparableConv2d(8, 16, (1,15), padding='same')
        self.conv2s = nn.Conv2d(16, 16, self.channels)
        self.conv3t = nn.SeparableConv2d(16, 32, (1,5), padding='same')
        self.conv3s = nn.Conv2d(32, 32, self.channels)

        # Flatteninig
        self.flatten = nn.Flatten()
        self.maxpool = nn.AdaptiveMaxPool2d((None,1))


        #classify

        self.layer_cls = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(56, 5)
        )
        self.softmax_cls = torch.nn.Softmax(dim=1)
        self.layer_cls.apply(weight_init_xavier_uniform)


        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Decision making

    def embedding(self, x, random_mask=False):
        # print(x.shape)
        x = x.unsqueeze(dim=1)
        x = self.bn(x)
        x = self.spectral_layer(x)

        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = torch.cat([f1, f2, f3], dim=1)

        feature = feature.permute(0, 2, 1, 3)
        feature = self.maxpool(feature).squeeze()

        return feature

    def classifier(self, x):
        # Flattening, dropout, mapping into the decision nodes
        x = x.view(-1, 56)
        x = self.layer_cls(x)
        x = self.softmax_cls(x)

        return x

    def forward(self, x):
        feature = self.embedding(x)
        y_hat = self.classifier(feature)
        return y_hat

    def getRep(self, x):
        feature = self.embedding(x)
        return feature
