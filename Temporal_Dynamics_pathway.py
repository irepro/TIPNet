import torch
import torch.nn as nn
import separableconv.nn as nns

class feature_extractor3(nn.Module):
    def __init__(self):
        super(feature_extractor3, self).__init__()

        self.channels = 1  # we use only single channel


        # Activation functions
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(1)

        # self.conv2t = nns.SeparableConv1d(16,32,10,padding ='same') (in_channels, out_channels, kernel size,,,)

        self.softmax = nn.Softmax()
        self.conv1t = nns.SeparableConv1d(1, 8, 30, padding='same')  # in_channels, out_channels, kernel_size,
        self.conv1s = nn.Conv1d(8, 8, self.channels)
        self.conv2t = nns.SeparableConv1d(8, 16, 15, padding='same')
        self.conv2s = nn.Conv1d(16, 16, self.channels)
        self.conv3t = nns.SeparableConv1d(16, 32, 5, padding='same')
        self.conv3s = nn.Conv1d(32, 32, self.channels)

        # Flatteninig
        self.flatten = nn.Flatten()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Decision making
        self.Linear = nn.Linear(112000, 4)  #


    # MSNN
    def embedding(self, x, random_mask=False):
        # print(x.shape)
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