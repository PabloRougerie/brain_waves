import torch
from torch import nn


#=================
#BASELINE ELEMENTS
#=================


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride= 1, kernel= 3, padding= 1):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels= in_channels,
                      out_channels= out_channels,
                      stride= stride,
                      padding= padding,
                      kernel_size= kernel,
                      bias= False),
            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.block(x)



class BaselineModel(nn.Module):

    def __init__(self, n_classes, n_channels):

        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.block1 = nn.Sequential(
            ConvBlock(in_channels= self.n_channels,
                      out_channels= 32),
            ConvBlock(in_channels= 32,
                      out_channels= 64),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            ConvBlock(in_channels= 64,
                      out_channels= 128),
            ConvBlock(in_channels= 128,
                      out_channels= 256),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            nn.Dropout(0.3)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.Dropout(0.3),
            nn.Linear(64, self.n_classes)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)

        return nn.functional.log_softmax(x, dim= 1)
