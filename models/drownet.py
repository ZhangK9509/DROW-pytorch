# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2018-12-17 Mon 20:17:01
# Filename     : net.py
# Email        : painter9509@126.com


from torch import nn
import torch.nn.functional as F

from .basic import BasicModule


class DROWNet(BasicModule):
    def __init__(self):
        super(DROWNet, self).__init__()
        self.net = nn.Sequential(
                nn.Conv1d(1, 64, 5),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Dropout(0.25),
                nn.Conv1d(64, 64, 5),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.MaxPool1d(2),
                nn.Dropout(0.25),
                nn.Conv1d(64, 128, 5),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(0.25),
                nn.Conv1d(128, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.MaxPool1d(2),
                nn.Dropout(0.25),
                nn.Conv1d(128, 256, 5),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(0.25))
        self.cls_conv = nn.Conv1d(256, 3, 3)
        self.offs_conv = nn.Conv1d(256, 2, 3)

    def classifer(self, x):
        x = self.cls_conv(x)
        x = x.view(-1, 3)
        return F.log_softmax(x, dim=1)

    def offset(self, x):
        x = self.offs_conv(x)
        x = x.view(-1, 2)
        return x

    def forward(self, x):
        x = x.view((-1, 1, x.shape[1]))
        x = self.net(x)
        cls = self.classifer(x)
        offs = self.offset(x)
        return cls, offs
