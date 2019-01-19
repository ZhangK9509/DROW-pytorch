# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2019-01-16 Wed 10:04:28
# Filename     : basic.py
# Email        : painter9509@126.com


import os
import time
from glob import glob

import torch
from torch import nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, name=None):
        if name is None:
            filenames = [f.split('_')[-1] for f in glob(
                os.path.join("./checkpoints", "*.pth"))]
            name = self.model_name + '_' + max(filenames)
        path = os.path.join("./checkpoints", name)
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            name = time.strftime(self.model_name + "_%Y%m%d%H%M%S.pth")
        path = os.path.join("./checkpoints", name)
        torch.save(self.state_dict(), path)
        return name
