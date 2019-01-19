# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2018-12-22 Sat 20:15:42
# Filename     : loader.py
# Email        : painter9509@126.com


import torch
from torch.utils.data import Dataset


class DROWDataset(Dataset):
    def __init__(self, inps, truth):
        self.inps = torch.tensor(inps)
        self.truth = torch.tensor(truth)

    def __getitem__(self, index):
        return self.inps[index], self.inps[index]

    def __len__(self):
        return self.inps.shape[0]
