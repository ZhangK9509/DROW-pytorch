# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2018-12-10 Mon 21:53:33
# Filename     : main.py
# Email        : painter9509@126.com


import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

import models
from data.dataset import DROWDataset
from data.preprocess import preprocess as ppro
from utils import visualize as v


def train(model, device, loader, optimizer, criteria, epoch, print_freq):
    model.train()
    for batch_index, (inps, truth) in enumerate(loader):
        inps = inps.to(device)  # (64, 48)
        cls_truth = truth[:, 0].long().to(device)  # (64)
        offs_truth = truth[:, 1:].float().to(device)  # (64, 2)

        optimizer.zero_grad()
        outp_cls, outp_offs = model(inps)  # (64, 3) (64, 2)
        cls_loss = criteria[0](outp_cls, cls_truth)
        offs_loss = criteria[1](outp_offs, offs_truth)
        loss = cls_loss + offs_loss
        loss.backward()
        optimizer.step()
        if batch_index % print_freq == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_index * len(inps), len(loader.dataset),
                100. * batch_index / len(loader), loss.item()))


def test(model, device, inp, criteria):
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(inp)
        outp_cls, outp_offs = model(inp)
    return outp_cls.numpy(), outp_offs.numpy()


def main():
    parser = argparse.ArgumentParser(description="DROW")
    parser.add_argument(
            "mode", type=str, choices=["TRAIN", "TEST"],
            help='Choose to train ("TRAIN") or test ("TEST") model.')
    parser.add_argument(
            "--train_data_dir", type=str, default="./DROW-data/train/",
            help="Input the directory of training data \
                    (default: ./Drow-data/train/).")
    parser.add_argument(
            "--test_data_dir", type=str, default="./DROW-data/test/",
            help="Input the directory of testing data \
                    (default: ./DROW-data/test/).")
    parser.add_argument(
            "--model", type=str, choices=["DROWNet"], default="DROWNet",
            help="Choose a model to use (default: DROWNet).")
    parser.add_argument(
            "--win_res", type=int, default=48, metavar='W',
            help="Set the size of window (default: 48).")
    parser.add_argument(
            "--batch_size", type=int, default=64, metavar='B',
            help="Set the batch size for training (default: 64).")
    parser.add_argument(
            "--num_epochs", type=int, default=10, metavar='N',
            help="Set the number of epochs of training (default: 10).")
    parser.add_argument(
            "--lr", type=int, default=0.01, metavar='LR',
            help="Set learning rate (default: 0.01).")
    parser.add_argument(
            "--use_cuda", action="store_true", default=False,
            help="Disable CUDA trainging or not (default: False).")
    parser.add_argument(
            "--save_model", action="store_true", default=False,
            help="Save the model or not (default: False).")
    parser.add_argument(
            "--print_freq", type=int, default=10,  metavar="Freq",
            help="Set the frequence about showing training status.")
    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    torch.manual_seed(1)

    model = getattr(models, args.model)().to(device)
    criteria = [nn.NLLLoss(), nn.MSELoss()]

    if args.mode == "TRAIN":
        inps, truth = ppro("TRAIN", args.win_res, data_dir=args.train_data_dir)
        inps = inps.reshape((-1, args.win_res))  # Process a dot, not a scan.
        truth = truth.reshape((-1, 3))

        drow_dataset = DROWDataset(inps, truth)
        loader = DataLoader(
                dataset=drow_dataset, batch_size=args.batch_size,
                shuffle=True, **kwargs)

        optimizer = optim.Adadelta(
                model.parameters(), lr=args.lr, rho=0.95, eps=1e-7)

        for epoch in range(1, args.num_epochs + 1):
            train(
                    model, device, loader, optimizer, criteria, epoch,
                    args.print_freq)

        if args.save_model:
            model.save()

    elif args.mode == "TEST":
        fnames = [f[:-4] for f in glob(
            os.path.join(args.test_data_dir, "*.csv"))]
        model.load()
        for f in fnames:
            new_dir = os.path.join("./results", f.split('/')[-1])
            if os.path.exists(new_dir):
                raise Exception("Directory Conflict: {}.".format(new_dir))
            os.makedirs(new_dir)
            inps, seqs, scans, wcs, was = ppro(
                    "TEST", args.win_res, fnames=[f])
            for i in range(len(inps)):
                print(i)
                inp = inps[i].reshape((-1, args.win_res))
                outp_cls, outp_offs = test(model, device, inp, criteria)
                ax = v.show_scan(scans[i], figsize=(10, 10))
                v.show_det(wcs[i], ax, radius=0.6)
                v.show_det(was[i], ax, radius=0.4, color="#FF99CC")
                #  v.show_votes(scan, outp_cls, outp_offs, ax=ax)
                v.show_cls(scans[i], outp_cls, ax=ax)
                plt.savefig(os.path.join(new_dir, "{}.png".format(seqs[i])))
                plt.clf()
                if (i + 1) % 10 == 0:
                    plt.close("all")


if __name__ == "__main__":
    main()
