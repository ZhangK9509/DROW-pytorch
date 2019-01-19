# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2018-12-23 Sun 14:21:05
# Filename     : preprocess.py
# Email        : painter9509@126.com


import os
from glob import glob

import numpy as np

from utils import utils as u


def preprocess(mode, win_res, fnames=None, data_dir=None):
    if fnames is None:
        fnames = [f[:-4] for f in glob(os.path.join(data_dir, "*.csv"))]
    # Load raw data.
    rawseqs, rawscans = zip(*[u.load_scan(f + ".csv") for f in fnames])
    # Load labelled data.
    detseqs, wcdets, wadets = zip(*map(u.load_dets, fnames))

    print("Loaded {:6.2f}k scans, {:5.2f}k labelled".format(
        sum(map(len, rawseqs)) / 1000, sum(map(len, detseqs)) / 1000))

    # Get annotated seqs with corresponding scans and throw the useless data
    # without annotation.
    seqs, scans, wcs, was = u.linearize(
            rawseqs, rawscans, detseqs, wcdets, wadets)

    if mode == "TRAIN":
        # Filp data horizontally.
        seqs = np.array(seqs + seqs)
        scans = np.array(
                scans + [scan[::-1] for scan in scans], dtype=np.float32)
        wcs = wcs + [[[d[0], -d[1]] for d in dets] for dets in wcs]
        was = was + [[[d[0], -d[1]] for d in dets] for dets in was]
        print("Augmented to {:.2f}k {} scans.".format(
            len(seqs) / 1000, mode.lower()))
    else:
        scans = np.array(scans, dtype=np.float32)

    inps = np.empty((len(scans), 450, win_res), dtype=np.float32)
    for i, scan in enumerate(scans):
        inps[i] = u.generate_cut_outs(scan, npts=win_res)

    if mode == "TEST":
        return inps, seqs, scans, wcs, was

    cls, offs = map(np.array, zip(*list(map(
        u.generate_votes, scans, wcs, was))))

    cls = np.expand_dims(cls, axis=3)
    truth = np.concatenate((cls, offs), axis=2)
    return inps, truth
