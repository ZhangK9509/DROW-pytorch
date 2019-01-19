# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2019-01-08 Tue 15:39:24
# Filename     : utils.py
# Email        : painter9509@126.com


import json

import numpy as np
import cv2
from scipy.spatial.distance import cdist


def load_scan(filename):
    data = np.genfromtxt(filename, delimiter=',')
    seqs, scans = data[:, 0].astype(np.int32), data[:, 1: -1]
    return seqs, scans


def load_dets(name):
    def _load(filename):
        seqs, dets = [], []
        with open(filename) as f:
            for line in f:
                seq, tail = line.split(',', 1)
                seqs.append(int(seq))
                dets.append(json.loads(tail))
        return seqs, dets

    s1, wcs = _load(name + ".wc")
    s2, was = _load(name + ".wa")
    return s1, wcs, was


def linearize(all_seqs, all_scans, all_detseqs, all_wcs, all_was):
    lin_seqs, lin_scans, lin_wcs, lin_was = [], [], [], []
    for seqs, scans, detseqs, wcs, was in zip(
            all_seqs, all_scans, all_detseqs, all_wcs, all_was):
        s2s = dict(zip(seqs, scans))
        for ds, wc, wa in zip(detseqs, wcs, was):
            lin_seqs.append(ds)
            lin_scans.append(s2s[ds])
            lin_wcs.append(wc)
            lin_was.append(wa)
    return lin_seqs, lin_scans, lin_wcs, lin_was


def generate_cut_outs(
        scan, standard_depth=4.0, window_size=48, threshold_distance=1.0,
        npts=None, center=True, border=29.99):
    '''
    Generate window cut outs that all have a fixed size independent of depth.
    This means ares close to the scanner will be subsampled and those far away
    will be upsampled.
    All cut outs will have values between '-threshold_distance' and
    '+threshold_distance' as they are normalized by the center point by
    default.

    - 'scan' an iterable of radii within a laser scan
    - 'standard_distance' the reference distance (in meters) at which a window
    with 'window_size' get cut out.
    - 'window_size' the number of laser rays that window at 'standard_distance'
    contains.
    - 'nots' the number of final samples to have per window. 'None' means same
    as 'window_size'.
    - 'threshold_distance' the distance in meters from the center point that
    will be used to clamp the laser radii.
    - 'center' whether to center the cut out around the current laser point's
    depth, or keep depth values raw.
    - 'border' the radius value to fill the half of the outermost windows with.
    '''
    s_np = np.fromiter(iter(scan), dtype=np.float32)
    N = len(s_np)

    npts = npts or window_size
    cut_outs = np.zeros((N, npts), dtype=np.float32)

    # The reference distance is 'standard_distance' with 'window_size'
    # If the distance/radii is less than 'standard_distance', the corresponding
    # window size will be large. And vice versa.
    current_size = (window_size * standard_depth / s_np).astype(np.int32)
    start = -current_size // 2 + np.arange(N)
    end = start + current_size
    s_np_extended = np.append(s_np, border)

    if threshold_distance != np.inf:
        near = s_np - threshold_distance
        far = s_np + threshold_distance

    for i in range(N):
        # Get the window.
        sample_points = np.arange(start[i], end[i])
        sample_points[sample_points < 0] = -1
        sample_points[sample_points >= N] = -1
        window = s_np_extended[sample_points]

        # Threshold the near and far values, i.e. normalization.
        if threshold_distance != np.inf:
            window = np.clip(window, near[i], far[i])

        # Shift everything to be centered around the middle point.
        if center:
            window -= s_np[i]

        # Resample
        interp = cv2.INTER_AREA if npts < len(window) else cv2.INTER_LINEAR
        cut_outs[i, :] = cv2.resize(
                window[None], (npts, 1), interpolation=interp)[0]

    return cut_outs


def laser_angles(N, fov=None):
    fov = fov or np.radians(225)
    return np.linspace(-fov * 0.5, fov * 0.5, N)


def rphi2xy(r, phi):
    return r * -np.sin(phi), r * np.cos(phi)


def scan2xy(scan, threshold=None, fov=None):
    s = np.array(scan, copy=True)
    if threshold is not None:
        s[s > threshold] = np.nan
    return rphi2xy(s, laser_angles(len(scan), fov))


def generate_votes(scan, wcs, was, rwc=0.6, rwa=0.4):
    def _closest_detection():
        if len(alldets) == 0:
            return np.zeros_like(scan, dtype=np.int32)

        assert len(alldets) == len(
                radii), "ERROR: Each detection needs a radius."

        scan_xy = np.array(scan2xy(scan)).T

        # Distance in XY space of each laser point with each detection.
        dists = cdist(scan_xy, np.array([rphi2xy(x, y) for x, y in alldets]))

        # Substract the radius from the distances, such that they are <0
        # if inside, >0 if outside.
        dists -= radii

        # Prepend zeros so that argmin is 0 for everything outside.
        dists = np.hstack([np.zeros((len(scan), 1)), dists])

        # Find out who is closest, including the threshold.
        return np.argmin(dists, axis=1)

    def _global2win(dr, dphi):
        dx = np.sin(dphi - phi) * dr
        dy = np.cos(dphi - phi) * dr - r
        return dx, dy

    N = len(scan)
    cls = np.zeros(N, dtype=np.uint32)
    offs = np.zeros((N, 2), dtype=np.float32)

    alldets = list(wcs) + list(was)
    radii = [0.6] * len(wcs) + [0.4] * len(was)
    dets = _closest_detection()
    labels = [0] + [1] * len(wcs) + [2] * len(was)

    for i, (r, phi) in enumerate(zip(scan, laser_angles(N))):
        if 0 < dets[i]:
            cls[i] = labels[dets[i]]
            offs[i, :] = _global2win(*alldets[dets[i] - 1])

    return cls, offs


def win2global(r, phi, dx, dy):
    y = r + dy
    dphi = np.arctan2(dx, y)
    return y / np.cos(dphi), phi + dphi
