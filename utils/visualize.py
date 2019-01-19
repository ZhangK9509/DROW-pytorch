# -*- coding=utf8 -*-

# Author       : Painter
# Created Time : 2019-01-09 Wed 20:31:00
# Filename     : visualize.py
# Email        : painter9509@126.com


import numpy as np
import matplotlib.pyplot as plt

from utils import utils as u


def fix_ax(ax, thresh):
    xr, yr = u.scan2xy(np.full(100, thresh or 35, dtype=np.float32), thresh)
    ax.set_xlim(np.nanmin(xr), np.nanmax(xr))
    ax.set_ylim(np.nanmin(yr), np.nanmax(yr))
    ax.set_aspect("equal", adjustable="box")
    return ax


def show_scan(scan, thresh=14, ax=None, color="#0033FF", alpha=0.33, **figkw):
    x, y = u.scan2xy(scan, thresh)
    if ax is None:
        fig, ax = plt.subplots(**figkw)
    if isinstance(color, (list, tuple, np.ndarray)):
        color = [c for i, c in enumerate(color) if True not in (
            np.isnan(x[i]), np.isnan(y[i]))]
    ax.scatter(x, y, s=10, color=color, alpha=alpha, lw=0)
    return fix_ax(ax, thresh)


def show_det(dets, ax=None, figkw={}, **kw):
    kw.setdefault("color", "#66FF99")
    if ax is None:
        fig, ax = plt.subplots(**figkw)
    for r, phi in dets:
        x, y = u.rphi2xy(r, phi)
        if kw.get("maker") in (None, "circ"):
            kw.setdefault("alpha", 0.3)
            kw.setdefault("radius", 0.5)
            ax.add_artist(plt.Circle((x, y), **kw))
        else:
            kw.setdefault("linewidths", 2)
            kw.setdefault('s', 50)
            ax.scatter(x, y, **kw)
        return ax


def show_votes(scan, cls, offs, ax=None, figkw={}, **arrowkw):
    if ax is None:
        fig, ax = plt.subplots(**figkw)

    arrowkw.setdefault("head_width", 0.1)
    arrowkw.setdefault("head_length", 0.2)

    for (pno, pwc, pwa), (dx, dy), r, phi in zip(
            cls, offs, scan, u.laser_angles(len(scan))):
        if pno > pwc and pno > pwa:
            continue

        vr, vphi = u.win2global(r, phi, dx, dy)
        vx, vy = u.rphi2xy(vr, vphi)

        x, y = u.rphi2xy(r, phi)
        c = ["#00CC00", "#FF0000"][pwc < pwa]

        kw = dict(**arrowkw)
        kw.setdefault("fc", c)
        kw.setdefault("ec", c)
        ax.arrow(x, y, vx - x, vy - y, **kw)
    return ax


def show_cls(scan, cls, ax):
    for (pno, pwc, pwa), r, phi in zip(cls, scan, u.laser_angles(len(scan))):
        if pno > pwc and pno > pwa:
            continue

        x, y = u.rphi2xy(r, phi)
        c = ["#00CC00", "#FF0000"][pwc < pwa]
        ax.scatter(x, y, s=10, color=c)
