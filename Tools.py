#!/usr/bin/python3

import numpy as np
import itertools
from PIL import Image, ImageDraw
import scipy.ndimage as ndi
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.interpolate import interp2d
from scipy.integrate import odeint
from scipy.ndimage import (gaussian_filter, label, distance_transform_edt,
                           generate_binary_structure, binary_erosion, label)
from skimage.morphology import watershed
from skimage.feature import peak_local_max, canny
from skimage.filters import hsobel, vsobel
from skimage import dtype_limits
from scipy.signal import argrelmin, argrelmax
from itertools import product
from numpy import empty, roll
from collections import defaultdict
from mpl_toolkits.mplot3d import axes3d
import time
from datetime import datetime


def CalcIgorTime(t):
    t0_igor = datetime(1904, 1, 1, 0, 0, 0)
    t0_linux = datetime(1970, 1, 1, 0, 0, 0)
    delta_t = t0_linux - t0_igor
    new_t = t - delta_t.total_seconds()
    return new_t


def CalcCellLength(cell, px_size):
    n1 = np.mean(cell['apical_wall']['nodes'], axis=0)
    n2 = np.mean(cell['basal_wall']['nodes'], axis=0)
    d = (n1 - n2) * px_size
    length = np.linalg.norm(d)
    return length


def CalcLenght(nodes, px_size):
    diff = np.diff(nodes, axis=0) * px_size
    length = np.linalg.norm(diff)
    return length


def GetWallData(sample):
    ana = sample.analysis
    s0 = sample.positions[0].scans[0]
    labels = s0.labels
    px_size = s0['ScanSize'] / s0['PointsLines']
    properties = ['storage', 'loss', 'losstan']
    columns = list(np.copy(properties))
    columns.extend(['length', 'direction'])
    ixs = [labels.index(p) for p in properties]

    WallData = pd.DataFrame(columns=columns)

    for wall in ana.keys():
        direction = ana[wall]
        wdata = pd.DataFrame(columns=columns)
        for wall_no in direction.keys():
            df = pd.DataFrame(data=[np.nanmean(direction[wall_no]['data'][ixs],
                                               axis=1)], columns=properties)
            nodes = direction[wall_no]['nodes']
            df['length'] = CalcLenght(nodes, px_size)
            df['direction'] = wall
            wdata = wdata.append(df)

        WallData = WallData.append(wdata)

    return WallData


def GetCellData(sample, date=True, time=True):
    ana = sample.analysis

    if 'cells' in ana.keys():
        cells = sample.analysis['cells']
        s0 = sample.positions[0].scans[0]
        labels = s0.labels
        px_size = s0['ScanSize'] / s0['PointsLines']
        properties = ['storage', 'loss', 'losstan']
        columns = list(np.copy(properties))

        if date:
            columns.extend(['length', 'cell', 'direction', 'date'])
        else:
            columns.extend(['length', 'cell', 'direction'])

        ixs = [labels.index(p) for p in properties]

        CellData = pd.DataFrame(columns=columns)

        for cell in cells.keys():
            Cell = cells[cell]
            cdata = pd.DataFrame(columns=columns)
            print('\n', cell)
            for cell_no in Cell.keys():
                print(cell_no)
                if cell_no == 'cell_ratio':
                    df = pd.DataFrame(data=[Cell[cell_no][ixs]],
                                      columns=properties)
                    df['length'] = CalcCellLength(Cell, px_size)
                    df['direction'] = cell_no
                    df['cell'] = cell
                    cdata = cdata.append(df)
                    if date:
                        cdata['date'] = sample.path.split('/')[1]
                    if time:
                        cdata['time'] = sample.time
                else:
                    df = pd.DataFrame(data=[np.nanmean(Cell[cell_no]['data'][ixs],
                                                       axis=1)], columns=properties)
                    df['length'] = CalcCellLength(Cell, px_size)
                    df['direction'] = cell_no
                    df['cell'] = cell
                    cdata = cdata.append(df)
                    if date:
                        cdata['date'] = sample.path.split('/')[1]
                        cdata['time'] = sample.time

            CellData = CellData.append(cdata)
        return CellData
    else:
        print('No cells available')


def RadarPlot(data, labels, category, idx=0):
    grp = data.groupby(category)
    means = grp.mean()
    idx = means.index
    stats = np.nanmean(means.loc[idx, labels].values, axis=0)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    return fig, ax


def AddRadarPlot(data, labels, category, idx, ax):
    grp = data.groupby(category)
    means = grp.mean()
    idx = means.index
    stats = np.nanmean(means.loc[idx, labels].values, axis=0)
    # stats = grp.mean().loc[idx, labels].values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)


def center_image_at(img, p0):
    cnx, cny = p0
    Ly, Lx = img.shape
    c0x = int(Lx / 2)
    c0y = int(Ly / 2)
    shiftx = cnx - c0x
    shifty = cny - c0y
    newcx = 2 * c0x
    newcy = 2 * c0x
    newLx = newcx + Lx
    newLy = newcy + Lx
    newcx -= shiftx
    newcy -= shifty
    sx = slice(int(newcx - Lx/2), int(newcx + Lx/2))
    sy = slice(int(newcy - Ly/2), int(newcy + Ly/2))
    newImg = np.full([newLy, newLx], np.nan)
    newImg[sy, sx] = img
    return newImg


def dummy_img(shape, periods=3):
    dimy, dimx = shape
    phix = np.linspace(0, periods * 2 * np.pi, dimx)
    phiy = np.linspace(0, periods * dimy / dimx * 2 * np.pi, dimy)
    Y, X = np.meshgrid(phix, phiy)
    img = np.sin(X) + np.cos(Y)
    return img


def toggle(switch):
    if switch:
        switch = False
    else:
        switch = True
    return switch


def concat_images_corner(imga, imgb, xoffset=0, yoffset=0, direction='horizontal',
                  ontop=True, adjust_z=False):
    """
    Combines two color image ndarrays side-by-side.
    """
    if direction == 'horizontal':
        max_dim = np.maximum.reduce([imga.shape, imgb.shape])

        offset = (abs(yoffset), abs(xoffset))
        tmp_offset = np.array(offset)

        # if (max_dim == imgb.shape).all():
            # tmp = np.copy(imgb)
            # imgb = np.copy(imga)
            # imga = np.copy(tmp)
            # ontop = toggle(ontop)
            # xoffset *= -1
            # yoffset *= -1

        # center_new = np.array(np.divide(max_dim, 2), dtype=int)
        new_img = np.full(np.add(max_dim, np.abs(offset)), np.nan)

        Sa0 = slice(0, imga.shape[0])
        Sa1 = slice(0, imga.shape[1])
        Sb0 = slice(abs(yoffset), abs(yoffset) + imgb.shape[0])
        Sb1 = slice(abs(xoffset), abs(xoffset) + imgb.shape[1])

        xdir = np.sign(xoffset)
        ydir = np.sign(yoffset)

        if ydir == 0:
            ydir = 1
        if xdir == 0:
            xdir = 1

        imga = imga[::ydir, ::xdir]
        imgb = imgb[::ydir, ::xdir]

        if adjust_z:
            top_img = 1 * new_img
            top_img[Sa0, Sa1] = imga
            top_img[Sb0, Sb1] = imgb
            low_img = 1 * new_img
            low_img[Sb0, Sb1] = imgb
            low_img[Sa0, Sa1] = imga

            diff = top_img - low_img
            m = np.nanmean(diff)
            s = np.nanstd(diff)
            mask = np.abs(diff) < m + s
            diff[mask] = np.nan
            add = np.nanmean(diff)

            print(add)

            imgb -= add

        if ontop:
            new_img[Sa0, Sa1] = imga
            new_img[Sb0, Sb1] = imgb
        else:
            new_img[Sb0, Sb1] = imgb
            new_img[Sa0, Sa1] = imga

        return new_img[::ydir, ::xdir]


def concat_3dimages_corners(imga, imgb, xoffset=0, yoffset=0, zoffset=0,
                    transpose=True, ontop=True, center_offset=True,
                    adjust_z=(0, 1)):
    """
    Combines two color image ndarrays side-by-side.
    """
    print("Concating images with reference point being the lower left corner")
    if transpose:
        print("Transpose images")
        imga = np.transpose(imga, axes=(0, 2, 1))
        imgb = np.transpose(imgb, axes=(0, 2, 1))

    offset = (abs(zoffset), abs(yoffset), abs(xoffset))
    max_dim = np.maximum.reduce([imga.shape, np.add(imgb.shape, offset)])

    # center_a = np.array(np.divide(imga.shape, 2), dtype=int)
    # center_b = np.array(np.divide(imgb.shape, 2), dtype=int)

    # if (max_dim == imgb.shape).all():
        # tmp = np.copy(imgb)
        # imgb = np.copy(imga)
        # imga = np.copy(tmp)
        # ontop = toggle(ontop)
        # xoffset *= -1
        # yoffset *= -1
        # zoffset *= -1

    # tmp_offset = np.array(offset)
    # tmp_offset[tmp_offset > 0] = 0
    # new_img = np.full(np.add(max_dim, np.abs(offset)), np.nan)
    new_img = np.full(max_dim, np.nan)

    Sa0 = slice(0, imga.shape[0])
    Sa1 = slice(0, imga.shape[1])
    Sa2 = slice(0, imga.shape[2])
    Sb0 = slice(abs(zoffset), abs(zoffset) + imgb.shape[0])
    Sb1 = slice(abs(yoffset), abs(yoffset) + imgb.shape[1])
    Sb2 = slice(abs(xoffset), abs(xoffset) + imgb.shape[2])

    xdir = np.sign(xoffset)
    ydir = np.sign(yoffset)
    zdir = np.sign(zoffset)

    if ydir == 0:
        ydir = 1
    if xdir == 0:
        xdir = 1
    if zdir == 0:
        zdir = 1

    imga = imga[::zdir, ::ydir, ::xdir]
    imgb = imgb[::zdir, ::ydir, ::xdir]

    if adjust_z:
        for ix in adjust_z:
            top_img = 1 * new_img[ix]
            top_img[Sa1, Sa2] = imga[ix]
            top_img[Sb1, Sb2] = imgb[ix]
            low_img = 1 * new_img[ix]
            low_img[Sb1, Sb2] = imgb[ix]
            low_img[Sa1, Sa2] = imga[ix]

            diff = top_img - low_img
            m = np.nanmean(diff)
            s = np.nanstd(diff)
            mask = np.abs(diff) < m + s
            diff[mask] = np.nan
            add = np.nanmean(diff)

            print(add)

            imgb[ix] -= add

    print("new_img shape: ", new_img.shape)

    if ontop:
        new_img[Sa0, Sa1, Sa2] = imga
        new_img[Sb0, Sb1, Sb2] = imgb
    else:
        new_img[Sb0, Sb1, Sb2] = imgb
        new_img[Sa0, Sa1, Sa2] = imga

    new_img

    if transpose:
        print("Transpose back")
        return np.transpose(new_img[::zdir, ::ydir, ::xdir], axes=(0, 2, 1))
    else:
        return new_img[::zdir, ::ydir, ::xdir]


def concat_images(imga, imgb, xoffset=0, yoffset=0, direction='horizontal',
                  ontop=True, adjust_z=False, center_offset=True):
    """
    Combines two color image ndarrays side-by-side.
    """
    if direction == 'horizontal':
        max_dim = np.maximum.reduce([imga.shape, imgb.shape])

        center_a = np.array(np.divide(imga.shape, 2), dtype=int)
        center_b = np.array(np.divide(imgb.shape, 2), dtype=int)
        offset = (abs(yoffset), abs(xoffset))

        if center_offset:
            new_offset = np.subtract(center_a, np.add(center_b, offset))

        if (max_dim == imgb.shape).all():
            tmp = np.copy(imgb)
            imgb = np.copy(imga)
            imga = np.copy(tmp)
            ontop = toggle(ontop)
            xoffset *= -1
            yoffset *= -1

        # elif not (max_dim == imga.shape).all():
            # for i, m in enumerate(max_dim):
                # if m not in imga.shape:
                    # new_offset[i] = center_a[i] - (center_b[i] + offset[i])
                # else:
                    # new_offset[i] = center_a[i] + offset[i] - center_b[i]

        new_offset[new_offset > 0] = 0
        center_new = np.array(np.divide(max_dim, 2), dtype=int)
        new_img = np.full(np.add(max_dim, np.abs(new_offset)), np.nan)

        Sa0 = slice(int(center_new[0] - imga.shape[0]/2 + 0.5),
                    int(center_new[0] + imga.shape[0]/2 + 0.5))
        Sa1 = slice(int(center_new[1] - imga.shape[1]/2 + 0.5),
                    int(center_new[1] + imga.shape[1]/2 + 0.5))
        Sb0 = slice(int(center_new[0] + abs(yoffset) - imgb.shape[0]/2 + 0.5),
                    int(center_new[0] + abs(yoffset) + imgb.shape[0]/2 + 0.5))
        Sb1 = slice(int(center_new[1] + abs(xoffset) - imgb.shape[1]/2 + 0.5),
                    int(center_new[1] + abs(xoffset) + imgb.shape[1]/2 + 0.5))

        xdir = np.sign(xoffset)
        ydir = np.sign(yoffset)

        if ydir == 0:
            ydir = 1
        if xdir == 0:
            xdir = 1

        imga = imga[::ydir, ::xdir]
        imgb = imgb[::ydir, ::xdir]

        if adjust_z:
            top_img = 1 * new_img
            top_img[Sa0, Sa1] = imga
            top_img[Sb0, Sb1] = imgb
            low_img = 1 * new_img
            low_img[Sb0, Sb1] = imgb
            low_img[Sa0, Sa1] = imga

            diff = top_img - low_img
            m = np.nanmean(diff)
            s = np.nanstd(diff)
            mask = np.abs(diff) < m + s
            diff[mask] = np.nan
            add = np.nanmean(diff)

            print(add)

            imgb -= add

        if ontop:
            new_img[Sa0, Sa1] = imga
            new_img[Sb0, Sb1] = imgb
        else:
            new_img[Sb0, Sb1] = imgb
            new_img[Sa0, Sa1] = imga

        return new_img[::ydir, ::xdir]


def concat_3dimages(imga, imgb, xoffset=0, yoffset=0, zoffset=0,
                    transpose=True, ontop=True, center_offset=True,
                    adjust_z=(0, 1)):
    """
    Combines two color image ndarrays side-by-side.
    """
    if transpose:
        print("Transpose images")
        imga = np.transpose(imga, axes=(0, 2, 1))
        imgb = np.transpose(imgb, axes=(0, 2, 1))

    max_dim = np.maximum.reduce([imga.shape, imgb.shape])

    center_a = np.array(np.divide(imga.shape, 2), dtype=int)
    center_b = np.array(np.divide(imgb.shape, 2), dtype=int)
    offset = (abs(zoffset), abs(yoffset), abs(xoffset))

    if center_offset:
        new_offset = np.subtract(center_a, np.add(center_b, offset))
    else:
        new_offset = np.array(offset)

    if (max_dim == imgb.shape).all():
        tmp = np.copy(imgb)
        imgb = np.copy(imga)
        imga = np.copy(tmp)
        ontop = toggle(ontop)
        xoffset *= -1
        yoffset *= -1
        zoffset *= -1

    new_offset[new_offset > 0] = 0
    center_new = np.array(np.divide(max_dim, 2), dtype=int)
    new_img = np.full(np.add(max_dim, np.abs(new_offset)), np.nan)

    Sa0 = slice(int(center_new[0] - imga.shape[0]/2 + 0.5),
                int(center_new[0] + imga.shape[0]/2 + 0.5))
    Sa1 = slice(int(center_new[1] - imga.shape[1]/2 + 0.5),
                int(center_new[1] + imga.shape[1]/2 + 0.5))
    Sa2 = slice(int(center_new[2] - imga.shape[2]/2 + 0.5),
                int(center_new[2] + imga.shape[2]/2 + 0.5))
    Sb0 = slice(int(center_new[0] + abs(zoffset) - imgb.shape[0]/2 + 0.5),
                int(center_new[0] + abs(zoffset) + imgb.shape[0]/2 + 0.5))
    Sb1 = slice(int(center_new[1] + abs(yoffset) - imgb.shape[1]/2 + 0.5),
                int(center_new[1] + abs(yoffset) + imgb.shape[1]/2 + 0.5))
    Sb2 = slice(int(center_new[2] + abs(xoffset) - imgb.shape[2]/2 + 0.5),
                int(center_new[2] + abs(xoffset) + imgb.shape[2]/2 + 0.5))

    xdir = np.sign(xoffset)
    ydir = np.sign(yoffset)
    zdir = np.sign(zoffset)

    if ydir == 0:
        ydir = 1
    if xdir == 0:
        xdir = 1
    if zdir == 0:
        zdir = 1

    imga = imga[::zdir, ::ydir, ::xdir]
    imgb = imgb[::zdir, ::ydir, ::xdir]

    if adjust_z:
        for ix in adjust_z:
            top_img = 1 * new_img[ix]
            top_img[Sa1, Sa2] = imga[ix]
            top_img[Sb1, Sb2] = imgb[ix]
            low_img = 1 * new_img[ix]
            low_img[Sb1, Sb2] = imgb[ix]
            low_img[Sa1, Sa2] = imga[ix]

            diff = top_img - low_img
            m = np.nanmean(diff)
            s = np.nanstd(diff)
            mask = np.abs(diff) < m + s
            diff[mask] = np.nan
            add = np.nanmean(diff)

            print(add)

            imgb[ix] -= add

    print("new_img shape: ", new_img.shape)

    if ontop:
        new_img[Sa0, Sa1, Sa2] = imga
        new_img[Sb0, Sb1, Sb2] = imgb
    else:
        new_img[Sb0, Sb1, Sb2] = imgb
        new_img[Sa0, Sa1, Sa2] = imga

    if transpose:
        print("Transpose back")
        return np.transpose(new_img[::zdir, ::ydir, ::xdir], axes=(0, 2, 1))
    else:
        return new_img[::zdir, ::ydir, ::xdir]


class DraggableImage:
    def __init__(self, img):
        self.img = img
        self.press = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.img.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.img.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.img.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.img.axes:
            return

        contains, attrd = self.img.contains(event)
        if not contains:
            return
        print('event contains', self.img.get_extent())
        x0, x1, y0, y1 = self.img.get_extent()
        self.img.set_alpha(0.3)
        self.press = x0, x1, y0, y1, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the img if the mouse is over us'
        if self.press is None:
            return
        if event.inaxes != self.img.axes:
            return
        x0, x1, y0, y1,  xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.img.set_extent([x0 + dx, x1 + dx, y0 + dy, y1 + dy])

        self.img.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.img.set_alpha(1)
        self.img.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


class ImageAlignment:

    def __init__(self, imgs):
        self.imgs = list(imgs)
        self.drs = []

        for img in self.imgs:
            dr = DraggableImage(img)
            dr.connect()
            self.drs.append(dr)

    def SetOffsets(self, center=False):
        tmp = []

        if center:
            centers = []
            corners = []
            for dr in self.drs:
                x0, x1, y0, y1 = dr.img.get_extent()
                corners.append([x0, x1, y0, y1])
                # centers.append(np.divide((y1 - y0, x1 - x0, 2)))
                tmp.append((y0, x0))
        else:
            for dr in self.drs:
                x0, x1, y0, y1 = dr.img.get_extent()
                tmp.append((y0, x0))

            tmp = np.array(tmp) + 0.5
            self.offsets = np.array(tmp, dtype=int)

        # self.offsets = np.round(
            # np.diff(tmp, axis=0).cumsum().reshape(tmp[1:].shape()) - tmp[1:],
            # 0)


class ImageShift(object):

    def __init__(self, img, xoff=0, yoff=0):
        self.image = img
        self.xoffset = xoff
        self.yoffset = yoff

    def xshift(self, xshift):
        self.xoffset += xshift

    def yshift(self, yshift):
        self.yoffset += yshift

    def xyshift(self, xshift, yshift):
        self.xoffset += xshift
        self.yoffset += yshift

    def apply_shift(self):
        size = np.add(self.image.shape, (abs(2 * self.yoffset), abs(2 * self.xoffset)))
        dimy, dimx = self.image.shape
        new = np.full(size, np.nan)
        cy, cx = np.add(np.divide(new.shape, (2, 2)), (self.yoffset,
                                                       self.xoffset))
        cy = int(cy)
        cx = int(cx)
        new[cy - int(dimy / 2):cy + int(dimy / 2), cx - int(dimx / 2):cx +
            int(dimx / 2)] = self.image
        self.image = 1 * new

    def get_original(self):
        mask = np.isnan(self.image)
        self.image = self.image[~mask]


class Stitching(object):

    def __init__(self, images):
        self.images = images

    def AddImage(self, image):
        self.images.append(image)

    def FullImage(self):
        for i, img in enumerate(self.images):
            if i == 0:
                old = 1 * img
            else:
                new = concat_images(old, img, xoffset=0, yoffset=0)
                old = 1 * new

        self.total = 1 * new
        return self.total


class reduced_qty:
    """
        This class gives a reduced quantity like the reduced mass m* =
        m1*m2/(m1 + m2) in a two body problem or
        k* = k1*k2/(k1 + k2) for parallel springs.
        This means dim(p) = 2 and

            p* = p[0]*p[1]/(p[0] + p[1])

        seek is the parameter for which quantity we are looking for. If
        seek='r' the output will be the reduce quantity
        if seek=1 the output will the first quantity assuming p[1] is the
        reduced quantity
    """
    def __init__(self, p, seek='r'):
        if seek=='r':
            self.x = p[0]
            self.y = p[1]
            self.r = 1./(1./p[0] + 1./p[1])
        else:
            self.y = p[0]
            self.r = p[1]
            self.x = 1./(1./p[1] - 1./p[0])

    def __str__(self):
        print("The reduced quantity p* = %.2e" %self.r)


class Qty_vs_Discance:
    def __init__(self, qty, dist):
        self.qty = qty
        self.dist = dist

    def dist(self):
        return self.dist

    def qty(self):
        return self.dist


def SimForce(x, E, g=10e-6, model='h', noise=4e-4):
    hertz = ['h', 'H', 'hertz', 'Hertz']
    sneddon = ['s', 'S', 'sneddon', 'Sneddon']
    if model in hertz:
        G = 4./3.*np.sqrt(g)
        gamma = 1.5
    elif model in sneddon:
        G = 2./np.pi*np.tan(g)
        gamma = 2
    f = 1*x
    f[x<0] = 0
    f = G*E*np.power(f,gamma)
    if noise:
        noise = np.random.normal(0, noise*f.max(), len(x))
    else:
        noise = 0
    f += noise
    return f


def dijkstra(aGraph,start):
    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in aGraph])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newDist = currentVert.getDistance() \
                    + currentVert.getWeight(nextVert)
            if newDist < nextVert.getDistance():
                nextVert.setDistance( newDist )
                nextVert.setPred(currentVert)
                pq.decreaseKey(nextVert,newDist)


def ForceFitPowerlaw(p0, f, x, model='h'):
    """
    This fits a force curve according to the power law A*delta**gamma
    the model can be either Hertz, i.e. model in ['h', 'H', 'hertz', 'Hertz']
    or Sneddon i.e. model in ['s', 'S', 'sneddon', 'Sneddon']
    or you can choose any arbitrary model i.e. model is a scalar
    """
    hertz = ['h', 'H', 'hertz', 'Hertz']
    sneddon = ['s', 'S', 'sneddon', 'Sneddon']
    if model in hertz:
        model = 3./2
        def erf(p, f, x, model):
            return f - p[0]*np.power(x,model)
    elif model in sneddon:
        model = 2.
        def erf(p, f, x, model):
            return f - p[0]*np.power(x,model)
    else:
        def erf(p, f, x, model):
            return f - p[0]*np.power(x,model)

    fit = leastsq(erf, p0, args=(f,x,model))[0]
    return fit


def ForceFitMultiply(f, x, model=False):
    hertz = ['h', 'H', 'hertz', 'Hertz']
    sneddon = ['s', 'S', 'sneddon', 'Sneddon']
    if model in hertz:
        gamma = 3./2
    elif model in sneddon:
        gamma = 2.
    f -= f.min()
    fm = f.max()
    b = np.linspace(0.5*fm, 0.9*fm, 1000)
    m = np.array([(x*(f-r)).argmin() for r in b])
    fit = np.polyfit(np.log(x[m]), np.log(b), 1)
    if model:
        print("exponent differenz d = %.2e" %(fit[0] - gamma))
    else:
        gamma = 1*fit[0]
    return gamma, np.exp(fit[1])/(1+gamma)


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def Trajectory(r0, U, t, dimx=None, dimy=None):
    if not dimx:
        dimx = U.shape[0]
    if not dimy:
        dimy = U.shape[1]
    x, dx = np.linspace(0, dimx, U.shape[0], retstep=True)
    y, dx = np.linspace(0, dimy, U.shape[1], retstep=True)
    vy, vx = np.gradient(U, dx)
    dfunx = interp2d(x, y, vx*-1, 'linear')
    dfuny = interp2d(x, y, vy*-1, 'linear')
    dfun = lambda r, t: [dfunx(r[0], r[1])[0], dfuny(r[0], r[1])[0]]
    R = odeint(dfun, r0, t)
    return R


def UniqueTuple(coord):
    coord_tuple = [tuple(x) for x in coord]
    unique_coord = sorted(set(coord_tuple),
                          key=lambda x: coord_tuple.index(x))
    unique_index = [coord_tuple.index(x) for x in unique_coord]
    coord = [coord[ix] for ix in unique_index]
    return coord


def PlaneSubtraction(data, direction='xy', xdim=20, ydim=20):
    """Does plane fit to AFM data and subtracts it in either x, y or x-y
    direction"""
    img = 1*data
    dy, dx = img.shape
    x = np.linspace(0, xdim, dx)
    y = np.linspace(0, ydim, dy)
    DX, DY = np.meshgrid(x, y)
    PX = []
    PY = []

    for i, j in zip(np.arange(dy), np.arange(dx)):
        lx = img[i]
        ly = img[:, j]
        maskx = np.isnan(lx)
        masky = np.isnan(ly)

        if len(lx[maskx]) < len(lx):
            s = np.polyfit(x[~maskx], lx[~maskx], 1)
            PX.append(s)

        if len(ly[masky]) < len(ly):
            s = np.polyfit(y[~masky], ly[~masky], 1)
            PY.append(s)

    px = np.nanmean(PX, axis=0)
    py = np.nanmean(PY, axis=0)
    print("x - slope: %.2e, y - slope: %.2e" % (px[0], py[0]))
    print("calculate planes")
    xplane = np.polyval(px, DX)
    yplane = np.polyval(py, DY)

    if direction == 'x':
        print('x plane subtraction')
        correction = xplane
    elif direction == 'y':
        print('y plane subtraction')
        correction = yplane
    else:
        print('x-y plane subtraction')
        correction = xplane + yplane

    corrected = data - correction
    corrected -= np.nanmin(corrected)
    return corrected

def AnticlinalWalls(topo):
    img = gaussian_filter(topo, 16)
    dx = 20./256
    # norm = img/img.max()
    # gx = hsobel(norm)
    # gy = vsobel(norm)
    gx, gy = np.gradient(img, dx)
    g = np.hypot(gx, gy)
    g = PlaneSubtraction(g)
    threshold = np.nanmean(g) - 1.5*np.nanstd(g)
    mask = g < threshold
    tmpmask = img < np.nanmean(img)
    mask[~tmpmask] = False
    mx, my = np.where(mask)
    return mx, my


def TransverseWalls(topo, nx=4, ny=4, axis=0):
    Mx = []
    My = []
    img = gaussian_filter(topo, 16)
    sx, sy = img.shape
    sx /= nx
    sy /= ny
    sx = int(sx)
    sy = int(sy)
    for i in range(4):
        for j in range(4):
            section = img[i*sx:(i+1)*sx, j*sy:(j+1)*sy]
            mx, my = argrelmin(section, axis=axis)
            mx += i*sx
            my += j*sy
            Mx.append(mx)
            My.append(my)
    MX = list(itertools.chain.from_iterable(Mx))
    MY = list(itertools.chain.from_iterable(My))
    return MX, MY


def LongitudinalWallsAlt(topo, topo_threshold=1):
    img = gaussian_filter(topo, 3)
    gx = ndi.sobel(img, axis=0)
    gy = ndi.sobel(img, axis=1)
    g = np.hypot(gx, gy)
    theta = np.arctan2(gx, gy)
    theta_x = np.arctan(gx) * 180 / np.pi
    theta_y = np.arctan(gy) * 180 / np.pi
    theta = np.abs(theta)
    mask_long = canny(np.abs(theta_y), sigma=4, use_quantiles=True,
            low_threshold=.83, high_threshold=.95)
    mask1 = theta < 45
    mask2 = theta < 90
    mask2 = ~mask1 & mask2
    mask3 = theta < 135
    mask3 = ~mask1 & ~mask2 & mask3
    mask4 = theta < 180
    mask4 = ~mask1 & ~mask2 & ~mask3 & mask4
    labels = np.zeros(img.shape, dtype=int)
    labels[mask1] = 1
    labels[mask2] = 2
    labels[mask3] = 3
    labels[mask4] = 4
    mask = canny(theta, sigma=5, use_quantiles=True,
            low_threshold=.83, high_threshold=0.95)
    topo_mask = img > topo_threshold*np.nanmean(img)
    mask[topo_mask] = False
    mask[mask2 | mask3] = False
    mask_long[topo_mask] = False
    lx, ly = np.where(mask_long)
    intervals = 8
    dx = int(256/intervals)
    count, bins = np.histogram(ly, np.arange(intervals + 1)*dx)
    marker = np.zeros(topo.shape, dtype=bool)
    threshold = 1.0*count.std()
    # threshold = count.max() - .1*count.std()
    for i in range(intervals):
        if count[i] > threshold:
            marker[:,i*dx:(i+1)*dx] = 1
    mask = marker & mask
    lx, ly = np.where(mask)
    count, bins = np.histogram(ly, np.arange(17)*16)
    marker = np.zeros(topo.shape, dtype=bool)
    threshold = 1.0*count.std()
    for i in range(16):
        if count[i] > threshold:
            marker[:,i*16:(i+1)*16] = 1
    mask = marker & mask
    mx, my = np.where(mask)
    return mx, my


def LongitudinalWalls(topo, axis=1, padding=0, topo_threshold=.75):
    img = gaussian_filter(topo, 3)
    dx = 20./256
    lx, ly = argrelmin(img, axis=axis, mode='clip')
    mask = np.zeros(img.shape, bool)
    mask[lx, ly] = 1
    intervals = 8
    dx = int(256/intervals)
    count, bins = np.histogram(ly, np.arange(intervals + 1)*dx)
    marker = np.zeros(topo.shape, dtype=bool)
    threshold = 1*count.std()
    for i in range(intervals):
        if count[i] > threshold:
            marker[:,i*dx:(i+1)*dx] = 1
    mask = marker & mask
    topo_mask = img > topo_threshold*np.nanmean(img)
    mask[topo_mask] = False
    mx, my = np.where(mask)
    return mx, my

def PericlinalWalls(topo):
    img = gaussian_filter(topo, 6)
    markers = np.zeros(img.shape, dtype=int)
    std = np.nanstd(img)
    mean = np.nanmean(img)
    markers[img < 2.5*std] = 1
    markers[img > mean + 1.30*std] = 2
    tmp = watershed(img, markers)
    mask = tmp > 1
    px, py = np.where(mask)
    return px, py

def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels

    Parameters
    ----------
    image : array
      The image to smooth

    function : callable
      A function that takes an image and returns a smoothed image

    mask : array
      Mask with 1's for significant pixels, 0 for masked pixels

    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image



def MakeRoi(pts, geometry='polygon',  shape=(256, 256)):
    shape = (shape[1], shape[0])
    img = Image.new('L', shape, 0)
    draw = ImageDraw.Draw(img)
    if geometry is 'polygon':
        draw.polygon(pts, outline=1, fill=1)
    elif geometry is 'line':
        draw.line(pts, fill=1)

    mask = np.array(img, dtype=bool)
    return mask


def BarRoi(pos, width=20, direction='horizontal', shape=(256, 256)):
    w = width/2.
    if direction is 'horizontal':
        pts = [(0, pos + w), (shape[0], pos + w),
               (shape[0], pos - w), (0, pos -w)]
    else:
        pts = [(pos + w, 0), (pos + w, shape[1]),
               (pos - w, shape[1]), (pos - w, 0)]

    return MakeRoi(pts, shape=shape)


def SquareRoi(pos, width=10, shape=(256, 256)):
    w = width/2.
    x, y = pos
    pts = [(x - w, y -w), (x - w, y + w),
           (x + w, y + w), (x + w, y - w)]
    mask = MakeRoi(pts, shape=shape)
    return mask


def RectangularRoi(lower_left, upper_right, shape=(256, 256)):
    v1 = lower_left
    v3 = upper_right
    v2 = (v1[0], v3[1])
    v4 = (v3[0], v1[1])
    pts = [v1, v2, v3, v4]
    mask = MakeRoi(pts, shape=shape)
    return mask


def LineRoi(pts, width=20, shape=(256, 256)):
    w = width/2.
    mask = np.zeros(shape, dtype=bool)
    for p in pts:
        roi = SquareRoi(p, width=width, shape=shape)
        mask = (mask | roi)
    return mask


def velocity(length, time, method='slope', sort_length=False, deg=2):
    """calculates velocity for length intervals"""
    delta_t = np.roll(time, -1) - time

    if sort_length:
        length = np.sort(length)

    if method == 'slope':
        vel = (np.roll(length, -1) - length) / delta_t
    elif method == 'gradient':
        vel = np.gradient(length, time)
    elif method == 'analytical':
        mask = np.isnan(length)
        # length = length[~mask]*1
        # time = time[~mask]*1
        para = np.polyfit(time[~mask], length[~mask], deg)
        vel = np.polyval(para[:-1] * np.arange(1, deg+1)[::-1], time)
    elif method == 'normalized':
        mask = np.isnan(length)
        # length = length[~mask]*1
        # time = time[~mask]*1
        para = np.polyfit(time[~mask], np.log(length[~mask]), deg)
        vel = (np.polyval(para[:-1] * np.arange(1, deg+1)[::-1], time) /
               np.polyval(para, time))
    elif method == 'correlated gradient':
        acc = (np.correlate(length, length, mode='full')[:len(time)])
        vel = np.gradient(np.gradient(acc, time))
    elif method == 'correlated velocity':
        vel = (np.roll(length, -1) - length) / delta_t
        vel = np.sqrt(np.abs((np.correlate(vel,
                                           vel,
                                           mode='full'
                                           )[len(time)-1:]
                              )
                             )
                      )

    return vel


def autocorrelate(x):
    """
    Compute the multidimensional autocorrelation of an nd array.
    input: an nd array of floats
    output: an nd array of autocorrelations
    """

    global i1, i2
    # used for transposes
    t = roll(range(x.ndim), 1)

    # pairs of indexes
    # the first is for the autocorrelation array
    # the second is the shift
    ii = [list(enumerate(range(1, s - 1))) for s in x.shape]

    # initialize the resulting autocorrelation array
    acor = empty(shape=[len(s0) for s0 in ii])

    # iterate over all combinations of directional shifts
    for i in product(*ii):
        # extract the indexes for
        # the autocorrelation array
        # and original array respectively
        i1, i2 = asarray(i).T

        x1 = x.copy()
        x2 = x.copy()

        for i0 in i2:
            # clip the unshifted array at the end
            x1 = x1[:-i0]
            # and the shifted array at the beginning
            x2 = x2[i0:]

            # prepare to do the same for
            # the next axis
            x1 = x1.transpose(t)
            x2 = x2.transpose(t)

        # normalize shifted and unshifted arrays
        x1 -= x1.mean()
        x1 /= x1.std()
        x2 -= x2.mean()
        x2 /= x2.std()

        # compute the autocorrelation directly
        # from the definition
        acor[tuple(i1)] = (x1 * x2).mean()

    return acor
