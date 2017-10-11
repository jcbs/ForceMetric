#!/usr/bin/python3

import numpy as np
import itertools
from PIL import Image, ImageDraw
import scipy.ndimage as ndi
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




class reduced_qty:
    """
        This class gives a reduced quantity like the reduced mass m* = m1*m2/(m1 + m2) in a two body problem or
        k* = k1*k2/(k1 + k2) for parallel springs.
        This means dim(p) = 2 and

            p* = p[0]*p[1]/(p[0] + p[1])

        seek is the parameter for which quantity we are looking for. If seek='r' the output will be the reduce quantity
        if seek=1 the output will the first quantity assuming p[1] is the reduced quantity
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
    dx, dy = img.shape
    x = np.linspace(0, xdim, dx)
    y = np.linspace(0, ydim, dy)
    DX, DY = np.meshgrid(y, x)
    px = np.array([np.polyfit(y, img[i], 1)
        for i in np.arange(dx)]).mean(axis=0)
    py = np.array([np.polyfit(x, img[:,i], 1)
        for i in np.arange(dy)]).mean(axis=0)
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
    corrected -= corrected.min()
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


def velocity(length, time, method='slope', sort_length=False):
    """calculates velocity for length intervals"""
    delta_t = np.roll(time, -1) - time

    if sort_length:
        length = np.sort(length)

    if method == 'slope':
        vel = (np.roll(length, -1) - length) / delta_t
    elif method == 'gradient':
        vel = np.gradient(length, time)
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
