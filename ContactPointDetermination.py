#!/usr/bin/python3

import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax, argrelmin
from scipy.optimize import leastsq
import scipy as sp
from matplotlib import pyplot as plt

def div0(a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def FitFunc(p, f, x):
    return f - (p[0] * np.power(x,1.5) - 0*p[1])

def ROV(x, I, smooth=True):
    if smooth:
        x = gaussian_filter1d(x, 0.1*I)
    N = len(x)
    low = np.array([x[i:i+I].std() for i in range(I,N-I)])
    high = np.array([x[i-I:i].std() for i in range(I,N-I)])
    r = div0(low, high)
    return r

def nearestPoint(x, x0):
    """ returns index of x which is closest to x0 """
    check = np.abs(x-x0)
    return np.amax(np.where(check == np.amin(check)))

def FitCP(x, f, stds=4):
    f -= f[0]
    r = int(0.6*len(f))
    fit = np.polyfit(x[:r], f[:r], 1)
    std = f[:r].std()
    diff = f - np.polyval(fit, x)
    if std/f.max()<0.1:
        i=0
    else:
        i = int(np.sqrt(nearestPoint(abs(f), std)))
    idx = nearestPoint(np.abs(diff), stds*std)
    # print("fit ", i, idx)
    return idx-i

def RovCP(f):
    R = []
    for I in np.linspace(10,200,199):
        I = int(I)
        fr = f[I:2*I].std()
        fl = f[:I].std()
        r = fr/fl
        R.append(r)
    I = nearestPoint(np.array(R), 1) + 10
    print('I = %i' %I)
    rov = ROV(f, I)
    cp = (rov.argmax() + I)
    # cp = np.array(cp)
    # h = np.histogram(cp, bins=np.arange(L))[0]
    # m = max(np.where(h==h.max())[0])
    # idx = np.amax(m)
    idx = cp
    return idx

def GradientCP(f, x):
    L = len(f)
    f -= f.min()
    f += f.max()
    dx = x[1] - x[0]
    dx = np.array([x[i+1] - x[i] for i in range(L-1)])
    g = np.gradient(f[1:], dx)
    T = div0(f[1:], g)
    # T[np.isinf(np.abs(T))]=0
    std = np.nanstd(T)*0.5
    print(std)
    # plt.plot(T)
    # plt.plot(np.ones(len(T))*std, c='r')
    # plt.show()
    m = argrelmax(T)
    M = T[m] - std
    v = M[M>0][-1] + std
    idx = nearestPoint(T,v) + 1
    # idx = np.max(np.where(T == v)[0]) + 1
    return idx

def MultiplyCP(f, x):
    f -= f.min()
    x -= x.min()
    tx = 1*x + 2*x.max()
    tx *= 1
    tf = 1*f - f.max()*0.6
    tm = tf*tx/(tf.max()*tx.max())
    idx = tm.argmin()
    # idx = 0
    # m = 0
    # for i in range(11):
        # r = i/10.
        # tf = 1*f - f.max() * r
        # tx = 1*x - x.max() * r
        # tm = tf*tx/(f.max()*x.max())
        # if tm.min() < m:
            # m = 1*tm.min()
            # idx = 1*tm.argmin()
        # print(idx, tm.min(), i)
    return idx

def DeltaECP(f, x, fitfunc, p0=[1e3,0]):
    L = len(f)
    I1 = int(0.15*L)
    I2 = int(0.99*L)
    E = []
    for i in range(I1,I2):
        tf = 1*f
        tf -= tf[i]
        tf[f<0] = 0
        x -= x[i]
        fit = leastsq(fitfunc, p0, args=(tf[i:],x[i:]))[0]
        E.append(fit[0])

    x -= x.min()
    x = gaussian_filter1d(x, 10)
    dx = np.array([x[j+1] - x[j] for j in range(I1,I2)])
    E = np.array(E)
    E = gaussian_filter1d(E, 10)
    dE = -np.gradient(np.log(E), dx)
    # dE = np.array(dE)
    # plt.plot(x[I1:I2], dE)
    # plt.show()
    m = argrelmax(dE)[0]
    v = dE[m].max()
    idx = max(np.where(dE == v)[0]) + I1
    return idx

def GofCP(f, x, model, mp=0.95):
    hertz = ['h', 'H', 'hertz', 'Hertz']
    sneddon = ['s', 'S', 'sneddon', 'Sneddon']
    if model in hertz:
        gamma = 1.5
    elif model in sneddon:
        gamma = 2.0
    else:
        gamma = model

    def errfunc(p, f, x,  gamma):
        return f - p * np.power(x,gamma)

    L = len(f)
    gof = []

    for i in range(int(0.5*L), int(mp*L)):
        f -= f[i]
        x -= x[i]
        tx = 1*x[i:]
        tf = 1*f[i:]
        m = -1 # int(0.3*len(tx))
        fit = leastsq(errfunc, 1e4, (tf[:m], tx[:m], gamma))[0]
        diff = tf[:m] - fit[0] * np.power(tx[:m], gamma)
        rsq = np.dot(diff, diff/tf[:m].mean())
        gof.append(rsq)
    gof = np.array(gof)
    # plt.plot(gof)
    # plt.show()
    # print(gof.argmin(), int(L/2.))
    idx = gof.argmin() + int(L/2.)
    return idx


if __name__ == "__main__":
    from ForceMetric import ForceCurve
    # path = '/media/jacob/MicroscopyData/Data/AFM/PCL/16-07-04 viscosities/H20/B2_H20_1um_0002.ibw'
    # path = '/media/jacob/MicroscopyData/Data/AFM/PCL/16-07-04 viscosities/H20/B2_H20_10um_0153.ibw'
    # fc = ForceCurve(path)
    # fc.trace = True

    # ind = 1*fc.indentation()
    # f = 1*fc.force()
    x = np.linspace(-1.502,0.162,3783)*1e-6
    Rs = 150.e-6
    Rt = 20.e-6
    R = Rt * Rs / (Rt + Rs)
    Et = 3.e9
    nu_t = .33
    Es = 750.e3
    nu_s = .33
    Er = 1./((1 - nu_s**2)/Es + (1 - nu_t**2)/Et)
    G = 4./3 * np.sqrt(R)
    A = G * Er
    f = x*1
    f[x<0] = 0
    f = A * np.power(f,1.5)


    noise = np.random.normal(0, 0.004*f.max(), len(x))
    f1 = sp.ndimage.gaussian_filter1d(f, 50)
    f1 = f + noise
    ind = 1*x
    f = 1*f1

    cp = np.abs(x).argmin()
    i1 = FitCP(ind, f, stds=5)
    i3 = GradientCP(f, ind)
    i2 = RovCP(f)
    i4 = MultiplyCP(f, ind)
    i5 = GofCP(f, x, 'h')
    # t4 = [GradientCP(f, ind) for i in range(100)]
    # print(np.average(t4), np.std(t4))
    # i5 = DeltaECP(f, ind, FitFunc)
    I = [i1,i2,i3,i4, i5]
    d = np.array([x[cp] - x[i] for i in I])
    method = ['fit', 'rov', 'gradient', 'multiply', 'gof']

    print("The actual contact point is: %i" %cp)
    print("Fit CP is: %i with dx = %.2e m" %(i1, d[0]))
    print("Rov CP is: %i with dx = %.2e m" %(i2, d[1]))
    print("Gradient CP is: %i with dx = %.2e m" %(i3, d[2]))
    print("Multiply CP is: %i with dx = %.2e m\n" %(i4, d[3]))
    print("Gof CP is: %i with dx = %.2e m\n" %(i5, d[4]))


    print("The winner is %s" %method[abs(d).argmin()])




    plt.plot(ind - ind[i1], f - f[i1], label='Fit')
    plt.plot(ind - ind[i2], f - f[i2], label='Rov')
    # plt.plot(ind - ind[i3], f - f[i3], label='Gradient')
    plt.plot(ind - ind[i4], f - f[i4], label='Multiply')
    plt.plot(ind - ind[i5], f - f[i5], label='Gof')
    # plt.plot(ind - ind[i5], f - f[i5])
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
