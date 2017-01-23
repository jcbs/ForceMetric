"""Module for AFM data analysis aquired with Asylum Research software. This
means reading .ibw data and bring them in an appropriate python format for AFM
analysis"""
#!/usr/bin/python3

import numpy as np
from tqdm import tqdm
from igor import binarywave as ibw
import scipy.constants as cons
from scipy import optimize as opt
from scipy.stats import chisquare
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import ellipe, ellipk
from scipy.signal import argrelextrema
import os
import glob
import itertools
import multiprocessing
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from mpl_toolkits.mplot3d import axes3d
from ContactPointDetermination import GradientCP, MultiplyCP, RovCP, FitCP, GofCP
from mayavi import mlab
from mayavi.api import Engine
from mayavi.sources.api import ArraySource
from mayavi.filters.api import WarpScalar, PolyDataNormals
from mayavi.modules.api import Surface


def DynamicCoefficients(A1, phi1, omega, A1_norm, phi1_norm, k, Q):
    """ Returns elastic and viscose parameter k_s and c_s, respectivly.

    A1_norm   ... list with free and damped amplitude [m]
    phi1_norm ... list with free and damped phase [rad]
    k    ... spring constant of cantilever [N/m]
    """

    A1_free = A1_norm[0]
    A1_near = A1_norm[1]
    phi1_free = phi1_norm[0]
    phi1_near = phi1_norm[1]

    C1 = A1_free * k/Q * np.sqrt(1-1. / (2*Q)**2)

    k_s = C1 * (np.cos(phi1)/A1 - np.cos(phi1_near)/A1_near)
    c_s = C1 / omega * (np.sin(phi1)/A1 - np.sin(phi1_near)/A1_near)
    return k_s, c_s

def IndentationFit(p, delta, gamma=2):
    """ General Form for Indentationmodels"""
    return p[0]*delta**gamma


def errfunc(p, data, delta, gamma=2):
    """ Function for fitting Young's modulus of force-distance data"""
    return IndentationFit(p, delta, gamma)-data


def eccentricity(alpha,beta):
    """Calculates eccentricity of an ellipse"""
    return np.sqrt(1-(np.cos(beta)*(1-np.tan(beta)**2*np.tan(alpha)**2))**2)


def SlopeCorrection(alpha,beta):
    """Calculates correction for force curves on inclined half-space"""
    e = eccentricity(alpha,beta)
    c1 = 1./(1- (np.tan(beta)**2*np.tan(alpha)**2))
    f = c1/np.cos(beta)/ellipk(e)*(np.pi - c1*ellipe(e)/np.cos(beta))
    return f


def RealSlopeCorrection(alpha,beta):
    """Calculates correction for force curves on inclined half-space"""
    e = eccentricity(alpha,beta)
    f = np.pi**2/4/ellipk(e)/ellipe(e)
    return f


def MaximumSlope(height, dimx=20):
    """Caculates maximum slope of a surface"""
    dx = dimx/height.shape[0]
    gx, gy = np.gradient(height, dx)
    m = gx + gy
    theta = np.arcsin(m/(np.sqrt(2+m**2)))
    return theta

def div0(a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def ROV(x, I, smooth=True):
    """Determines the ratio of variances of 1D data with an interval I"""
    if smooth:
        x = gaussian_filter1d(x, 0.1*I)
    N = len(x)
    low = np.array([x[i:i+I].std() for i in range(I,N-I)])
    high = np.array([x[i-I:i].std() for i in range(I,N-I)])
    r = div0(low, high)
    return r


def process1(point, trace=False, method='fiv',
             model='s', fmin=None, fmax=None):
    """ Subprocess for Young's map multiprocessing"""
    try:
        fc = ForceCurve(point)
        fc.trace = trace
        fc.correct(4, method=method)
        return fc
    except:
        return np.nan


def process3(point, trace=False):
    """ Subprocess for Young's map multiprocessing"""
    if point:
        fc = ForceCurve(point)
        fc.trace = trace
        fc.correct(2)
        mini = fc.indentation.Trace().min()
        maxi = fc.indentation.Trace().max()
        newind = np.linspace(mini-maxi,
                             0,
                             2000)
        f = interp1d(fc.indentation.Trace()-maxi, fc.force.Trace(), 'linear')
        newf = f(newind)
        return [newind, newf]
    else:
        return np.nan


def nearestPoint(x, x0):
    """ returns index of x which is closest to x0 """
    check = np.abs(x-x0)
    return np.amax(np.where(check == np.amin(check)))


class Header(dict):
    """ Class which accociates the header as a dictionary of itself"""
    def __init__(self, header):
        for i in header:
            self[i] = header[i]

    def AddHeaderEntries(self, entries):
        for i in entries:
            self[i] = entries[i]


class Wave(Header):
    """
    Classe for igor binary wave files for better access to data than with the
    igor module

    The header information is saved as a dictionary of self, i.e. if you want
    to read the spring constant you can do this by self["SpringConstant"]
    """
    def __init__(self, path, verbose=0):
        self.path = path
        f = ibw.load(path)
        self.wave = f.get('wave')
        H = self.wave.get('note').splitlines()
        header = dict()
        for h in H:
            try:
                dec = h.decode("utf-8").split(":")
                header[dec[0]]= float(dec[1])
            except:
                try:
                    header[dec[0]] = dec[1]
                except:
                    if verbose:
                        print("Can't decode ", h)
        Header.__init__(self, header)
        self.data = 1*np.rollaxis(self.wave.get('wData'), -1)
        label = self.wave.get('labels')
        tmp = []
        while not tmp:
            tmp = label.pop(0)

        tmp.pop(0)
        self.labels = [t.decode("utf-8") for t in tmp]

    def getParam(self, key):
        return self.header[key]

    def getData(self, key):
        idx = self.labels.index(key)
        return self.data[idx]


class FDIndices(object):
    """Class for indices of force-distance data,
    i.e. approach, retract, dwelltime"""
    def __init__(self, indices):
        self.indices = indices
        self.traceidx = [indices[0], indices[1]]
        self.dwell = False
        if len(indices) > 3:
            self.dwell = True
        if self.dwell:
            self.retraceidx = [indices[2], indices[3]]
            self.dwellidx = [indices[1], indices[2]]
        else:
            self.retraceidx = [indices[1], indices[2]]


class FDData(FDIndices):
    """
    Class to describe qantity-distance data, i.e. the data has an approach part
    and a retract part and it could also
    have a dwell part.
    """

    def __init__(self, data, indices):
        FDIndices.__init__(self, indices)
        self.data = data
        self.trace = True

    def Trace(self):
        return self.data[self.traceidx[0]:self.traceidx[1]]

    def Retrace(self):
        return self.data[self.retraceidx[0]:self.retraceidx[1]]

    def Dwell(self):
        if self.dwell:
            return self.data[self.dwellidx[0]:self.dwellidx[1]]
        else:
            print("No dwell time")

    def Data(self):
        if self.trace:
            return self.Trace()
        elif self.trace == 'dwell':
            return self.Dwell()
        else:
            return self.Retrace()


class ParameterDict(object):
    """A parameter dictionary class for Reading and writing information about a
    file into a dictionary with the extension .npy"""
    def __init__(self, path):
        ext = path.split('.')[-1]
        if ext == "ibw":
            directory = os.path.dirname(path)
            basename = os.path.basename(path).split('.')[0]
            parapath = directory + os.sep + 'Parameters.npy'
        elif ext == "npy":
            parapath = path
        self.path = parapath
        check = os.path.isfile(self.path)
        if check:
            print("Load existing File")
            self.parameters = np.load(self.path).item()
        else:
            print("New instance created")
            self.parameters = dict()
        # self.parameters = para[basename]

    def AddFileInfo(self, fnb, dictionary):
        """ This adds another dictionary to the parameter list. In the case fnb
        already exists, the dictionary gets changed"""
        self.parameters[fnb] = dictionary


    def Write(self, path=None):
        if path:
            self.path = path
        np.save(self.path, self.parameters)



class AFMScan(Wave):
    """A class for 2D AFM Scans including all parameters saved by the
    Asylum Research Software. It also has a display function and a plane
    subtraction. If wanted any observable can be projected on the height."""
    def __init__(self, path):
        Wave.__init__(self, path)
        self.scan = [self['FastScanSize'],
                self['SlowScanSize']]
        self.dimensions = self.data.shape[1:]

    def PlaneSubtraction(self, data, direction='xy', xdim=20e-6, ydim=20e-6):
        """Does plane fit to AFM data and subtracts it in either x, y or x-y
        direction"""
        if data in self.labels:
            img = self.getData(data)
        else:
            img = 1*data
        dx, dy = img.shape
        print(img.shape)
        print(dx, dy)
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

    def Display(self, data,
            title='Height',
            zlabel=r'$z$ in um',
            cmap = cm.gray,
            save=False):
        """Displays AFM Data in an appropriate way"""
        if data in self.labels:
            img = self.getData(data)
        else:
            img = 1*data
        area = [0, self.scan[0]*1e6, 0, self.scan[1]*1e6]
        fig, ax = plt.subplots(1, 1)
        cax = ax.imshow(img,
                        extent=area,
                        origin='lower',
                        interpolation='nearest',
                        cmap=cmap)
        ax.set_xlabel(r'$x$ in um')
        ax.set_ylabel(r'$x$ in um')
        ax.set_title(title)
        plt.colorbar(cax, ax=ax, label=zlabel)
        if save:
            plt.savefig(save)

    def ProjectOnHeight(self, data,
            cmap='gnuplot',
            zlabel='E in MPa',
            theta=50, phi=70,
            sub_plane=True,
            trace=True,
            save=False):
        """Projects any quantity (data) with a color coding onto the height
        profile of the AFM scan"""
        area = [0, self.scan[0]*1e6, 0, self.scan[1]*1e6]
        print ("Initialize projection map")
        if trace:
            z = self.getData("HeightTrace")
        else:
            z = self.getData("HeightRetrace")

        print ("Initialize plot")
        z -= z.min()
        z *= 1e6
        if sub_plane:
            z = self.PlaneSubtraction(z)
        dimx, dimy = z.shape
        x = np.linspace(area[0], area[1], dimx)
        y = np.linspace(area[0], area[1], dimy)
        X, Y = np.meshgrid(x, y)
        Z = 1*z
        fig = mlab.figure(figure='Projection on height',
                bgcolor=(1,1,1),
                fgcolor=(0,0,0),
                )
        obj = mlab.mesh(X, Y, Z, scalars=data, colormap=cmap, figure=fig)
        mlab.scalarbar(title=zlabel)
        mlab.orientation_axes()
        mlab.outline()
        # fig.axes()
        if save:
            mlab.savefig(save)
        else:
            mlab.show()

class DynamicViscoelastic(object):
    """ Calculates dynamic viscoelastic properties from AFM tapping mode
    data """
    def __init__ (self, amplitude=False, phase=False,
            free_amplitude=1.e-9, free_phase=np.pi/2.,
            near_amplitude=1.e-9, near_phase=np.pi/2.,
            Q=1., k=1.):
        self.amplitude = amplitude
        self.free_amplitude = free_amplitude
        self.near_amplitude = near_amplitude
        self.phase = phase
        self.free_phase = free_phase
        self.near_phase = near_phase
        self.Q = Q
        self.k = k

    def conservative(self):
        Q = self.Q
        C1 = self.k * self.free_amplitude / Q * np.sqrt(1 - 1./ 4 / Q**2)
        C2 = 1./self.near_amplitude
        phi = self.phase
        phi_n = self.near_phase
        A = self.amplitude
        A_n = self.near_amplitude
        return C1 * (np.cos(phi) / A - C2 * np.cos(phi_n))

    def dissipative(self):
        Q = self.Q
        C1 = self.k * self.free_amplitude / Q * np.sqrt(1 - 1./ 4 / Q**2)
        C2 = 1./self.near_amplitude
        phi = self.phase
        phi_n = self.near_phase
        A = self.amplitude
        A_n = self.near_amplitude
        return C1 * (np.sin(phi) / A - C2 * np.sin(phi_n))


class DynamicYoung(DynamicViscoelastic):
    """Calculates Dynamic Young's modulus from dynamic viscoelastic data
    measurements using known contact models from continuum mechanics.
    Hertz and Sneddon Model are given"""
    def __init__ (self, amplitude=False, phase=False,
            free_amplitude=1.e-9, free_phase=np.pi/2.,
            near_amplitude=1.e-9, near_phase=np.pi/2.,
            Q=1., k=1.):
        DynamicViscoelastic.__init__(self, amplitude=amplitude, phase=phase,
                free_amplitude=free_amplitude, free_phase=free_phase,
                near_amplitude=near_amplitude, near_phase=near_phase,
                Q=Q, k=k)

    def Storage(self, model='Hertz',
            F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9):
        Sneddon = ['Sneddon', 'sneddon', 'S', 's']
        Hertz = ['Hertz', 'hertz', 'H', 'h']
        if not F0:
            F0 = self.F0
        if model in Sneddon:
            SneddonFactor = np.tan(alpha)*8/np.pi*(1 - nu**2)
            E = (self.conservative()/np.sqrt(F0))**2/SneddonFactor
        elif model in Hertz:
            E = (np.sqrt(0.5 * self.conservative() *
                (4./(3*F0))**(1./3))**3 / np.sqrt(R))
        return E

    def Loss(self, model='Hertz',
            F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9, omega_d=1):
        Sneddon = ['Sneddon', 'sneddon', 'S', 's']
        Hertz = ['Hertz', 'hertz', 'H', 'h']
        if not F0:
            F0 = self.F0
        if model in Sneddon:
            SneddonFactor = np.tan(alpha)*8/np.pi*(1 - nu**2)
            E = (self.dissipative() * omega_d/np.sqrt(F0))**2/SneddonFactor
        elif model in Hertz:
            E = (np.sqrt(0.5 * self.dissipative() * omega_d *
                (4./(3*F0))**(1./3))**3 / np.sqrt(R))
        return E

    def ComplexModulus(self, model='Hertz',
            F0=1., alpha=15*np.pi/180, nu=.5, R=10e-9, omega_d=1):
        E_s = self.Storage(model=model,
            F0=F0, alpha=alpha, nu=nu, R=R)
        E_l = self.Loss(model=model,
            F0=F0, alpha=alpha, nu=nu, R=R, omega_d=omega_d)
        E = np.zeros(E_s.shape, dtype=complex)
        E.real = E_s
        E.imag = E_l
        return E
    

class DynamicMechanicAFMScan(DynamicYoung, AFMScan):
    """This class processes 2D AFM scans directly into dynamic mechanical
    information"""

    def load(self, path):
        """This loads the file and all parameters"""
        print("Load file: %s" %path)
        AFMScan.__init__(self, path)
        directory = os.path.dirname(path)
        basename = os.path.basename(path).split('.')[0]
        parapath = directory + os.sep + 'Parameters.npy'
        para = np.load(parapath).item()
        self.AddHeaderEntries(para[basename])
        invOLS = self["AmpInvOLS"]
        A = self.getData("Amplitude1Retrace")
        try:
            phi = self.getData("Phase1Retraceace") * np.pi / 180.
        except:
            phi = self.getData("Phase1Retrace") * np.pi / 180.
        A_free = self['free A1']*1e-3 * invOLS
        A_near = self['near A1']*1e-3 * invOLS
        phi_free = self['free Phi1'] * np.pi / 180.
        phi_near = self['near Phi1'] * np.pi / 180.
        k = self["SpringConstant"]
        Q = self["ThermalQ"]
        self.omega_0 = self["DriveFrequency"] * 2 * np.pi
        self.F0 = self.getData("DeflectionTrace") * k
        DynamicYoung.__init__(self, A, phi,
                A_free, phi_free,
                A_near, phi_near,
                Q, k)





class ContactPoint(object):
    """Determines the Contact point for a force-distant curve. Input is an array
    of force values. Possible algorithms are:
        Force-Indentation variation: 'fiv'
        Ratio-of-Variances: 'rov'
        Gradient: 'grad'
        Godness-of-Fit: 'gof'
        Fit: 'fit'
        """

    def getCP(self, ind=None, f=None, method='fiv', model='h', stds=4):
        cdef int idx = 0
        if ind==None:
            ind = 1*self.indentation.Trace()
        if f==None:
            f = 1*self.force.Trace()
        if method == 'fit':
            idx = FitCP(ind, f, stds)
        elif method == 'rov':
            idx = RovCP(f)
        elif method == 'grad':
            idx = GradientCP(f, ind)
        elif method == 'fiv':
            idx = MultiplyCP(f, ind)
        elif method == 'gof':
            idx = GofCP(f, ind, model)
        self.contactidx = idx
        return self.contactidx


class StaticYoung(ContactPoint):
    """Calculates Young's modulus for different indentation models.
    This means Hertz model for different indenters,
    i.e.: Sphere, Cone, Punch or arbitrary exponent."""

    def Young(self, model='h', ind=None, f=None, p0=[1e6],
            fmin=1e-9, fmax=20e-9,
            imin=5e-9, imax=50e-9,
            R=10e-6,
            alpha=35,
            beta=False,
            constant='force'):

        if ind==None:
            ind = self.indentation.Trace()
        if f==None:
            f = self.force.Trace()
        idx = self.getCP(method='fiv', ind=ind, f=f)
        if self.surfaceidx:
            idx = self.surfaceidx
        f -= f[idx]
        ind -= ind[idx]
        if fmin and fmax:
            f0 = nearestPoint(f, fmin)
            f1 = nearestPoint(f, fmax)
        if imin and imax:
            i0 = nearestPoint(ind, imin)
            i1 = nearestPoint(ind, imax)

        r0 = int(0.2 * len(ind[idx:])) + idx
        r1 = int(0.8 * len(ind[idx:])) + idx

        if constant=='indentation':
            r0 = i0
            r1 = i1
        elif constant=='force':
            r0 = f0
            r1 = f1
        else:
            r0 = r0
            r1 = r1

        if model in ['h', 'H', 'hertz', 'Hertz']:
            gamma = 1.5
            if not R:
                R = float(input("What's the radius of the sphere "))
            G = 4./3.*np.sqrt(R)
        elif model in ['s', 'S', 'sneddon', 'Sneddon']:
            gamma = 2
            alpha *= np.pi/180
            if beta:
                xi = SlopeCorrection(alpha, beta)
            else:
                xi = 1
            G = 2.*np.tan(alpha)*xi/np.pi
        elif model in ['p', 'P', 'punch', 'Punch']:
            gamma = 1
            R = float(input("What's the radius of the cylinder "))
            G = 2.*R
        elif type(model) != str:
            gamma=model
            G = 1.
        try:
            fit = opt.leastsq(errfunc, p0,
                              args=(f[r0:r1],
                                    ind[r0:r1],
                                    gamma))
            exp = IndentationFit(fit[0], ind[r0:r1], gamma)
            self.chi_sq = chisquare(f[r0:r1]*1e9, exp*1e9, ddof=1)
            self.fit = fit
            E = fit[0][0]/G
        except:
            print('Unable to fit %s' % self.path)
            E = np.nan
        self.E = E
        return E


class ForceCurve(StaticYoung, Wave):
    """Instance for an force-distance experiment"""
    def __init__(self, path=None):
        if path:
            Wave.__init__(self, path)
            self.idxs = [int(ix) for ix in self['Indexes'].split(',')]
            self.k = self["SpringConstant"]
            self.v = self["Velocity"]
            self.zsnr = FDData(self.getData('ZSnsr'), self.idxs)
            self.time = FDData(self.getData('ZSnsr')*self.v, self.idxs)
            self.deflection = FDData(self.getData('Defl'), self.idxs)
            self.force = FDData(self.getData('Defl')*self.k, self.idxs)
            self.indentation = FDData(self.getData('ZSnsr') -
                    self.getData('Defl'),
                    self.idxs)
            if "Amp" in self.labels:
                self.amp1 = FDData(self.getData('Amp'), self.idxs)
            if "Amp2" in self.labels:
                self.amp2 = FDData(self.getData('Amp2'), self.idxs)
            if "Phase" in self.labels:
                self.phase1 = FDData(self.getData('Phase'), self.idxs)
            if "Phas2" in self.labels:
                self.phase2 = FDData(self.getData('Phas2'), self.idxs)
        self.trace = True
        self.adhesion = 0
        self.surfaceidx = 0

    def load(self, path):
        """Loads ibw file if no path was given when creating the instance"""
        self.__init__(path)

    def correct(self, stds=4, method='fiv',
                     fitrange=0.6,
                     fmin=25e-9, fmax=100e-9):
        """Determins the contact point with your favourite method"""
        cdef int contactidx
        contactidx = self.getCP(stds=stds, method=method)
        self.zsnr.data -= self.zsnr.Trace()[contactidx]
        self.deflection.data -= self.deflection.Trace()[contactidx]
        if self.force.Trace()[contactidx] < self.force.Trace()[0]:
            self.force.data -= self.force.Trace()[0]
            self.adhesion = self.force.data.min()
        else:
            self.force.data -= self.force.Trace()[contactidx]
        self.indentation.data -= self.indentation.Trace()[contactidx]
        self.time.data -= self.time.Trace()[contactidx]

    def difference(self, method='fiv'):
        """calculates the difference between load and unload z-sensr position
        using your favourite method to determin the contact point"""
        self.correct(method=method)
        ind1 = self.indentation.Trace()*1
        f1 = self.force.Trace()*1
        cidx1 = self.contactidx*1
        print(cidx1)

        ind2 = self.indentation.Retrace()[::-1]*1
        f2 = self.force.Retrace()[::-1]*1
        d = []
        for i in range(len(f1[cidx1:])):
            i += cidx1
            p = nearestPoint(f2[cidx1:], f1[i])
            p += cidx1
            d.append(ind2[p] - ind1[i])

        return np.array(d)

    def surface_idx(self, method='fiv'):
        """Calculates the index of the surface release of the retrace curve"""
        self.correct(method=method)
        f = self.force.Trace()*1
        ind = self.indentation.Trace()*1
        ix = self.contactidx*1
        d = self.difference(method=method)
        k = self.k
        F = f*1
        F[ix:] -= k*d
        sidx = nearestPoint(F[ix:], 0) + ix
        self.surfaceidx = sidx
        return sidx
        

    def ContactPhase(self, dist=-10e-9):
        phi = self.phase1.Trace()
        contactidx = self.contactidx
        if not contactidx:
            contactidx = self.getCP()
        ind = self.indentation.Trace()
        idx = nearestPoint(ind, dist)
        phi_c = phi[idx]
        return phi_c

    def ContactAmp(self, dist=-10e-9, unit='V'):
        amp = self.amp1.Trace()
        contactidx = self.contactidx
        iols = 1.
        if not contactidx:
            contactidx = self.getCP()
        ind = self.indentation.Trace()
        idx = nearestPoint(ind, dist)
        if unit == 'V':
            iols = self['AmpInvOLS']
        a_c = amp[idx]/iols
        return a_c

    def plot(self, X='indentation', save=False):
        micro = 1e-6
        nano = 1e-9
        if self.data.shape[0] == 3:
            f, ax1 = plt.subplots(1)
        elif self.data.shape[0] == 5:
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

        ax1.plot(self.indentation.Trace()/micro,
                 self.force.Trace()/nano, c='r', label='Trace')
        ax1.plot(self.indentation.Retrace()/micro,
                 self.force.Retrace()/nano, c='b', label='Retrace')
        plt.ylabel(r'$F$ in nN')
        ax1.legend(loc='upper left')
        ax1.grid()
        if self.data.shape[0] == 5:
            ax2.plot(self.indentation.Trace()/micro,
                     self.phase.Trace(),
                     c='r', label='Trace')
            ax2.plot(self.indentation.Retrace()/micro,
                     self.phase.Retrace(), c='b', label='Retrace')
            plt.xlabel(r'$\delta$ in um')
            plt.ylabel(r'$\phi$ in deg')
            plt.legend(loc='upper left')

            ax3.plot(self.indentation.Trace()/micro,
                     self.amp.Trace()/nano, c='r', label='Trace')
            ax3.plot(self.indentation.Retrace()/micro,
                     self.amp.Retrace()/nano, c='b', label='Retrace')
            ax2.grid()
            plt.xlabel(r'$\delta$ in um')
            plt.ylabel(r'$A$ in nm')
            plt.legend(loc='upper left')
            ax3.grid()
        plt.show()

class Multicurve(ForceCurve):
    def __init__(self, path):
        ForceCurve.__init__(self, path)

    def LoadCurves(self, trace=True):
        p = self.path
        dirname = os.path.dirname(p)
        fnb = os.path.basename(p)
        ext = fnb.split('.')[-1]
        l1 = len(ext)*-1 - 1
        l0 = l1 - 4
        fn = fnb.replace(fnb[l0:l1], '*')
        print('fn = %s' %fn)
        files = np.sort(glob.glob(dirname + os.sep + fn))
        self.files = files

        num_cores = multiprocessing.cpu_count()
        self.loadcurves = True

        curves = Parallel(n_jobs=num_cores)(
            delayed(process1)(point, trace) for point in tqdm(files))
        print("All curves in cache.")
        self.curves = curves

    def Scatter(self, method='fiv',):
        if not self.curves:
            self.LoadCurves()
        curves = []
        for fc in self.curves:
            if type(fc) != type(np.nan):
                curves.append(fc)
        self.curves = curves
        ind = [fc.indentation.Trace() for fc in self.curves if fc]
        f = [fc.force.Trace() for fc in self.curves if fc]
        indentation = list(itertools.chain.from_iterable(ind))
        force = list(itertools.chain.from_iterable(f))
        self.force = np.array(force)
        self.indentation = np.array(indentation)



# class ForceMap:
    # """ Instance of an indentation experiment on multiple points"""
    # def __init__(self, path):
        # self.path = path
        # self.MeasuredLines = np.sort(glob.glob(
            # self.path + os.sep + 'FMLine*' + os.sep))
        # self.MeasuredPoints = np.array([np.sort(glob.glob(
            # line + 'Line*Point*.ibw'))
            # for line in self.MeasuredLines])
        # fc0 = ForceCurve(self.MeasuredPoints[0][0])
        # Lines = int(fc0.getParam('ScanLines'))
        # Points = int(fc0.getParam('ScanPoints'))
        # # check experiment for completeness
        # if self.MeasuredPoints.shape == (Lines, Points):
            # Complete = True
            # print('Experiment complete')
        # else:
            # Complete = False
            # print('Experiment incomplete')
        # if Complete:
            # self.experiment = self.MeasuredPoints
        # else:
            # self.experiment = [[self.path + os.sep +
                               # 'FMLine%s/Line%sPoint%s.ibw'
                                # % (str(i).zfill(4),
                                   # str(i).zfill(4),
                                   # str(j).zfill(4))
                                # for j in range(Points)]
                               # for i in range(Lines)]
    # def LoadCurves(self, trace=False):
        # num_cores = multiprocessing.cpu_count()
        # exp = self.experiment
        # self.loadcurves = True

        # curves = [Parallel(n_jobs=num_cores)(
            # delayed(process1)(point, trace) for point in lines)
            # for lines in exp]
        # print("All curves in cache.")
        # self.curves = curves


    # def HeightMap(self, trace=False):
        # if not self.loadcurves:
            # self.LoadCurves(trace)
        # curves = self.curves
        # Hmap = []
        # exp = np.array(self.experiment)
        # N = exp.shape[0]
        # M = exp.shape[1]
        # for i in np.arange(N):
            # for j in np.arange(M):
                # fc = curves[i][j]
                # if i == 0:
                    # try:
                        # zidx = fc.get('ZSnsr')
                    # except:
                        # zidx = 0
                # try:
                    # if trace:
                        # zsnr = fc.data[zidx][:fc.idx]
                    # else:
                        # zsnr = fc.data[zidx][fc.idx:][::-1]
                    # idx = fc.contactidx
                    # z = zsnr[idx]
                # except:
                    # z = np.nan
                # Hmap.append(z)
        # Hmap = np.array(Hmap).reshape(N,M)
        # Hmap -= np.nanmax(Hmap)
        # Hmap *= -1
        # self.Hmap = Hmap
        # return Hmap

    # def Slope(self, dx=1):
        # if not self.Hmap:
            # hmap = self.HeightMap()
        # else:
            # hmap = self.Hmap

        # gx,gy = np.gradient(hmap,dx)
        # return gx,gy

    # def YoungMap(self, model='s', trace=False,
              # fmin=10e-9, fmax=50e-9,
              # imin=5e-9, imax=50e-9,
              # R=50e-9,
              # beta=False,
              # alpha=35,
              # constant='indentation'):
        # exp = np.array(self.experiment)
        # curves = np.array(self.curves)
        # N = exp.shape[0]
        # M = exp.shape[1]
        # Emap = np.zeros((N,M))
        # for i in np.arange(N):
            # for j in np.arange(M):
                # try:
                    # fc = curves[i,j]
                    # E = fc.Young(model=model,
                            # fmin=fmin, fmax=fmax,
                            # imin=imin, imax=imax,
                            # R=R,
                            # beta=False,
                            # alpha=35,
                            # constant='indentation')
                    # Emap[i,j] = E
                # except:
                    # print("Can't fit E")
                    # Emap[i,j] = np.nan
        # return Emap

    # def GofMap(self):
        # exp = np.array(self.experiment)
        # curves = np.array(self.curves)
        # N = exp.shape[0]
        # M = exp.shape[1]
        # gof = np.zeros((N,M))
        # for i in np.arange(N):
            # for j in np.arange(M):
                # try:
                    # fc = curves[i,j]
                    # chi  = fc.chi_sq[0]
                    # gof[i,j] = chi
                # except:
                    # print("No chi_square available")
                    # gof[i,j] = np.nan
        # return gof

    # def AdhesionMap(self):
        # exp = np.array(self.experiment)
        # curves = np.array(self.curves)
        # N = exp.shape[0]
        # M = exp.shape[1]
        # Admap = np.zeros((N,M))
        # for i in np.arange(N):
            # for j in np.arange(M):
                # try:
                    # fc = curves[i,j]
                    # ad = fc.adhesion
                    # Admap[i,j] = ad
                # except:
                    # print("No Adhesion available")
                    # Admap[i,j] = np.nan
        # return Admap
        

    # def AverageCurve(self, trace=False):
        # num_cores = multiprocessing.cpu_count()
        # exp = self.experiment
        # FD = [Parallel(n_jobs=num_cores)(
            # delayed(process3)(point, trace) for point in lines)
            # for lines in exp]
        # return np.array(FD).mean(0).mean(0)

class PolymerBrush(ForceCurve):
    """
    Calculates Polymerbrushes according to Isralachvilli for Force Curves with
    PYRAMIDAL indenter.

    Trace is on position 0  and retrace on position 1.

    alpha = half opening angle of tip (if pyramidal tip)
    temperature = total temperature of experiment
    """

    def CalculateBrush(self):
        self.k = self.getParam('^SpringConstant')
        self.indentation = [self.load[0]-self.load[1],
                            self.unload[0]-self.unload[1]]
        self.force = [self.load[1]*self.k, self.unload[1]*self.k]
        self.alpha = 35*np.pi/180.
        self.temperature = 298
        low = 300
        high = -100
        pfit = []
        L = []
        N = []
        for i in range(2):
            self.force[i] = self.force[i][self.indentation[i].argsort()]
            self.indentation[i] = np.sort(self.indentation[i])
            self.indentation[i] -= np.amax(self.indentation[i])
            self.indentation[i] *= -1
            self.force[i] -= np.min(self.force[i])
            low = max(np.where(self.force[i] == np.amin(self.force[i]))[0])+1
            tmp = np.polyfit(self.indentation[i][low:high],
                             np.log(np.sort(self.force[i][low:high])),
                             1,
                             w=self.force[i][low:high])
            pfit.append(tmp)
            L.append(2*np.pi/tmp[0])
            Aprime = (25/np.pi*np.tan(self.alpha)**2
                      * self.temperature
                      * cons.Boltzmann*L[-1]**2
                      )
            A = np.exp(tmp[1])
            N.append((A/Aprime)**(2./3.))
        self.pfit = np.array(pfit)
        self.N = np.array(N)
        self.L = np.array(L)
