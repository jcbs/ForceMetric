#!/usr/bin/python3
"""
Module for AFM data analysis aquired with Asylum Research software. This
means reading .ibw data and bring them in an appropriate python format for AFM
analysis
"""
import os
import glob
import itertools
import numpy as np
import multiprocessing
import h5py
from tqdm import tqdm
from igor import binarywave as ibw
import scipy.constants as cons
from scipy import optimize as opt
from scipy.stats import chisquare
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter, map_coordinates
from scipy.special import ellipe, ellipk
from scipy.signal import argrelextrema
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


def GetLine(data, p0, p1, num=1000):
    """
    Extracts an interpolated line from a dataset.

    Parameters
    ----------
    data: ndarray of dim 2
        image data from which the profile should be extracted
    p0: tuple
        x, y coordinates where the line should start
    p1: tuple
        x, y coordinates where the line should end
    num: int
        number of points for interpolation

    Returns
    -------
    line: ndarray of dim 1
        extracted line
    """
    x0, y0 = p0
    x1, y1 = p1
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    line = map_coordinates(data, np.vstack((x, y)))
    return line


def DynamicCoefficients(A1, phi1, omega, A1_norm, phi1_norm, k, Q):
    """
    Returns elastic and viscose parameter k_s and c_s, respectivly.

    Parameters
    ----------
    A1_norm :    list with free and damped amplitude [m]
    phi1_norm : list with free and damped phase [rad]
    k    : spring constant of cantilever [N/m]

    Returns
    -------
    k_s : ndarray or float
        The spring constant for dynamic indentation experiments
    c_s : ndarray or float
        The viscosity for dynamic indentation experiments
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
    """
    General Form for elastic indentation models, e.g. Hertz contact

    Parameters
    ----------
    p : list of floats
        This is related to the stiffness by a geometric factor which depends on
        the contact model.
    delta : ndarray of floats
        These are the indentation values to fit
    gamma : geometric exponent according to the contact model, e.g. for Hertz
        gamma = 1.5

    Returns
    -------
    res : ndarray of floats

    """

    return p[0] * delta**gamma + p[1]


def errfunc(p, data, delta, gamma=2):
    """
    Error function for fitting Young's modulus of force-distance data using
    scipy.optimize.leastsq minimization.

    Parameters
    ----------
    p : list of floats
        This is related to the stiffness by a geometric factor which depends on
        the contact model.
    delta : ndarray of floats
        These are the indentation values to fit
    gamma : geometric exponent according to the contact model, e.g. for Hertz
        gamma = 1.5
    data : ndarray of floats
        force data to fit. This is usually from an indentation experiment but it
        can also be simulated

    Returns
    -------
    IndentationFit(p, delta, gamma)-data : ndarray of floats
        difference between fit and data
    """
    return IndentationFit(p, delta, gamma)-data


def NeoHookeanBead(delta, p, R=10e-9):
    """
    This function represents the relationship between the force and the
    indentation delta for a neo-hookean material in contact with a spherical
    bead. If adhesion should be considered an additional term has to be
    considered.

    Parameters
    ----------
    p : list of floats
        This is related to the stiffness by a geometric factor which depends on
        the contact model.
    delta : ndarray of floats
        These are the indentation values to fit
    R : radius of the bead [m]

    Returns
    -------
    force : ndarray with floats
        The response force for a neo-Hookean material.
    """
    C = p * np.pi * np.sqrt(R)
    num = delta**(5./2) - 3 * np.sqrt(R) * delta**2 + 3 * R * delta**(3./2)
    den = delta - 2 * np.sqrt(R * delta) + R
    force = C * num / den
    return force


def eccentricity(alpha,beta):
    """
    Calculates eccentricity of an ellipse

    Parameters
    ----------
    alpha : float
        Half-opening angle of the cone [rad]
    beta : float
        Inclination angle of the section plane [rad]

    Returns
    -------
    ecc : float
        Eccentricity of a cone section.
    """
    ecc = np.sqrt(1-(np.cos(beta)*(1-np.tan(beta)**2*np.tan(alpha)**2))**2)
    return ecc


def SlopeCorrection(alpha, beta):
    """
    Calculates correction for conical indentations on an inclined half-space

    Parameters
    ----------
    alpha : float
        Half-opening angle of the cone [rad]
    beta : float
        Inclination angle of the section plane [rad]

    Returns
    -------
    f : float
        correction parameter for mesured force F, i.e.
        :math:`F_{new} = F \cdot f`.
    """
    e = eccentricity(alpha, beta)
    c1 = 1./(1- (np.tan(beta)**2*np.tan(alpha)**2))
    f = c1/np.cos(beta)/ellipk(e)*(np.pi - c1*ellipe(e)/np.cos(beta))
    return f


def RealSlopeCorrection(alpha, beta):
    """
    Calculates correction for conical indentations on an inclined half-space

    Parameters
    ----------
    alpha : float
        Half-opening angle of the cone [rad]
    beta : float
        Inclination angle of the section plane [rad]

    Returns
    -------
    f : float
        correction parameter for mesured force F, i.e.
        :math:`F_{new} = F \cdot f`.
    """

    e = eccentricity(alpha, beta)
    f = np.pi**2/4/ellipk(e)/ellipe(e)
    return f


def MaximumSlope(height, dimx=20):
    """Caculates maximum slope of a surface"""
    dx = dimx/height.shape[0]
    gx, gy = np.gradient(height, dx)
    m = gx + gy
    theta = np.arcsin(m/(np.sqrt(2+m**2)))
    return theta

def div0(a, b):
    """returns zeros for division by zero"""
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c

def ROV(x, I, smooth=True):
    """
    Determines a significant point using the ratio of variances of 1D data with
    an interval I.

    Parameters
    ----------
    x : 1d array
        data with noisy data to determin characteristic point
    I : int
        Interval length for which the ratio of variances should be determined.

    Returns
    -------
        r : float
            ratio of variances
    """
    if smooth:
        x = gaussian_filter1d(x, 0.1*I)

    N = len(x)
    low = np.array([x[i:i+I].std() for i in range(I,N-I)])
    high = np.array([x[i-I:i].std() for i in range(I,N-I)])
    r = div0(low, high)
    return r


def process1(point, trace=False, method='fiv',
             model='s', fmin=None, fmax=None):
    """Subprocess for Young's map multiprocessing"""
    try:
        fc = ForceCurve(point)
        fc.trace = trace
        fc.correct(4, method=method)
        return fc
    except:
        return np.nan


def process3(point, trace=False):
    """Subprocess for Young's map multiprocessing"""
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
    """
    Determines index of x which is closest to x0. This is necessary if the value
    one is interested is not in the array.

    Parameters
    ----------
    x : ndarray with floats
    x0 : float
        value of interest

    Returns
    -------
    idx : int
        largest index for value which is closest to x0
    """
    check = np.abs(x-x0)
    idx = int(np.amax(np.where(check == np.amin(check))))
    return idx


class Header(dict):
    """
    Class which accociates the header as a dictionary of itself
    """
    def __init__(self, header):
        for key in header:
            self[key] = header[key]

    def AddHeaderEntries(self, entries):
        for key in entries:
            self[key] = entries[key]


class Wave(Header):
    """
    Class for igor binary wave files for better access to data than with the
    igor module. If igor file was converted to HDF5 this can be read as well and
    will be in the same structure accessible for easy programming.

    The header information is saved as a dictionary of self, i.e. if you want
    to read the spring constant you can do this by self["SpringConstant"]

    Parameters
    ----------
    path : str
        path of wave file
    Note
    ----
    Only ibw and hdf5 is supported.
    """
    def __init__(self, path, verbose=0):
        self.path = path
        ext = path.split('.')[-1]

        if ext == 'ibw':
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
                        dec = h.decode("utf-8").split(":")
                        header[dec[0]] = dec[1]
                    except:
                        if verbose:
                            print("Can't decode ", h)


            self.data = 1*np.rollaxis(self.wave.get('wData'), -1)
            label = self.wave.get('labels')
            tmp = []
            while not tmp:
                tmp = label.pop(0)
            tmp.pop(0)

            self.labels = [t.decode("utf-8") for t in tmp]

        elif ext == 'hdf5':
            f = h5py.File(path, 'a')
            data = f['data']
            header = dict(data.attrs)
            self.labels = list(data.keys())
            self.data = np.array([data.get(l) for l in self.labels])


        Header.__init__(self, header)

    def getParam(self, key):
        """
        This function gives a parameter from the header. This is only for
        readibility since the parameters can be accessed from the dictionary
        """
        return self.header[key]

    def getData(self, key):
        """
        Returns data from the wave. self.labels shows available data.
        """
        idx = self.labels.index(key)
        return self.data[idx]


class FDIndices(object):
    """
    Class for indices of force-distance data,
    i.e. approach, retract, dwelltime

    Parameters
    ----------
    indices : list of int
        the indices are 3 for only aproach and retract and 4 for a dwell time.

        The first index represents the first value of the approach curve
        (usually 0), the last value of the approach curve (is also the first
        value of the retract curve or if exist dwell time), and so on.
    """
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
    and a retract part and it could also have a dwell part.

    Parameters
    ----------
    data : ndarray of floats
        force distance data, such as indentation data, or force data
    indices : list of int
        indices for trace, retrace, and dwell. see also FDIndices

    Examples
    --------
    >>> from ForceMetric import FDData
    >>> import numpy as np
    >>> force = np.linspace(0, 2, 1000)**2
    """

    def __init__(self, data, indices):
        FDIndices.__init__(self, indices)
        self.data = data
        self.trace = True

    def Trace(self):
        """
        The approach data of a force-indentation curve

        Returns
        -------
        trace : ndarray of floats
        """
        return self.data[self.traceidx[0]:self.traceidx[1]]

    def Retrace(self):
        """
        The retract data of a force-indentation curve

        Returns
        -------
        retrace : ndarray of floats
        """
        return self.data[self.retraceidx[0]:self.retraceidx[1]]

    def Dwell(self):
        """
        The dwell data of a force-indentation curve

        Returns
        -------
        dwell : ndarray of floats
        """
        if self.dwell:
            return self.data[self.dwellidx[0]:self.dwellidx[1]]
        else:
            print("No dwell time")

    def Data(self):
        """
        The whole data of a force-indentation curve

        Returns
        -------
        data : ndarray of floats
        """
        if self.trace:
            return self.Trace()
        elif self.trace == 'dwell':
            return self.Dwell()
        else:
            return self.Retrace()


class ParameterDict(object):
    """
    A parameter dictionary class for Reading and writing information about a
    file into a dictionary with the extension .npy

    Parameters
    ----------
    path : str
        Path from which to load a parameter dictionary/header or where to save
        it if it does not exist.
    Note
    ----
    At the moment only .npy and .ibw can be added.
    """
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
    """
    A class for 2D AFM Scans including all parameters saved by the
    Asylum Research Software. It also has a display function and a plane
    subtraction. If wanted any observable can be projected on the height.

    Parameters
    ----------
    path : str
        path of AFM file
    Note
    ----
    Only ibw and hdf5 is supported.
    """
    def __init__(self, path):
        Wave.__init__(self, path)
        self.scan = [self['FastScanSize'],
                     self['SlowScanSize']]
        self.dimensions = self.data.shape[1:]

    def PlaneSubtraction(self, data, direction='xy', xdim=20e-6, ydim=20e-6):
        """
        Does plane fit to AFM data and subtracts it in either x, y or x-y
        direction
        """
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
        vmin = np.nanmedian(img) - 2 * np.nanstd(img)
        vmin = max(0, vmin)
        vmax = np.nanmedian(img) + 2 * np.nanstd(img)
        fig, ax = plt.subplots(1, 1)
        cax = ax.imshow(img,
                        extent=area,
                        origin='lower',
                        interpolation='nearest',
                        vmin=vmin, vmax=vmax,
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
    """
    Wrapper for dynamic viscoelastic properties for AFM tapping mode data.

    Parameters
    ----------
    amplitude : float
        measured cantilever amplitude
    phase : float
        measured cantilever phase
    free_amplitude : float
        free cantilever amplitude
    free_phase : float
        free cantilever phase. This should be about :math:`\pi/2`
    near_amplitude : float
        cantilever amplitude about 10 nm above the surface for hydrodynamic
        correction
    near_phase : float
        cantilever phase about 10 nm above the surface for hydrodynamic
        correction. This should be :math:`< \pi/2`. If this is not the case
        you are not in the repulsiv mode
    Q : float
        cantilever quality factor
    k : float
        cantiliver spring constant
    omega0 : float
        resonance/drive frequency of the cantilever :math:`2 \pi f_0 = \omega_0`
    """
    def __init__ (self, amplitude=False, phase=False,
                  free_amplitude=1.e-9, free_phase=np.pi/2.,
                  near_amplitude=1.e-9, near_phase=np.pi/2.,
                  Q=1., k=1., omega0=1.):
        self.amplitude = amplitude
        self.free_amplitude = free_amplitude
        self.near_amplitude = near_amplitude
        self.phase = phase
        self.free_phase = free_phase
        self.near_phase = near_phase
        self.Q = Q
        self.k = k
        self.omega0 = omega0

    def conservative(self):
        """
        This gives the conservative part of the dynamic modulus, i.e. it is
        related to a spring constant. In fact only for the Kelvin-Voigt model
        this is really a spring constant.

        Returns
        -------
        cons : ndarray of floats
            conservative part of the dynamic modulus
        Note
        ----
        This is not a Young's shear modulus, which need an underlying
        indentation model.

        see DynamicYoung
        """
        Q = self.Q
        C1 = self.k * self.free_amplitude / Q * np.sqrt(1 - 1./ 4 / Q**2)
        C2 = 1./self.near_amplitude
        phi = self.phase
        phi_n = self.near_phase
        A = self.amplitude
        A_n = self.near_amplitude
        cons = C1 * (np.cos(phi) / A - C2 * np.cos(phi_n))
        cons[cons<0] = np.nan
        return cons

    def dissipative(self):
        """
        This gives the dissipative part of the dynamic modulus, i.e. it is
        related to a viscosity. In fact only for the Kelvin-Voigt model
        this is really a vicosity.

        Returns
        -------
        diss : ndarray of floats
            dissipative part of the dynamic modulus
        Note
        ----
        This is not a Young's shear modulus, which need an underlying
        indentation model.

        see DynamicYoung
        """
        Q = self.Q
        C1 = self.k * self.free_amplitude / Q * np.sqrt(1 - 1./ 4 / Q**2)
        C2 = 1./self.near_amplitude
        phi = self.phase
        phi_n = self.near_phase
        A = self.amplitude
        A_n = self.near_amplitude
        omega0 = self.omega0
        diss = C1 * (np.sin(phi) / A - C2 * np.sin(phi_n)) / omega0
        diss[diss<0] = np.nan
        return diss


class DynamicYoung(DynamicViscoelastic):
    """
    Calculates Dynamic Young's modulus from dynamic viscoelastic data
    measurements using known contact models from continuum mechanics.
    Hertz and Sneddon Model are given.

    Wrapper for dynamic viscoelastic properties for AFM tapping mode data.

    Parameters
    ----------
    amplitude : float
        measured cantilever amplitude
    phase : float
        measured cantilever phase
    free_amplitude : float
        free cantilever amplitude
    free_phase : float
        free cantilever phase. This should be about :math:`\pi/2`
    near_amplitude : float
        cantilever amplitude about 10 nm above the surface for hydrodynamic
        correction
    near_phase : float
        cantilever phase about 10 nm above the surface for hydrodynamic
        correction. This should be :math:`< \pi/2`. If this is not the case
        you are not in the repulsiv mode
    Q : float
        cantilever quality factor
    k : float
        cantiliver spring constant
    omega0 : float
        resonance/drive frequency of the cantilever :math:`2 \pi f_0 = \omega_0`
    """
    def __init__ (self, amplitude=False, phase=False,
                  free_amplitude=1.e-9, free_phase=np.pi/2.,
                  near_amplitude=1.e-9, near_phase=np.pi/2.,
                  Q=1., k=1., omega0=1., E0=3e7):
        DynamicViscoelastic.__init__(self, amplitude=amplitude, phase=phase,
                                     free_amplitude=free_amplitude, free_phase=free_phase,
                                     near_amplitude=near_amplitude, near_phase=near_phase,
                                     Q=Q, k=k, omega0=omega0)
        self.E0 = E0

    def Storage(self, model='Sneddon',
                F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9):
        """
        Calculates storage modulus from dynamic AFM experiments according to
        the theory of Raman et al (2011)
        """
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

    def MyStorage(self, model='Sneddon',
                F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9, E0=None):
        Sneddon = ['Sneddon', 'sneddon', 'S', 's']
        Hertz = ['Hertz', 'hertz', 'H', 'h']
        if not F0:
            F0 = self.F0

        if not E0:
            E0 = self.E0

        if model in Sneddon:
            SneddonFactor = np.tan(alpha)*2/np.pi*(1 - nu**2)
            E = (self.conservative() * np.sqrt(E0 / F0)
                 / (2 * np.sqrt(SneddonFactor)))
        elif model in Hertz:
            HertzFactor = 3./4 * R / (1 - nu**2)**2
            E = (self.conservative() * (E0 / F0)**(1/3.)
                 / (2 * np.sqrt(HertzFactor)))

        return E

    def Loss(self, model='Hertz',
             F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9, omega0=1.):
        Sneddon = ['Sneddon', 'sneddon', 'S', 's']
        Hertz = ['Hertz', 'hertz', 'H', 'h']
        omega0 = self.omega0

        if not F0:
            F0 = self.F0

        if model in Sneddon:
            SneddonFactor = np.tan(alpha)*8/np.pi*(1 - nu**2)
            E = (self.dissipative() * omega0/np.sqrt(F0))**2/SneddonFactor
        elif model in Hertz:
            E = (np.sqrt(0.5 * self.dissipative() * omega0 *
                            (4./(3*F0))**(1./3))**3 / np.sqrt(R))

        return E

    def MyLoss(self, model='Sneddon',
               F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9, E0=None,
               omega0=1.):
        Sneddon = ['Sneddon', 'sneddon', 'S', 's']
        Hertz = ['Hertz', 'hertz', 'H', 'h']

        omega0 = self.omega0

        if not F0:
            F0 = self.F0

        if not E0:
            E0 = self.E0

        if model in Sneddon:
            SneddonFactor = np.tan(alpha)*2/np.pi*(1 - nu**2)
            E = (self.dissipative()  * omega0 * np.sqrt(E0 / F0)
                 / (2 * np.sqrt(SneddonFactor)))
        elif model in Hertz:
            HertzFactor = 3./4 * R / (1 - nu**2)**2
            E = (self.dissipative() * omega0 * (E0 / F0)**(1/3.)
                 / (2 * np.sqrt(HertzFactor)))

        return E

    def Delta(self, model='Sneddon',
               F0=None, alpha=15*np.pi/180, nu=.5, R=10e-9, E0=None):
        """
        Calculates indentation depth assuming an indentation behaviour
        accourding to Hertz or Sneddon model, with E' beein Youngs' modulus
        """
        Sneddon = ['Sneddon', 'sneddon', 'S', 's']
        Hertz = ['Hertz', 'hertz', 'H', 'h']

        if not E0:
            E0 = self.E0

        if not F0:
            F0 = self.F0


        if model in Sneddon:
            G = np.tan(alpha) * 2 / np.pi / (1 - nu**2)
            gamma = 2
        elif model in Hertz:
            G = 4. / 3 * np.sqrt(R) / (1 - nu**2)**2
            gamma = 3/2.

        delta = (F0 / G / E0)**(1/gamma)

        return delta


    def ComplexModulus(self, model='Hertz',
                       F0=1., alpha=15*np.pi/180, nu=.5, R=10e-9):
        E_s = self.Storage(model=model,
                           F0=F0, alpha=alpha, nu=nu, R=R)
        E_l = self.Loss(model=model,
                        F0=F0, alpha=alpha, nu=nu, R=R)
        E = np.zeros(E_s.shape, dtype=complex)
        E.real = E_s
        E.imag = E_l
        return E


class DynamicMechanicAFMScan(DynamicYoung, AFMScan):
    """
    This class processes 2D AFM scans directly into dynamic mechanical
    information.
    """

    def __init__(self, eigenmode=1):
        self.eigenmode = eigenmode

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
        E0 = self["E0"]
        if self.eigenmode == 1:
            print("first eigenmode")
            try: 
                A = self.getData("Amplitude1Retrace")
            except:
                A = self.getData("AmplitudeRetrace")
            try:
                try:
                    phi = self.getData("Phase1Retraceace") * np.pi / 180.
                except:
                    try:
                        phi = self.getData("PhaseRetracerace") * np.pi / 180.
                    except:
                        phi = self.getData("Phase1Retrace") * np.pi / 180.
            except:
                phi = self.getData("PhaseRetrace") * np.pi / 180.

            A_free = self['free A1']*1e-3 * invOLS
            A_near = self['near A1']*1e-3 * invOLS
            phi_free = self['free Phi1'] * np.pi / 180.
            phi_near = np.pi/2 #self['near Phi1'] * np.pi / 180.
            k = self["SpringConstant"]
            Q = self["ThermalQ"]
            try:
                omega0 = self["ResFreq1"] * 2 * np.pi
            except:
                omega0 = self["DriveFrequency"] * 2 * np.pi
        elif self.eigenmode == 2:
            print("second eigenmode")
            A = self.getData("Amplitude2Retrace")
            try:
                phi = self.getData("Phase2Retraceace") * np.pi / 180.
            except:
                phi = self.getData("Phase2Retrace") * np.pi / 180.

            A_free = self['LossTanAmp2'] * invOLS #self['free A2']*1e-3 * invOLS
            r = A_free / self['free A2']
            A_near = self['near A2']*1e-3 * invOLS / r
            phi_free = self['free Phi2'] * np.pi / 180.
            phi_near = self['near Phi2'] * np.pi / 180.
            k = self["Springconstant2"]
            Q = self["ThermalQHigh"]
            omega0 = self["ResFreq2"] * np.pi * 2

        self.F0 = self.getData("DeflectionRetrace") * self["SpringConstant"]
        DynamicYoung.__init__(self, A, phi,
                                A_free, phi_free,
                                A_near, phi_near,
                                Q, k, omega0, E0)

    def tau(self, model='s', blurs=1, blurl=3):
        es = gaussian_filter(self.Storage(model=model), blurs)
        el = gaussian_filter(self.Loss(model=model), blurl)
        dx = 20./256
        glx, gly = np.gradient(el, dx)
        glx = gaussian_filter(glx, 1)
        gly = gaussian_filter(gly, 1)
        gx, gy = np.gradient(es, dx)
        g = np.hypot(gx/glx, gy/gly)
        return g / self.omega0

    def True_tau(self, model='s', blurs=0, blurl=0, method='max_gradient'):
        es = gaussian_filter(self.MyStorage(model=model), blurs)
        el = gaussian_filter(self.MyLoss(model=model), blurl)

        if self.eigenmode == 2:
            dyna = DynamicMechanicAFMScan(eigenmode=1)
            dyna.load(self.path)
            es1 = gaussian_filter(dyna.MyStorage(model=model), blurs)
            el1 = gaussian_filter(dyna.MyLoss(model=model), blurl)
            kinf1 = dyna.True_kinf(model=model, blurs=blurs, blurl=blurl)
            elr = el - el1
            esr = es - es1 - kinf1
            el = 1 * elr
            es = 1 * esr

        dx = 20./256
        glx, gly = np.gradient(el, dx)
        glx = gaussian_filter(glx, blurl)
        gly = gaussian_filter(gly, blurl)
        gx, gy = np.gradient(es, dx)
        
        if method  == 'max_gradient':
            g = np.hypot(gx, gy) / np.hypot(glx, gly)
        elif method == 'abs_gradient':
            g = np.hypot(gx/glx, gy/gly)
        else:
            g = gx/glx + gy/gly

        G = 1/g
        g = -1*G + np.sqrt(G**2 + 4)

        return g / self.omega0

    def km(self, model='s', blurs=1, blurl=3):
        tau = self.tau(model=model)
        omega = self.omega0
        ot = tau*omega
        el = self.Loss(model=model)
        return el * (1 + ot**2) / ot

    def True_km(self, model='s', blurs=0, blurl=0, method='max_gradient'):
        tau = self.True_tau(model=model, blurs=blurs, blurl=blurl,
                            method=method)
        omega = self.omega0
        ot = tau*omega
        el = self.MyLoss(model=model)
        return el * (1 + ot**2) / ot

    def kinf(self, model='s', blurs=1, blurl=3):
        tau = self.tau(model=model)
        omega = self.omega0
        ot = tau*omega
        el = self.Loss(model=model)
        es = self.Storage(model=model)
        return es - ot * el

    def True_kinf(self, model='s', blurs=0, blurl=0, method='max_gradient'):
        tau = self.True_tau(model=model, blurs=blurs, blurl=blurl,
                            method=method)
        omega = self.omega0
        ot = tau*omega
        el = self.MyLoss(model=model)
        es = self.MyStorage(model=model)
        return es - ot * el




class ContactPoint(object):
    """
    Determines the Contact point for a force-distant curve. Input is an array
    of force values.
    """

    def getCP(self, ind=None, f=None, method='fiv', model='h', stds=4):
        """
        Parameters
        ----------
        ind: ndarray of float
            indentation data for which contact point should be determined.
        f: ndarray of float
            force data for which contact point should be determined.
        method: string
            Force-Indentation variation: 'fiv'\n
            Ratio-of-Variances: 'rov'\n
            Gradient: 'grad'\n
            Godness-of-Fit: 'gof'\n
            Fit: 'fit'\n
        model: str
            indentation model, only needed for gof method.
        """
        # cdef int idx = 0

        if ind is None:
            ind = 1*self.indentation.Trace()

        if f is None:
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
    """
    Calculates Young's modulus for different indentation models.
    This means Hertz model for different indenters,
    i.e.: Sphere, Cone, Punch or arbitrary exponent.
    """

    def Young(self, model='h', ind=None, f=None, p0=[1e4, 0],
              fmin=1e-9, fmax=20e-9,
              imin=5e-9, imax=50e-9,
              R=10e-6,
              alpha=15,
              beta=False,
              constant='force'):

        if ind is None:
            ind = 1*self.indentation.Trace()

        if f is None:
            f = 1*self.force.Trace()

        idx = self.getCP(method='fiv', ind=1*ind, f=1*f)

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
        elif model in ['nh', 'NH', 'neo', 'neohookean']:
            G = 4./9/np.pi * np.sqrt(R)

        try:
            if model in ['nh', 'NH', 'neo', 'neohookean']:
                p = opt.curve_fit(NeoHookeanBead, ind[r0:r1], f[r0:r1], p0=p0)
                E = p / G
            else:
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
            self.idxs = [int(ix)
                         for ix in self['Indexes'].strip(',').split(',')]
            self.k = self["SpringConstant"]
            self.v = self["Velocity"]
            self.zsnr = FDData(self.getData('ZSnsr'), self.idxs)
            self.time = FDData(self.getData('ZSnsr')*self.v, self.idxs)
            self.deflection = FDData(self.getData('Defl'), self.idxs)
            self.force = FDData(self.getData('Defl')*self.k, self.idxs)
            self.indentation = FDData(self.getData('ZSnsr') -
                                      self.getData('Defl'),
                                      self.idxs)
            self.trace = True
            self.adhesion = 0
            self.surfaceidx = 0

            if "Amp" in self.labels:
                self.amp1 = FDData(self.getData('Amp'), self.idxs)

            if "Amp2" in self.labels:
                self.amp2 = FDData(self.getData('Amp2'), self.idxs)

            if "Phase" in self.labels:
                self.phase1 = FDData(self.getData('Phase'), self.idxs)

            if "Phas2" in self.labels:
                self.phase2 = FDData(self.getData('Phas2'), self.idxs)


    def load(self, path):
        """Loads ibw file if no path was given when creating the instance"""
        self.__init__(path)

    def correct(self, stds=4, method='fiv',
                fitrange=0.6, cix=None,
                fmin=25e-9, fmax=100e-9):
        """Determins the contact point with your favourite method"""
        # cdef int contactidx
        if cix:
            contactidx=cix
        else:
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
        elif self.data.shape[0] >= 5:
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

        ax1.plot(self.indentation.Trace()/micro,
                 self.force.Trace()/nano, c='r', label='Trace')
        ax1.plot(self.indentation.Retrace()/micro,
                 self.force.Retrace()/nano, c='b', label='Retrace')
        plt.ylabel(r'$F$ in nN')
        ax1.legend(loc='upper left')
        ax1.grid()
        if self.data.shape[0] >= 5:
            ax2.plot(self.indentation.Trace()/micro,
                     self.phase1.Trace(),
                     c='r', label='Trace')
            ax2.plot(self.indentation.Retrace()/micro,
                     self.phase1.Retrace(), c='b', label='Retrace')
            plt.xlabel(r'$\delta$ in um')
            plt.ylabel(r'$\phi$ in deg')
            plt.legend(loc='upper left')

            ax3.plot(self.indentation.Trace()/micro,
                     self.amp1.Trace()/nano, c='r', label='Trace')
            ax3.plot(self.indentation.Retrace()/micro,
                     self.amp1.Retrace()/nano, c='b', label='Retrace')
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



class ForceMap:
    """
    Instance of an indentation experiment on multiple points
    """
    def __init__(self, path):
        self.path = path
        self.MeasuredLines = np.sort(glob.glob(self.path + os.sep +
                                               'FMLine*' + os.sep))
        self.MeasuredPoints = np.array([np.sort(glob.glob(line +
                                                          'Line*Point*.ibw'))
                                        for line in self.MeasuredLines])
        fc0 = ForceCurve(self.MeasuredPoints[0][0])
        Lines = int(fc0['FMapScanLines'])
        Points = int(fc0['FMapScanPoints'])

        # check experiment for completeness
        if self.MeasuredPoints.shape == (Lines, Points):
            Complete = True
            print('Experiment complete')
        else:
            Complete = False
            print('Experiment incomplete')

        if Complete:
            self.experiment = self.MeasuredPoints
        else:
            self.experiment = [[self.path + os.sep +
                                'FMLine%s/Line%sPoint%s.ibw' % (str(i).zfill(4),
                                                                str(i).zfill(4),
                                                                str(j).zfill(4))
                                for j in range(Points)]
                                for i in range(Lines)]

    def LoadCurves(self, trace=False):
        num_cores = multiprocessing.cpu_count()
        exp = self.experiment
        self.loadcurves = True

        curves = [Parallel(n_jobs=num_cores)(
        delayed(process1)(point, trace) for point in lines)
        for lines in exp]
        print("All curves in cache.")
        self.curves = curves


    def HeightMap(self, trace=False):
        if not self.loadcurves:
            self.LoadCurves(trace)
        curves = self.curves
        Hmap = []
        exp = np.array(self.experiment)
        N = exp.shape[0]
        M = exp.shape[1]
        for i in np.arange(N):
            for j in np.arange(M):
                fc = curves[i][j]
                if i == 0:
                    try:
                        zidx = fc.get('ZSnsr')
                    except:
                        zidx = 0
                try:
                    if trace:
                        zsnr = fc.data[zidx][:fc.idx]
                    else:
                        zsnr = fc.data[zidx][fc.idx:][::-1]
                        idx = fc.contactidx
                    z = zsnr[idx]
                except:
                    z = np.nan

                Hmap.append(z)

        Hmap = np.array(Hmap).reshape(N,M)
        Hmap -= np.nanmax(Hmap)
        Hmap *= -1
        self.Hmap = Hmap
        return Hmap

    # def Slope(self, dx=1):
    # if not self.Hmap:
    # hmap = self.HeightMap()
    # else:
    # hmap = self.Hmap

        # gx,gy = np.gradient(hmap,dx)
        # return gx,gy

    def YoungMap(self, model='s', trace=False,
                 fmin=10e-9, fmax=50e-9,
                 imin=5e-9, imax=50e-9,
                 R=50e-9,
                 beta=False,
                 alpha=35,
                 constant='indentation'):
        exp = np.array(self.experiment)
        curves = np.array(self.curves)
        N = exp.shape[0]
        M = exp.shape[1]
        Emap = np.zeros((N,M))

        for i in np.arange(N):
            for j in np.arange(M):
                try:
                    fc = curves[i,j]
                    E = fc.Young(model=model,
                    fmin=fmin, fmax=fmax,
                    imin=imin, imax=imax,
                    R=R,
                    beta=False,
                    alpha=35,
                    constant='indentation')
                    Emap[i,j] = E
                except:
                    print("Can't fit E")
                    Emap[i,j] = np.nan
        return Emap

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
