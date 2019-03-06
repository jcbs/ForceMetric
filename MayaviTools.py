import numpy as np
from Tools import PlaneSubtraction
from mayavi import mlab
from mayavi.api import Engine
from mayavi.sources.api import ArraySource
from mayavi.filters.api import WarpScalar, PolyDataNormals
from mayavi.modules.api import Surface


def ProjectOn(data, topography, pixel_size=78e-3, cmap='gnuplot',
              zlabel='E in MPa', theta=50, phi=70, vmin=None, vmax=None,
              dist='auto', sub_plane=False, save=False):
    """Projects any quantity (data) with a color coding onto the height
    profile of the AFM scan"""
    dimy, dimx = topography.shape
    area = [0, dimx * pixel_size, 0, dimy * pixel_size]
    print("Initialize plot")

    dimx, dimy = topography.shape

    if (not vmin and vmin != 0):
        vmin = np.nanmin(data)

    if not vmax:
        vmax = np.nanmax(data)

    if sub_plane:
        print("subtract plane")
        topography = PlaneSubtraction(topography, xdim=dimx*pixel_size,
                                      ydim=dimx*pixel_size)

    print("vmin = %s , vmax = %s" % (vmin, vmax))

    x = np.linspace(area[2], area[3], dimx)
    y = np.linspace(area[0], area[1], dimy)
    X, Y = np.meshgrid(y, x)
    Z = 1 * topography
    fig = mlab.figure(figure='Projection on height',
                      bgcolor=(1, 1, 1),
                      fgcolor=(0, 0, 0),
                      )
    obj = mlab.mesh(X, Y, Z, scalars=data, colormap=cmap, figure=fig,
                    vmin=vmin, vmax=vmax)
    mlab.view(azimuth=phi, elevation=theta, distance=dist)
    mlab.colorbar(title=zlabel, orientation='vertical', nb_labels=5)
    mlab.orientation_axes()
    mlab.outline()
    # fig.axes()
    if save:
        mlab.savefig(save, magnification=3)
        mlab.close("Projection on height")
    else:
        mlab.show()
