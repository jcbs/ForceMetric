
"""Example to load multiple curves and fit the Young's modulus"""
import ForceMetric as fm
from matplotlib import pyplot as plt
import os
import numpy as np

if __name__ == "__main__":
    path = os.path.join("Data", "X60_3_x4_0100.ibw")
    mc = fm.Multicurve(path)  # loads all force curves with the name X60_3_x4_*.ibw
    Youngs = []

    mc.CorrectAll(method='fiv')
    mc.FitAll(model='Hertz', fmin=10e-9, fmax=100e-9)
    Youngs = mc.E

    mc.Scatter()  # creates indentation and force array for all curves to
                  # display scatter plot

    print("E = %.2e +- %.2e Pa" % (np.nanmean(Youngs), np.nanstd(Youngs)))

    fig, ax = plt.subplots()
    ax.plot(mc.indentation * 1e6, mc.force * 1e9, '.')
    ax.set_xlabel(r"$\delta$ (um)")
    ax.set_ylabel(r"$F$ (nN)")

    ax.grid()

    # plot fit
    mc.PlotAverageFit()

    plt.show()
