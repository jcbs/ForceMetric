"""Example for a force-volume curve to load it and find the contact point"""
import ForceMetric as fm
from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = "LRCol0_MS_LG7d0001.ibw"

    # Default plot settings
    fc = fm.ForceCurve(path)
    fc.correct('fiv')
    fc.plot()

    # customized plots
    fig, ax = plt.subplots()
    ax.plot(fc.indentation.Trace() * 1e6, fc.force.Trace() * 1e9, c='r', lw=2)
    fig.suptitle('Custumized plot')
    ax.set_xlabel(r"$\delta$ (um)")
    ax.set_ylabel(r"$F$ (nN)")

    ax.grid()

    # determine and print Young's modulus
    E = fc.Young(model='sneddon', fmin=10e-9, fmax=500e-9)
    print("E = %.2e Pa" % E)

    plt.show()
