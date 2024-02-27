"""
Really these are super simple functions, which haven't found their way into the library yet.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()
plt.rcParams['figure.figsize'] = (7, 2.5)

from tdwg.lib.holography import *
from tdwg.lib.plot_utils import *

def plot_result(wg, Eout_x_exp, Eout_f_exp, xaxis_exp, faxis_exp, xlim=300, flim=60):
    Iout_x_exp = abs(Eout_x_exp)**2
    Iout_f_exp = abs(Eout_f_exp)**2

    filter_size = 50*u.um
    xspec_sim, fspec_sim, spec_sim = get_spectrogram(wg.Eout_x, wg.x_axis, filter_size)
    xspec_exp, fspec_exp, spec_exp = get_spectrogram(Eout_x_exp, xaxis_exp, filter_size)

    fig, axs = plt.subplots(2, 2, figsize=(9, 5.5)
            , gridspec_kw={'height_ratios': [0.3, 0.7]})

    plt.subplots_adjust(hspace=0.45)

    plt.sca(axs[0, 0])
    plot_norm(wg.x_axis.to("um"), wg.Iout_x, label="simulation", color="tab:blue", alpha=0.8)
    plot_norm(xaxis_exp, Iout_x_exp, label="experiment", color="tab:red", alpha=0.8)
    plot_norm(wg.x_axis.to("um"), wg.Iin_x, label="input", color="gray", ls="--")
    plt.grid()
    plt.xlim(-xlim, xlim)
    plt.legend(ncol=3, loc=(0.5, 1.25))
    plt.title("Intensity (x)")

    plt.sca(axs[0, 1])
    plot_norm(wg.fx_axis.to("1/mm"), wg.Iout_f, label="simulation", color="tab:blue", alpha=0.8)
    plot_norm(faxis_exp, Iout_f_exp, label="experiment", color="tab:red", alpha=0.8)
    plot_norm(wg.fx_axis.to("1/mm"), wg.Iin_f, label="input", color="gray", ls="--")
    plt.grid()
    plt.xlim(-flim, flim)
    plt.title("Intensity (f)")

    plt.sca(axs[1, 0])
    plt.pcolormesh(xspec_sim.to("um").value, fspec_sim.to("1/mm").value, spec_sim, shading='auto', cmap="hot")
    plt.xlabel("x (mm)")
    plt.ylabel("f (1/mm)")
    plt.title("Simulation spectrogram")

    plt.sca(axs[1, 1])
    plt.pcolormesh(xspec_exp.to("um").value, fspec_exp.to("1/mm").value, spec_exp, shading='auto', cmap="hot")
    plt.xlim(-0.7, 0.7)
    plt.ylim(-55, 55)
    plt.xlabel("x (mm)")
    plt.title("Experiment spectrogram")

    for ax in axs[1, :].flatten():
        plt.sca(ax)
        plt.grid(alpha=0.4)
        plt.xlim(-0.6, 0.6)
        plt.ylim(-60, 60)
        plt.xlim(-xlim, xlim)
        plt.ylim(-flim, flim)