# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2020

@author: Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data 
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in 
# the file 'all_data.txt' located at the same directory as this script 
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scripts.helper_functions import getInputFile, getConfig, getData
from scripts.helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from scripts.helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from scripts.helper_functions import storeEvaluationResults, load_all_data, get_bootstrap_fit
import numpy as np

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import glob
import pylustrator

pylustrator.start()

files = [
    [
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_11_alginate2%_diff_xposition\inlet\inlet_1\*_result.txt",
    ],
    [
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_21_alginate2%_NIH_xposition_[1-3]\inlet\inlet_[0-9]\*_result.txt",
    ],
    [
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_11_alginate2%_diff_xposition\inlet\inlet_3\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_14_alginate2%_diffxposition_2\inlet\inlet_[0-9]\*_result.txt",
    ],
]


pos = "inlet"
axes = []
for j in range(3):
    data_list = []
    config_list = []
    for pressure in [1, 2, 3]:
        data1, config1 = load_all_data(files[j], pressure=pressure)
        data_list.append(data1)
        config_list.append(config1)

    data = pd.concat(data_list)
    config = config_list[0]

    p = fitStiffness(data, config)
    if 0:
        fits = get_bootstrap_fit(data, config, 100)
        np.save(__file__[:-3]+f"_fits{j}.npy", fits)
    else:
        fits = np.load(__file__[:-3]+f"_fits{j}.npy")
        print("fits", fits.shape)
    p2 = np.std(fits, axis=0)
    for i in config_list:
        config1["fit"] = config["fit"]

    for i in range(3):
        axes.append(plt.subplot(3, 4, 4*j+1+i))
        if i == 0:
            plt.text(0.5, 0.5, f"k = {config['fit']['p'][0]:4.0f}±{p2[0]:3.0f} Pa\n$\\alpha$ = {config['fit']['p'][1]:2.2f}±{p2[1]:2.2f}")
        plotStressStrain(data_list[i], config)
        plt.xlim(0, 300)


    from scipy import stats
    import numpy as np
    for i in range(3):
        plt.subplot(3, 4, 4 * i + 1 + 3)
        xx = fits[:, i]
        if 1:
            kde = stats.gaussian_kde(xx)
            x = [np.linspace(50, 250, 100), np.linspace(0, 1, 100), np.linspace(0., 0.2, 100)][i]
            p, = plt.plot(x, kde(x), "k--", lw=0.8, zorder=2)
        plt.hist(xx, density=True)#, color=p.get_color())
        plt.ylabel("probability density")
        plt.xlabel(["k", "alpha", "offset"][i])

pylustrator.helper_functions.axes_to_grid(axes)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.078992, 0.710283, 0.203663, 0.274770])
plt.figure(1).axes[0].texts[0].set_position([94.591759, 0.084863])
plt.figure(1).axes[1].set_position([0.304615, 0.710259, 0.203663, 0.274770])
plt.figure(1).axes[2].set_position([0.534193, 0.710283, 0.203663, 0.274770])
plt.figure(1).axes[3].set_xlim(0.0, 250.0)
plt.figure(1).axes[3].set_position([0.812565, 0.759895, 0.168636, 0.196030])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[4].set_xlim(0.0, 0.6)
plt.figure(1).axes[4].set_position([0.812565, 0.464302, 0.168636, 0.196030])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[5].set_xlim(-0.02300812139024562, 0.2)
plt.figure(1).axes[5].set_position([0.812565, 0.168708, 0.168636, 0.196030])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[6].set_position([0.078992, 0.411926, 0.203663, 0.274770])
plt.figure(1).axes[6].texts[0].set_position([94.591759, 0.039885])
plt.figure(1).axes[7].set_position([0.306593, 0.410577, 0.203663, 0.274770])
plt.figure(1).axes[8].set_position([0.534193, 0.410577, 0.203663, 0.274770])
plt.figure(1).axes[9].set_position([0.078992, 0.110871, 0.203663, 0.274770])
plt.figure(1).axes[9].texts[0].set_position([94.591759, 0.035540])
plt.figure(1).axes[10].set_position([0.306593, 0.110871, 0.203663, 0.274770])
plt.figure(1).axes[11].set_position([0.534193, 0.110871, 0.203663, 0.274770])
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
