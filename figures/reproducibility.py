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
from scripts.helper_functions import storeEvaluationResults, load_all_data
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
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
    ],
    [
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_21_alginate2%_NIH_xposition_[1-3]\inlet\inlet_[0-9]\*_result.txt",
    ],
    [
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_11_alginate2%_diff_xposition\inlet\inlet_1\*_result.txt",
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_11_alginate2%_diff_xposition\inlet\inlet_3\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_14_alginate2%_diffxposition_2\inlet\inlet_[0-9]\*_result.txt",
    ],
]

pos = "inlet"

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
    for i in config_list:
        config1["fit"] = config["fit"]

    for i in range(3):
        plt.subplot(3, 3, 3*j+1+i)
        if i == 0:
            plt.text(0.5, 0.5, f"k = {config['fit']['p'][0]:3.0f} Pa\n$\\alpha$ = {config['fit']['p'][1]:2.2f}")
        plotStressStrain(data_list[i], config, skip=100)
        plt.xlim(0, 300)


print("parameter", p)
if 0:
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(16.260000/2.54, 5.460000/2.54, forward=True)
    plt.figure(1).axes[0].set_xlim(0.0, 300.0)
    plt.figure(1).axes[0].set_ylim(-0.0, 1.0)
    plt.figure(1).axes[0].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
    plt.figure(1).axes[0].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[0].set_position([0.125000, 0.706022, 0.243705, 0.248342])
    plt.figure(1).axes[0].spines['right'].set_visible(False)
    plt.figure(1).axes[0].spines['top'].set_visible(False)
    plt.figure(1).axes[0].lines[3].set_markersize(2.0)
    plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
    plt.figure(1).axes[0].texts[0].set_ha("center")
    plt.figure(1).axes[0].texts[0].set_position([124.322604, -0.091003])
    plt.figure(1).axes[0].texts[0].set_text("1 bar")
    plt.figure(1).axes[1].set_xlim(0.0, 300.0)
    plt.figure(1).axes[1].set_ylim(-0.0, 1.0)
    plt.figure(1).axes[1].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
    plt.figure(1).axes[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.figure(1).axes[1].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[1].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
    plt.figure(1).axes[1].set_position([0.417446, 0.706022, 0.243705, 0.248342])
    plt.figure(1).axes[1].spines['right'].set_visible(False)
    plt.figure(1).axes[1].spines['top'].set_visible(False)
    plt.figure(1).axes[1].lines[3].set_markersize(2.0)
    plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
    plt.figure(1).axes[2].set_xlim(0.0, 300.0)
    plt.figure(1).axes[2].set_ylim(-0.0, 1.0)
    plt.figure(1).axes[2].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
    plt.figure(1).axes[2].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.figure(1).axes[2].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[2].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
    plt.figure(1).axes[2].set_position([0.709892, 0.706022, 0.243705, 0.248342])
    plt.figure(1).axes[2].spines['right'].set_visible(False)
    plt.figure(1).axes[2].spines['top'].set_visible(False)
    plt.figure(1).axes[2].lines[3].set_markersize(2.0)
    plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
    plt.figure(1).axes[3].set_position([0.125000, 0.408011, 0.243705, 0.248342])
    plt.figure(1).axes[3].spines['right'].set_visible(False)
    plt.figure(1).axes[3].spines['top'].set_visible(False)
    plt.figure(1).axes[3].texts[0].set_position([124.322604, -0.095731])
    plt.figure(1).axes[4].set_position([0.417446, 0.408011, 0.243705, 0.248342])
    plt.figure(1).axes[4].spines['right'].set_visible(False)
    plt.figure(1).axes[4].spines['top'].set_visible(False)
    plt.figure(1).axes[5].set_position([0.709892, 0.408011, 0.243705, 0.248342])
    plt.figure(1).axes[5].spines['right'].set_visible(False)
    plt.figure(1).axes[5].spines['top'].set_visible(False)
    plt.figure(1).axes[6].set_position([0.125000, 0.110000, 0.243705, 0.248342])
    plt.figure(1).axes[6].spines['right'].set_visible(False)
    plt.figure(1).axes[6].spines['top'].set_visible(False)
    plt.figure(1).axes[6].texts[0].set_position([124.322604, -0.114643])
    plt.figure(1).axes[7].set_position([0.417446, 0.110000, 0.243705, 0.248342])
    plt.figure(1).axes[7].spines['right'].set_visible(False)
    plt.figure(1).axes[7].spines['top'].set_visible(False)
    plt.figure(1).axes[8].set_position([0.709892, 0.110000, 0.243705, 0.248342])
    plt.figure(1).axes[8].spines['right'].set_visible(False)
    plt.figure(1).axes[8].spines['top'].set_visible(False)
    #% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
