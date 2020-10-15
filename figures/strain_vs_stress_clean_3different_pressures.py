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

data1, config1 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
            ], pressure=1)

data2, config2 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
            ], pressure=2)

data3, config3 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
            ], pressure=3)

data = pd.concat([data1, data2, data3])
config = config1

p = fitStiffness(data, config)
config1["fit"] = config["fit"]
config2["fit"] = config["fit"]
config3["fit"] = config["fit"]

plt.subplot(131)
plotStressStrain(data1, config1)
plt.xlim(0, 300)

plt.subplot(132)
plotStressStrain(data2, config2)
plt.xlim(0, 300)

plt.subplot(133)
plotStressStrain(data3, config3)
plt.xlim(0, 300)

print("parameter", p)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 5.460000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.0, 300.0)
plt.figure(1).axes[0].set_ylim(-0.0, 1.0)
plt.figure(1).axes[0].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
plt.figure(1).axes[0].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.093175, 0.212765, 0.256415, 0.693863])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].lines[3].set_markersize(2.0)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.500000, 1.026615])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([0.652341, 0.318166])
plt.figure(1).axes[0].texts[1].set_text("k = 102 Pa\n$\\alpha$ = 0.43")
plt.figure(1).axes[1].set_xlim(0.0, 300.0)
plt.figure(1).axes[1].set_ylim(-0.0, 1.0)
plt.figure(1).axes[1].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
plt.figure(1).axes[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[1].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_position([0.400874, 0.212765, 0.256415, 0.693863])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].lines[3].set_markersize(2.0)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.500000, 1.026615])
plt.figure(1).axes[1].texts[0].set_text("2 bar")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
plt.figure(1).axes[2].set_xlim(0.0, 300.0)
plt.figure(1).axes[2].set_ylim(-0.0, 1.0)
plt.figure(1).axes[2].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
plt.figure(1).axes[2].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[2].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[2].set_position([0.708572, 0.212765, 0.256415, 0.693863])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].lines[3].set_markersize(2.0)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.500000, 1.026615])
plt.figure(1).axes[2].texts[0].set_text("3 bar")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
