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

percentiles = np.quantile(data.area, [0.33, 0.66])
for i in range(3):
    if i == 0:
        d = data[(data.area < percentiles[0])]
        d1 = data1[(data1.area < percentiles[0])]
        d2 = data2[(data2.area < percentiles[0])]
        d3 = data3[(data3.area < percentiles[0])]
    elif i == 1:
        d = data[(data.area > percentiles[0]) & (data.area < percentiles[1])]
        d1 = data1[(data1.area > percentiles[0]) & (data1.area < percentiles[1])]
        d2 = data2[(data2.area > percentiles[0]) & (data2.area < percentiles[1])]
        d3 = data3[(data3.area > percentiles[0]) & (data3.area < percentiles[1])]
    else:
        d = data[(data.area > percentiles[1])]
        d1 = data1[(data1.area > percentiles[1])]
        d2 = data2[(data2.area > percentiles[1])]
        d3 = data3[(data3.area > percentiles[1])]
    print(d.area.min(), d.area.max(), percentiles)

    p = fitStiffness(d, config)
    config1["fit"] = config["fit"]
    config2["fit"] = config["fit"]
    config3["fit"] = config["fit"]
    print("->", config["fit"])

    plt.subplot(3, 3, i*3+1)
    plotStressStrain(d1, config1, skip=1)
    plt.xlim(0, 300)

    plt.subplot(3, 3, i*3+2)
    plotStressStrain(d2, config2, skip=1)
    plt.xlim(0, 300)

    plt.subplot(3, 3, i*3+3)
    plotStressStrain(d3, config3, skip=1)
    plt.xlim(0, 300)

print("parameter", p)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_xlim(0.0, 300.0)
plt.figure(1).axes[0].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[0].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.093429, 0.705331, 0.248672, 0.248055])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_fontsize(10)
plt.figure(1).axes[0].texts[0].set_ha("left")
plt.figure(1).axes[0].texts[0].set_position([0.470104, 0.133853])
plt.figure(1).axes[0].texts[0].set_text("k =  98.29 Pa\nalpha = 0.43\noffset = 0.044")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].set_xlim(0.0, 300.0)
plt.figure(1).axes[1].set_ylim(-0.2, 1.0)
plt.figure(1).axes[1].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[1].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[1].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_position([0.391835, 0.705331, 0.248672, 0.248055])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
plt.figure(1).axes[2].set_xlim(0.0, 300.0)
plt.figure(1).axes[2].set_ylim(-0.2, 1.0)
plt.figure(1).axes[2].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[2].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[2].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[2].set_position([0.690242, 0.705331, 0.248672, 0.248055])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].yaxis.labelpad = -11.374026
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([1.020968, 0.424395])
plt.figure(1).axes[2].texts[0].set_rotation(90.0)
plt.figure(1).axes[2].texts[0].set_text("small")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
plt.figure(1).axes[3].set_xlim(0.0, 300.0)
plt.figure(1).axes[3].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[3].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[3].set_position([0.093429, 0.407665, 0.248672, 0.248055])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([0.470104, 0.119793])
plt.figure(1).axes[3].texts[0].set_text("k =  97.43 Pa\nalpha = 0.44\noffset = 0.059")
plt.figure(1).axes[3].get_xaxis().get_label().set_text("")
plt.figure(1).axes[4].set_xlim(0.0, 300.0)
plt.figure(1).axes[4].set_ylim(-0.2, 1.0)
plt.figure(1).axes[4].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[4].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[4].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[4].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[4].set_position([0.391835, 0.407665, 0.248672, 0.248055])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].get_xaxis().get_label().set_text("")
plt.figure(1).axes[4].get_yaxis().get_label().set_text("")
plt.figure(1).axes[5].set_xlim(0.0, 300.0)
plt.figure(1).axes[5].set_ylim(-0.2, 1.0)
plt.figure(1).axes[5].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[5].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[5].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[5].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[5].set_position([0.690241, 0.407665, 0.248672, 0.248055])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_ha("center")
plt.figure(1).axes[5].texts[0].set_position([1.020968, 0.376732])
plt.figure(1).axes[5].texts[0].set_rotation(90.0)
plt.figure(1).axes[5].texts[0].set_text("medium")
plt.figure(1).axes[5].get_xaxis().get_label().set_text("")
plt.figure(1).axes[5].get_yaxis().get_label().set_text("")
plt.figure(1).axes[6].set_xlim(0.0, 300.0)
plt.figure(1).axes[6].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[6].set_xticklabels(["0", "100", "200"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[6].set_position([0.093429, 0.110000, 0.248672, 0.248055])
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].texts[0].set_position([0.470104, 0.137075])
plt.figure(1).axes[6].texts[0].set_text("k = 103.27 Pa\nalpha = 0.43\noffset = 0.086")
plt.figure(1).axes[7].set_xlim(0.0, 300.0)
plt.figure(1).axes[7].set_ylim(-0.2, 1.0)
plt.figure(1).axes[7].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[7].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[7].set_xticklabels(["0", "100", "200"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[7].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[7].set_position([0.391835, 0.110000, 0.248672, 0.248055])
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[7].get_yaxis().get_label().set_text("")
plt.figure(1).axes[8].set_xlim(0.0, 300.0)
plt.figure(1).axes[8].set_ylim(-0.2, 1.0)
plt.figure(1).axes[8].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[8].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[8].set_xticklabels(["0", "100", "200"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[8].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[8].set_position([0.690241, 0.110000, 0.248672, 0.248055])
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[8].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[8].transAxes)  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[8].texts[0].set_ha("center")
plt.figure(1).axes[8].texts[0].set_position([1.020968, 0.428833])
plt.figure(1).axes[8].texts[0].set_rotation(90.0)
plt.figure(1).axes[8].texts[0].set_text("large")
plt.figure(1).axes[8].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
