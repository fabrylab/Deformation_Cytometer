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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_functions import plotDensityScatter, load_all_data
import numpy as np
import pylustrator
pylustrator.start()

for index, pressure in enumerate([0.5, 1, 1.5]):
    ax = plt.subplot(3, 3, index+1)
    data, config = load_all_data(rf"\\131.188.117.96\biophysDS\meroles\2020.05.27_THP1_RPMI_2pc_Ag\THP1_27_05_2020_2replicate\2\*_result.txt", pressure=pressure)
    plotDensityScatter(data.stress, data.strain)

for index, pressure in enumerate([1, 2, 3]):
    ax = plt.subplot(3, 3, 3+index+1)
    data, config = load_all_data([
            rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt",
            rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",
        ], pressure=pressure)
    plotDensityScatter(data.stress, data.strain)

for index, pressure in enumerate([1, 2, 3]):
    ax = plt.subplot(3, 3, 6+index+1)
    data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\2\*_result.txt", pressure=pressure)
    plotDensityScatter(data.stress, data.strain)


#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 12.000000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.115537, 0.734373, 0.268221, 0.256605])
plt.figure(1).axes[0].set_xlim(0.0, 260.0)
plt.figure(1).axes[0].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[0].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[0].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.877105, 0.057061])
plt.figure(1).axes[0].texts[0].set_text("0.5 bar")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("THP1\nstrain")
plt.figure(1).axes[1].set_position([0.415447, 0.734373, 0.268221, 0.256605])
plt.figure(1).axes[1].set_xlim(0.0, 260.0)
plt.figure(1).axes[1].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[1].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[1].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[1].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[1].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.921815, 0.057061])
plt.figure(1).axes[1].texts[0].set_text("1 bar")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
plt.figure(1).axes[2].set_position([0.715360, 0.734373, 0.268221, 0.256605])
plt.figure(1).axes[2].set_xlim(0.0, 260.0)
plt.figure(1).axes[2].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[2].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[2].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[2].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[2].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].xaxis.labelpad = -4.000000
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.883353, 0.057061])
plt.figure(1).axes[2].texts[0].set_text("1.5 bar")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("")
plt.figure(1).axes[3].set_position([0.115537, 0.427256, 0.268221, 0.256605])
plt.figure(1).axes[3].set_xlim(0.0, 260.0)
plt.figure(1).axes[3].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[3].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[3].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[3].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_ha("center")
plt.figure(1).axes[3].texts[0].set_position([0.911246, 0.065714])
plt.figure(1).axes[3].texts[0].set_text("1 bar")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("K562\nstrain")
plt.figure(1).axes[4].set_position([0.415447, 0.427256, 0.268221, 0.256605])
plt.figure(1).axes[4].set_xlim(0.0, 260.0)
plt.figure(1).axes[4].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[4].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[4].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[4].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[4].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[4].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_ha("center")
plt.figure(1).axes[4].texts[0].set_position([0.921815, 0.065714])
plt.figure(1).axes[4].texts[0].set_text("2 bar")
plt.figure(1).axes[5].set_position([0.715360, 0.427256, 0.268221, 0.256605])
plt.figure(1).axes[5].set_xlim(0.0, 260.0)
plt.figure(1).axes[5].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[5].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[5].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[5].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[5].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[5].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_ha("center")
plt.figure(1).axes[5].texts[0].set_position([0.883353, 0.065714])
plt.figure(1).axes[5].texts[0].set_text("3 bar")
plt.figure(1).axes[6].set_position([0.115537, 0.120139, 0.268221, 0.256605])
plt.figure(1).axes[6].set_xlim(0.0, 260.0)
plt.figure(1).axes[6].set_xticklabels(["50", "150", "250", ""], minor=True)
plt.figure(1).axes[6].set_xticks([50.0, 150.0, 250.0, np.nan], minor=True)
plt.figure(1).axes[6].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].texts[0].set_ha("center")
plt.figure(1).axes[6].texts[0].set_position([0.911246, 0.054184])
plt.figure(1).axes[6].texts[0].set_text("1 bar")
plt.figure(1).axes[6].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[6].get_yaxis().get_label().set_text("NIH 3T3\nstrain")
plt.figure(1).axes[7].set_position([0.415447, 0.120139, 0.268221, 0.256605])
plt.figure(1).axes[7].set_xlim(0.0, 260.0)
plt.figure(1).axes[7].set_xticklabels(["50", "150", "250", ""], minor=True)
plt.figure(1).axes[7].set_xticks([50.0, 150.0, 250.0, np.nan], minor=True)
plt.figure(1).axes[7].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[7].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[7].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[7].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[7].transAxes)  # id=plt.figure(1).axes[7].texts[0].new
plt.figure(1).axes[7].texts[0].set_ha("center")
plt.figure(1).axes[7].texts[0].set_position([0.921815, 0.054184])
plt.figure(1).axes[7].texts[0].set_text("2 bar")
plt.figure(1).axes[7].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[8].set_position([0.715360, 0.120139, 0.268221, 0.256605])
plt.figure(1).axes[8].set_xlim(0.0, 260.0)
plt.figure(1).axes[8].set_xticklabels(["50", "150", "250", ""], minor=True)
plt.figure(1).axes[8].set_xticks([50.0, 150.0, 250.0, np.nan], minor=True)
plt.figure(1).axes[8].set_ylim(-0.04887777819543629, 1.2)
plt.figure(1).axes[8].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[8].set_yticks([0.0, 0.5, 1.0])
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[8].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[8].transAxes)  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[8].texts[0].set_ha("center")
plt.figure(1).axes[8].texts[0].set_position([0.883353, 0.054184])
plt.figure(1).axes[8].texts[0].set_text("3 bar")
plt.figure(1).axes[8].get_xaxis().get_label().set_text("stress (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


