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
from helper_functions import getInputFile, getConfig, getData, getInputFolder
from helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pylustrator
pylustrator.start()

pylustrator.load("figure_stress_vs_radialposition.py")
pylustrator.load("figure_margination_histogram.py", offset=[0, 1])
pylustrator.load("figure_angle_and_size.py", offset=[0, 1])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.250000/2.54, 6.420000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.096051, 0.498363, 0.185083, 0.484703])
plt.figure(1).axes[0].set_xlim(-90.0, 90.0)
plt.figure(1).axes[0].set_xticklabels(["", "", ""])
plt.figure(1).axes[0].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[0].texts[1].set_position([-0.374725, 0.955913])
plt.figure(1).axes[0].get_xaxis().get_label().set_text("")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("strain")
plt.figure(1).axes[1].set_position([0.302915, 0.498363, 0.185083, 0.484703])
plt.figure(1).axes[1].set_xlim(-90.0, 90.0)
plt.figure(1).axes[1].set_xticklabels(["", "", ""])
plt.figure(1).axes[1].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[1].texts[1].set_position([-0.186918, 0.950514])
plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
plt.figure(1).axes[2].set_position([0.509781, 0.498363, 0.185083, 0.484703])
plt.figure(1).axes[2].set_xlim(-90.0, 90.0)
plt.figure(1).axes[2].set_xticklabels(["", "", ""])
plt.figure(1).axes[2].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[2].texts[1].set_position([-0.161016, 0.950514])
plt.figure(1).axes[2].get_xaxis().get_label().set_text("")
plt.figure(1).axes[3].set_position([0.096051, 0.194818, 0.185083, 0.248456])
plt.figure(1).axes[3].set_ylim(0.0, 0.011)
plt.figure(1).axes[3].yaxis.labelpad = -5.854955
plt.figure(1).axes[3].texts[0].set_visible(False)
plt.figure(1).axes[3].texts[1].set_position([-0.374725, 1.077305])
plt.figure(1).axes[3].texts[1].set_text("e")
plt.figure(1).axes[4].set_position([0.302915, 0.194818, 0.185083, 0.248456])
plt.figure(1).axes[4].set_ylim(0.0, 0.011)
plt.figure(1).axes[4].texts[0].set_visible(False)
plt.figure(1).axes[4].texts[1].set_position([-0.186918, 0.959572])
plt.figure(1).axes[4].texts[1].set_text("f")
plt.figure(1).axes[5].set_position([0.509780, 0.192011, 0.185083, 0.248456])
plt.figure(1).axes[5].set_ylim(0.0, 0.011)
plt.figure(1).axes[5].texts[0].set_visible(False)
plt.figure(1).axes[5].texts[1].set_position([-0.161009, 0.970868])
plt.figure(1).axes[5].texts[1].set_text("g")
plt.figure(1).axes[6].set_position([0.791249, 0.498363, 0.185083, 0.481965])
plt.figure(1).axes[6].set_xlim(-90.0, 90.0)
plt.figure(1).axes[6].set_xticklabels(["", "", ""])
plt.figure(1).axes[6].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[6].texts[0].set_text("d")
plt.figure(1).axes[6].get_xaxis().get_label().set_text("")
plt.figure(1).axes[7].set_position([0.791249, 0.192011, 0.185083, 0.248456])
plt.figure(1).axes[7].set_ylim(0.0, 30.0)
plt.figure(1).axes[7].set_yticklabels(["0", "10", "20", "30"])
plt.figure(1).axes[7].set_yticks([0.0, 10.0, 20.0, 30.0])
plt.figure(1).axes[7].texts[0].set_position([-0.232258, 1.192373])
plt.figure(1).axes[7].texts[0].set_text("h")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


