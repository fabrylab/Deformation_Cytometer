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

from scripts.helper_functions import load_all_data, fitStiffness
from scripts.helper_functions import plotStressStrain, plotStressStrainFit
import numpy as np
import matplotlib.pyplot as plt

import pylustrator

pylustrator.start()

data3, config3 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_30_alginate2%_NIH3T3_blebbistatin\inlet\[0-9]\*_result.txt"
            ], pressure=3)

data3b, config3b = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_30_alginate2%_NIH3T3_DMSO\inlet\[0-9]\*_result.txt",
            ], pressure=3)

fitStiffness(data3, config3)
fitStiffness(data3b, config3b)

plt.subplot(121)
plotStressStrain(data3b, config3b)
plotStressStrainFit(data3b, config3b)
plt.title("DMSO")

plt.text(0.5, 0.5, f"k = {config3b['fit']['p'][0]:3.0f} Pa\n$\\alpha$ = {config3b['fit']['p'][1]:2.2f}")

plt.subplot(122)
plotStressStrain(data3, config3)
plotStressStrainFit(data3, config3)
plt.title("blebbistation")

plt.text(0.5, 0.5, f"k = {config3['fit']['p'][0]:3.0f} Pa\n$\\alpha$ = {config3['fit']['p'][1]:2.2f}")

print("p3b", config3b["fit"]["p"])
print("p3", config3["fit"]["p"])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 5.530000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.111446, 0.213746, 0.345357, 0.672724])
plt.figure(1).axes[0].set_xlim(0.0, 300.0)
plt.figure(1).axes[0].set_ylim(-0.0, 1.0)
plt.figure(1).axes[0].set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
plt.figure(1).axes[0].set_yticks([0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].lines[3].set_markersize(2.0)
plt.figure(1).axes[0].texts[0].set_position([249.328258, 0.049992])
plt.figure(1).axes[1].set_position([0.559253, 0.213746, 0.345357, 0.672724])
plt.figure(1).axes[1].set_xlim(0.0, 300.0)
plt.figure(1).axes[1].set_ylim(0.0, 1.0)
plt.figure(1).axes[1].set_yticklabels(["", "", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_yticks([0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].lines[3].set_markersize(2.0)
plt.figure(1).axes[1].texts[0].set_position([260.352304, 0.049992])
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
