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
from helper_functions import plotDensityScatter, load_all_data
import pylustrator
pylustrator.start()

for index, pressure in enumerate([1, 2, 3]):
    ax = plt.subplot(1, 3, index+1)

    data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data" +
                                 r"\microscope4\2020_may\2020_05_22_alginateDMEM2%\2\*_result.txt", pressure=pressure)

    plotDensityScatter(data.stress, data.strain)

#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 4.390000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.115765, 0.257929, 0.268860, 0.717020])
plt.figure(1).axes[0].set_xlim(0.0, 260.0)
plt.figure(1).axes[0].set_xticklabels(["0", "50", "100", "150", "200", "250"])
plt.figure(1).axes[0].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])
plt.figure(1).axes[0].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.509148, 0.891045])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("NIH 3T3\nstrain")
plt.figure(1).axes[1].set_position([0.416266, 0.257929, 0.268860, 0.717020])
plt.figure(1).axes[1].set_xlim(0.0, 260.0)
plt.figure(1).axes[1].set_xticklabels(["0", "50", "100", "150", "200", "250"])
plt.figure(1).axes[1].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])
plt.figure(1).axes[1].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[1].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[1].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[1].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.545742, 0.891045])
plt.figure(1).axes[1].texts[0].set_text("2 bar")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[2].set_position([0.716769, 0.257929, 0.268860, 0.717020])
plt.figure(1).axes[2].set_xlim(0.0, 260.0)
plt.figure(1).axes[2].set_xticklabels(["0", "50", "100", "150", "200", "250"])
plt.figure(1).axes[2].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])
plt.figure(1).axes[2].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[2].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[2].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[2].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.504574, 0.891045])
plt.figure(1).axes[2].texts[0].set_text("3 bar")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("stress (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()


