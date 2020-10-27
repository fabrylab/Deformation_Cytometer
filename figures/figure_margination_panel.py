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
import numpy as np

import pylustrator
pylustrator.start()

pylustrator.load("figure_stress_vs_radialposition.py")
#pylustrator.load("figure_margination_histogram.py", offset=[0, 1])
pylustrator.load("strain_vs_stress_clean_3different_pressures.py", offset=[0, 1])
#pylustrator.load("figure_angle_and_size.py", offset=[0, 1])


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_ylim(-0.04991886291525161, 1.2)
plt.figure(1).axes[0].get_yaxis().get_label().set_text("cell strain $\epsilon$")
plt.figure(1).axes[1].set_ylim(-0.04991886291525161, 1.2)
plt.figure(1).axes[2].set_ylim(-0.04991886291525161, 1.2)
plt.figure(1).axes[3].set_xlim(0.0, 270.0)
plt.figure(1).axes[3].set_ylim(0.0, 1.2)
plt.figure(1).axes[3].set_xticks([0.0, 100.0, np.nan, 200.0, np.nan])
plt.figure(1).axes[3].set_xticklabels(["", "", ""], minor=True)
plt.figure(1).axes[3].set_position([0.115765, 0.156698, 0.268860, 0.345859])
plt.figure(1).axes[3].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[3].texts[0].set_visible(False)
plt.figure(1).axes[3].texts[1].set_position([0.617472, 0.311448])
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[2].new
plt.figure(1).axes[3].texts[2].set_position([-0.232258, 0.970685])
plt.figure(1).axes[3].texts[2].set_text("d")
plt.figure(1).axes[3].texts[2].set_weight("bold")
plt.figure(1).axes[4].set_xlim(0.0, 270.0)
plt.figure(1).axes[4].set_ylim(0.0, 1.2)
plt.figure(1).axes[4].set_xticks([0.0, 100.0, np.nan, 200.0, np.nan])
plt.figure(1).axes[4].set_xticklabels(["", "", ""], minor=True)
plt.figure(1).axes[4].set_position([0.416266, 0.156698, 0.268860, 0.345859])
plt.figure(1).axes[4].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[4].texts[0].set_visible(False)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[1].new
plt.figure(1).axes[4].texts[1].set_position([-0.127650, 0.970685])
plt.figure(1).axes[4].texts[1].set_text("e")
plt.figure(1).axes[4].texts[1].set_weight("bold")
plt.figure(1).axes[5].set_xlim(0.0, 270.0)
plt.figure(1).axes[5].set_ylim(0.0, 1.2)
plt.figure(1).axes[5].set_xticks([0.0, 100.0, np.nan, 200.0, np.nan])
plt.figure(1).axes[5].set_xticklabels(["", "", ""], minor=True)
plt.figure(1).axes[5].set_position([0.716769, 0.156698, 0.268860, 0.345859])
plt.figure(1).axes[5].set_xticks([50.0, 150.0, 250.0], minor=True)
plt.figure(1).axes[5].texts[0].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[1].new
plt.figure(1).axes[5].texts[1].set_position([-0.110215, 0.970685])
plt.figure(1).axes[5].texts[1].set_text("f")
plt.figure(1).axes[5].texts[1].set_weight("bold")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


