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
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from deformationcytometer.includes.helper_functions_scripts import getInputFile, getConfig, getData
from deformationcytometer.includes.helper_functions_scripts import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from deformationcytometer.includes.helper_functions_scripts import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from deformationcytometer.includes.helper_functions_scripts import storeEvaluationResults, load_all_data, get_folders

""" loading data """
# get the results file (by config parameter or user input dialog)

out_put_file = r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\out.pdf"


base_folders= [r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\\",
               #r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\\"
               ]
base_folders_ = [os.path.join(b, "[0-9]\*_result.txt") for b in base_folders]
data, config = load_all_data(base_folders_, pressure=None)
print(len(get_folders(base_folders_, pressure=None, repetition=None)))


""" evaluating data"""

p = fitStiffness(data, config)

""" plotting data """

initPlotSettings()

# add multipage plotting
pp = PdfPages(out_put_file)

# generate the velocity profile plot
#plotVelocityProfile(data, config)
#pp.savefig()
#plt.cla()

# generate the stress strain plot
plotStressStrain(data, config)
pp.savefig()

# generate the info page with the data
plotMessurementStatus(data, config)

pp.savefig()
#plt.show()
pp.close()

# store the evaluation data in a file
storeEvaluationResults(data, config)