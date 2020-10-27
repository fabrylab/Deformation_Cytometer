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
from matplotlib.backends.backend_pdf import PdfPages
from scripts.helper_functions import getInputFile, getConfig, getData
from scripts.helper_functions import getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from scripts.helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from scripts.helper_functions import storeEvaluationResults

""" loading data """
# get the results file (by config parameter or user input dialog)
datafile = getInputFile(filetype=[("txt file",'*_result.txt')])

# load the data and the config
data = getData(datafile)
config = getConfig(datafile)

""" evaluating data"""

getVelocity(data, config)

data = filterCells(data, config)

correctCenter(data, config)

getStressStrain(data, config)

fitStiffness(data, config)

""" plotting data """

initPlotSettings()

# add multipage plotting
pp = PdfPages(datafile[:-11] + '.pdf')

# generate the velocity profile plot
plotVelocityProfile(data, config)
pp.savefig()

# generate the stress strain plot
plotStressStrain(data, config)
pp.savefig()

# generate the info page with the data
plotMessurementStatus(data, config)
pp.savefig()
pp.close()

# store the evaluation data in a file
storeEvaluationResults(data, config)
