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
from helper_functions import getInputFile, getConfig, getData
from helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from helper_functions import storeEvaluationResults
import numpy as np

""" loading data """
# get the results file (by config parameter or user input dialog)
datafile = getInputFile(filetype=[("txt file",'*_result.txt')])

# load the data and the config
data = getData(datafile)
config = getConfig(datafile)

""" evaluating data"""

#refetchTimestamps(data, config)

getVelocity(data, config)

# take the mean of all values of each cell
data = data.groupby(['cell_id']).mean()

correctCenter(data, config)

data = filterCells(data, config)

# reset the indices
data.reset_index(drop=True, inplace=True)

getStressStrain(data, config)


def velfit(r):  # for stress versus strain
    p0, p1, p2 = config["vel_fit"]
    R = config["channel_width_m"] / 2 * 1e6
    return p0 * (1 - np.abs((r) / R) ** p1)

def getVelGrad(r):
    p0, p1, p2 = config["vel_fit"]
    r = r * 1e-6
    p0 = p0 * 1e-3
    r0 = 100e-6
    return - (p1 * p0 * (np.abs(r) / r0) ** p1) / r

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import glob

v = velfit(data.rp)
w = np.abs(getVelGrad(data.rp))/(2*np.pi)/4

def fitfunc(x, p0, p1, p2):  # for stress versus strain
    k = p0#(p0 + x)
    alpha = p1
    #return x / (k * (np.abs(w)) ** alpha)
    return x / (k * (np.abs(w) + (v/(np.pi*2*config["imaging_pos_mm"]))) ** alpha) + p2

sigma = data.stress
strain = data.strain

def fitStiffness(data, config):

    #k
    #alpha

    #y = sigma / (k*np.abs(w)**alpha)


#        return (1 / p0) * np.log((x / p1) + 1) + 0.05#p2

    pstart = (50, 0.01, 0)  # initial guess
    pstart = (120, 0.3, 0)  # initial guess

    xy = np.vstack([data.stress, data.strain])
    kd = gaussian_kde(xy)(xy)

    # fit weighted by the density of points
    p, pcov = curve_fit(fitfunc, data.stress, data.strain, pstart, maxfev=10000)
    print("fit", p, pcov)
    return p

p = fitStiffness(data, config)
#fitStiffness(data, config)

plotStressStrain(data, config)

plt.plot(sigma, fitfunc(sigma, p[0], p[1],p[2]), ".")
plt.plot(sigma, fitfunc(sigma, 120, 0.3), ".")

#w = getVelGrad(data.rp)/(2*np.pi)/5
sigma = data.stress
strain = data.strain

#k
#alpha

#y = sigma / (k*np.abs(w)**alpha)


import numpy as np
x = np.arange(np.min(sigma), np.max(sigma))
plt.plot(x, fitfunc(x, *p), "-k")

plt.show()

if 0:
    """ plotting data """

    initPlotSettings()

    # add multipage plotting
    pp = PdfPages(datafile[:-11] + '_new.pdf')

    # generate the velocity profile plot
    plotVelocityProfile(data, config)
    pp.savefig()

    # generate the stress strain plot
    plotStressStrain(data, config)
    pp.savefig()

    # generate the info page with the data
    plotMessurementStatus(data, config)

    pp.savefig()
    #plt.show()
    pp.close()

    # store the evaluation data in a file
    #storeEvaluationResults(data, config)
