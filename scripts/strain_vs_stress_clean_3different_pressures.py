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
# r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_13_alginate2%_sync_k562_diff_xpositions\end_[0-9]series\*_result.txt"

    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",

#    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt",
#    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",

#            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt",
            ], pressure=1)

data2, config2 = load_all_data([
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_13_alginate2%_sync_k562_diff_xpositions\end_[0-9]series\*_result.txt"

    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",

#    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt",
#    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",

#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt",
            ], pressure=2)

data3, config3 = load_all_data([
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_13_alginate2%_sync_k562_diff_xpositions\end_[0-9]series\*_result.txt"

    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",

#    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt",
#    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt",
            ], pressure=3)

data = pd.concat([data1, data2, data3])
config = config1

def velfit(config, r):  # for stress versus strain
    p0, p1, p2 = config["vel_fit"]
    R = config["channel_width_m"] / 2 * 1e6
    return p0 * (1 - np.abs((r) / R) ** p1)

def getVelGrad(config, r):
    p0, p1, p2 = config["vel_fit"]
    r = r * 1e-6
    p0 = p0 * 1e-3
    r0 = 100e-6
    return - (p1 * p0 * (np.abs(r) / r0) ** p1) / r

def omega(x, p=52.43707149):
    return 0.25 * x
    #return 0.25 * p * (1 - np.exp(-x / p))

v1 = velfit(config1, data1.rp)
w1 = omega(np.abs(getVelGrad(config1, data1.rp))/(2*np.pi))
v2 = velfit(config2, data2.rp)
w2 = omega(np.abs(getVelGrad(config2, data2.rp))/(2*np.pi))
v3 = velfit(config3, data3.rp)
w3 = omega(np.abs(getVelGrad(config3, data3.rp))/(2*np.pi))
if 0:
    plt.figure(2)
    plt.plot(data1.stress, getVelGrad(config1, data1.rp), "o")
    plt.plot(data2.stress, getVelGrad(config2, data2.rp), "o")
    plt.plot(data3.stress, getVelGrad(config3, data3.rp), "o")

v = np.hstack((v1, v2, v3))
w = np.hstack((w1, w2, w3))

def fitfunc(x, p0, p1, p2):  # for stress versus strain
    k = p0#(p0 + x)
    alpha = p1
    #return x / (k * (np.abs(w)) ** alpha)
    #return np.log(x/k + 1) / ((np.abs(w) + (v/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2
    return ( x/(k * (np.abs(w) + (v/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2 )# * (x < 100)

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

import scipy.optimize
def curve_fit(func, x, y, start, maxfev=None, bounds=None):
    def cost(p):
        return np.mean(np.abs(func(x, *p)-y))

    res = scipy.optimize.minimize(cost, start)#, maxfev=maxfev, bounds=bounds)
    #print(res)
    return res["x"], []


def fitfunc1(x, p0, p1, p2):  # for stress versus strain
    k = p0#(p0 + x)
    alpha = p1
    #return x / (k * (np.abs(w)) ** alpha)
    return x / (k * (np.abs(w1) + (v1 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** alpha) + p2
    #return np.log(x/k + 1) / ((np.abs(w1) + (v1/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2

def fitfunc2(x, p0, p1, p2):  # for stress versus strain
    k = p0#(p0 + x)
    alpha = p1
    #return x / (k * (np.abs(w)) ** alpha)
    return x / (k * (np.abs(w2) + (v2 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** alpha) + p2
    #return np.log(x/k + 1) / ((np.abs(w2) + (v2/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2

def fitfunc3(x, p0, p1, p2):  # for stress versus strain
    k = p0#(p0 + x)
    alpha = p1
    #return x / (k * (np.abs(w)) ** alpha)
    return x / (k * (np.abs(w3) + (v3 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** alpha) + p2
    #return np.log(x/k + 1) / ((np.abs(w3) + (v3/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2



p = fitStiffness(data, config)
p1 = p
p2 = p
p3 = p

if 0:
    def fitStiffness1(data, config):
        pstart = (120, 0.3, 0)  # initial guess
        # fit weighted by the density of points
        p, pcov = curve_fit(fitfunc1, data.stress, data.strain, pstart, maxfev=10000, bounds=[(0, 0, 0), (500, 1, 1)])
        print("fit", p, pcov)
        return p
    p1 = fitStiffness1(data1, config1)
    print("1", p1)
    def fitStiffness2(data, config):
        pstart = (120, 0.3, 0)  # initial guess
        # fit weighted by the density of points
        p, pcov = curve_fit(fitfunc2, data.stress, data.strain, pstart, maxfev=10000)
        print("fit", p, pcov)
        return p
    p2 = fitStiffness2(data2, config2)
    print("2", p2)
    def fitStiffness3(data, config):
        pstart = (120, 0.3, 0)  # initial guess
        # fit weighted by the density of points
        p, pcov = curve_fit(fitfunc3, data.stress, data.strain, pstart, maxfev=10000)
        print("fit", p, pcov)
        return p
    p3 = fitStiffness3(data3, config3)
    print("3", p3)
    #fitStiffness(data, config)

plt.subplot(131)
plotStressStrain(data1, config1)

plt.plot(data1.stress, fitfunc1(data1.stress, p1[0], p1[1], p1[2]), "C0.")
#plt.plot(data2.stress, fitfunc2(data2.stress, p2[0], p2[1], p2[2]), ".")
#plt.plot(data3.stress, fitfunc3(data3.stress, p3[0], p3[1], p3[2]), ".")
plt.xlim(0, 300)

plt.subplot(132)
plotStressStrain(data2, config2)

#plt.plot(data1.stress, fitfunc1(data1.stress, p1[0], p1[1], p1[2]), ".")
plt.plot(data2.stress, fitfunc2(data2.stress, p2[0], p2[1], p2[2]), "C1.")
#plt.plot(data3.stress, fitfunc3(data3.stress, p3[0], p3[1], p3[2]), ".")
plt.xlim(0, 300)

plt.subplot(133)
plotStressStrain(data3, config3)

#plt.plot(data1.stress, fitfunc1(data1.stress, p1[0], p1[1], p1[2]), ".")
#plt.plot(data2.stress, fitfunc2(data2.stress, p2[0], p2[1], p2[2]), ".")
plt.plot(data3.stress, fitfunc3(data3.stress, p3[0], p3[1], p3[2]), "C2.")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 5.980000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.093290, 0.275336, 0.249687, 0.610991])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.487546, 1.026615])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[1].set_position([0.392915, 0.275336, 0.249687, 0.610991])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.450183, 1.026615])
plt.figure(1).axes[1].texts[0].set_text("2 bar")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
plt.figure(1).axes[2].set_position([0.692540, 0.275336, 0.249687, 0.610991])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.456410, 1.026615])
plt.figure(1).axes[2].texts[0].set_text("3 bar")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("")
#% end: automatic generated code from pylustrator
plt.show()

plt.plot(sigma, fitfunc(sigma, 120, 0.3), ".")

from scripts.helper_functions import  plotDensityScatter, plotBinnedData
if 1:
    plt.subplot(131)
    k = data1.stress / ( (data1.strain - p[2]) * (np.abs(w1) + (v1 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** p[1])
    plotDensityScatter(np.abs(data1.rp), k)
    plotBinnedData(np.abs(data1.rp), k, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])

    plt.xlim(0, 100)
    plt.ylim(-50, 600)

    plt.subplot(132)
    k = data2.stress / ((data2.strain - p[2]) * (np.abs(w2) + (v2 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** p[1])
    plotDensityScatter(np.abs(data2.rp), k)
    plotBinnedData(np.abs(data2.rp), k, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])

    plt.xlim(0, 100)
    plt.ylim(-50, 600)

    plt.subplot(133)
    k = data3.stress / ((data3.strain - p[2]) * (np.abs(w3) + (v3 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** p[1])
    plotDensityScatter(np.abs(data3.rp), k)
    plotBinnedData(np.abs(data3.rp), k, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])


    plt.xlim(0, 100)
    plt.ylim(-50, 600)


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
