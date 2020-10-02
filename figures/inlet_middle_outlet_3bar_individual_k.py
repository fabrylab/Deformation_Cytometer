# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from helper_functions import getInputFile, getConfig, getData
from helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from helper_functions import storeEvaluationResults, load_all_data
import numpy as np

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import glob
import pylustrator
#pylustrator.start()

pressure = 3
data1, config1 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
            ], pressure=pressure)

data2, config2 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\middle\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\middle\[0-9]\*_result.txt",
            ], pressure=pressure)

data3, config3 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\outlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\outlet\[0-9]\*_result.txt",
            ], pressure=pressure)

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


#p = fitStiffness(data, config)
#p1 = p
#p2 = p
#p3 = p
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

if 0:
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

    plt.show()


from helper_functions import  plotDensityScatter, plotBinnedData
if 1:
    plt.subplot(131)
    k = data1.stress / ( (data1.strain - p1[2]) * (np.abs(w1) + (v1 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** p1[1])
    #print(len(k), len(data1.rp))
    plt.hist(data1.rp, bins=np.linspace(-100, 100, 20), density=True, alpha=0.8)

    #plt.xlim(0, 100)
    #plt.ylim(0, 200)

    plt.subplot(132)
    k = data2.stress / ((data2.strain - p2[2]) * (np.abs(w2) + (v2 / (np.pi * 2 * config2["imaging_pos_mm"]))) ** p2[1])
    plt.hist(data2.rp, bins=np.linspace(-100, 100, 20), density=True, alpha=0.8)

    #plt.xlim(0, 100)
    #plt.ylim(0, 200)

    plt.subplot(133)
    k = data3.stress / ((data3.strain - p3[2]) * (np.abs(w3) + (v3 / (np.pi * 2 * config3["imaging_pos_mm"]))) ** p3[1])

    plt.hist(data3.rp, bins=np.linspace(-100, 100, 20), density=True, alpha=0.8)


if 1:
    plt.subplot(131)
    k = data1.stress / ( (data1.strain - p1[2]) * (np.abs(w1) + (v1 / (np.pi * 2 * config1["imaging_pos_mm"]))) ** p1[1])
    #print(len(k), len(data1.rp))
    plotDensityScatter(np.abs(data1.rp), k)
    plotBinnedData(np.abs(data1.rp), k, np.arange(0, 90, 10))

    plt.xlim(0, 100)
    plt.ylim(0, 200)

    plt.subplot(132)
    k = data2.stress / ((data2.strain - p2[2]) * (np.abs(w2) + (v2 / (np.pi * 2 * config2["imaging_pos_mm"]))) ** p2[1])
    plotDensityScatter(np.abs(data2.rp), k)
    plotBinnedData(np.abs(data2.rp), k, np.arange(0, 90, 10))

    plt.xlim(0, 100)
    plt.ylim(0, 200)

    plt.subplot(133)
    k = data3.stress / ((data3.strain - p3[2]) * (np.abs(w3) + (v3 / (np.pi * 2 * config3["imaging_pos_mm"]))) ** p3[1])

    plotDensityScatter(np.abs(data3.rp), k)
    plotBinnedData(np.abs(data3.rp), k, np.arange(0, 90, 10))


    plt.xlim(0, 100)
    plt.ylim(0, 200)

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(16.250000/2.54, 5.000000/2.54, forward=True)
    plt.figure(1).axes[0].grid(True)
    plt.figure(1).axes[0].set_position([0.124970, 0.226507, 0.245158, 0.648012])
    plt.figure(1).axes[0].set_ylim(50.0, 150.0)
    plt.figure(1).axes[0].set_yticklabels(["50", "75", "100", "125", "150", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
    plt.figure(1).axes[0].set_yticks([50.0, 75.0, 100.0, 125.0, 150.0, np.nan])
    plt.figure(1).axes[0].spines['right'].set_visible(False)
    plt.figure(1).axes[0].spines['top'].set_visible(False)
    plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
    plt.figure(1).axes[0].texts[0].set_ha("center")
    plt.figure(1).axes[0].texts[0].set_position([0.438800, 1.046685])
    plt.figure(1).axes[0].texts[0].set_text("inlet")
    plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
    plt.figure(1).axes[0].texts[1].set_position([-0.343920, 1.046685])
    plt.figure(1).axes[0].texts[1].set_text("a")
    plt.figure(1).axes[0].texts[1].set_weight("bold")
    plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
    plt.figure(1).axes[0].get_yaxis().get_label().set_text("individual stiffness")
    plt.figure(1).axes[1].grid(True)
    plt.figure(1).axes[1].set_position([0.419159, 0.226507, 0.245158, 0.648012])
    plt.figure(1).axes[1].set_ylim(50.0, 150.0)
    plt.figure(1).axes[1].set_yticklabels(["50", "75", "100", "125", "150", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
    plt.figure(1).axes[1].set_yticks([50.0, 75.0, 100.0, 125.0, 150.0, np.nan])
    plt.figure(1).axes[1].spines['right'].set_visible(False)
    plt.figure(1).axes[1].spines['top'].set_visible(False)
    plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
    plt.figure(1).axes[1].texts[0].set_ha("center")
    plt.figure(1).axes[1].texts[0].set_position([0.490337, 1.046685])
    plt.figure(1).axes[1].texts[0].set_text("middle")
    plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[1].new
    plt.figure(1).axes[1].texts[1].set_position([-0.099119, 1.046685])
    plt.figure(1).axes[1].texts[1].set_text("b")
    plt.figure(1).axes[1].texts[1].set_weight("bold")
    plt.figure(1).axes[1].get_xaxis().get_label().set_text("radial position (µm)")
    plt.figure(1).axes[2].grid(True)
    plt.figure(1).axes[2].set_position([0.713349, 0.226507, 0.245158, 0.648012])
    plt.figure(1).axes[2].set_ylim(50.0, 150.0)
    plt.figure(1).axes[2].set_yticklabels(["50", "75", "100", "125", "150", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
    plt.figure(1).axes[2].set_yticks([50.0, 75.0, 100.0, 125.0, 150.0, np.nan])
    plt.figure(1).axes[2].spines['right'].set_visible(False)
    plt.figure(1).axes[2].spines['top'].set_visible(False)
    plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
    plt.figure(1).axes[2].texts[0].set_ha("center")
    plt.figure(1).axes[2].texts[0].set_position([0.493558, 1.046685])
    plt.figure(1).axes[2].texts[0].set_text("outlet")
    plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[1].new
    plt.figure(1).axes[2].texts[1].set_position([-0.102340, 1.046685])
    plt.figure(1).axes[2].texts[1].set_text("c")
    plt.figure(1).axes[2].texts[1].set_weight("bold")
    plt.figure(1).axes[2].get_xaxis().get_label().set_text("radial position (µm)")
    #% end: automatic generated code from pylustrator
    plt.savefig(__file__[:-3] + ".png", dpi=300)
    plt.savefig(__file__[:-3] + ".pdf")
    plt.show()