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
from scripts.helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
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

#pylustrator.start()

global_im = None
global_index = 0

def densityPlot(x, y, cmap, alpha=0.5):
    global global_im, global_index
    from scipy.stats import kde

    ax = plt.gca()

    # Thus we can cut the plotting window in several hexbins
    nbins = np.max(x) / 10
    ybins = 20

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(np.vstack([x, y]))
    if 0:
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():ybins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # plot a density
        ax.set_title('Calculate Gaussian KDE')
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', alpha=alpha, cmap=cmap)
    else:
        xi, yi = np.meshgrid(np.linspace(-10, 300, 200),
                             np.linspace(0, 1, 80))  # np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():ybins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        im = zi.reshape(xi.shape)
        if 0:
            if global_im is None:
                global_im = np.zeros((im.shape[0], im.shape[1], 3), dtype="uint8")
            if 1:  # global_index == 1:
                print("_____", im.min(), im.max())
                im -= np.percentile(im, 10)
                global_im[:, :, global_index] = im / im.max() * 255
                print("_____", global_im[:, :, global_index].min(), global_im[:, :, global_index].max())
            print("COLOR", global_index)
            global_index += 1
            if global_index == 3:
                print(global_im.shape, global_im.dtype)
                plt.imshow(global_im[::-1], extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)], aspect="auto")
        else:
            if global_im is None:
                global_im = []
            im -= im.min()
            im /= im.max()
            global_im.append(plt.get_cmap(cmap)(im ** 0.5))
            global_im[-1][:, :, 3] = im
            plt.imshow(global_im[-1][::-1], vmin=0, vmax=1, extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
                       aspect="auto")
            global_index += 1
            if global_index == 3:
                print("COLOR", global_im[0].shape, global_im[0].min(), global_im[0].max())
                im = global_im[0] + global_im[1] + global_im[2] - 2
                # im[im<0] = 0
                # im[im>255] = 255
                print("COLOR", im.shape, im.min(), im.max())
                # plt.imshow(im[::-1], vmin=0, vmax=1, extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)], aspect="auto")

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

v1 = data1.velocity_fitted#velfit(config1, data1.rp)
w1 = omega(data1.velocity_gradient)#omega(np.abs(getVelGrad(config1, data1.rp))/(2*np.pi))
v2 = data2.velocity_fitted#velfit(config2, data2.rp)
w2 = omega(data2.velocity_gradient)#omega(np.abs(getVelGrad(config2, data2.rp))/(2*np.pi))
v3 = data3.velocity_fitted#velfit(config3, data3.rp)
w3 = omega(data3.velocity_gradient)#omega(np.abs(getVelGrad(config3, data3.rp))/(2*np.pi))

v1 = velfit(config1, data1.rp)
w1 = omega(np.abs(getVelGrad(config1, data1.rp))/(2*np.pi))
v2 = velfit(config2, data2.rp)
w2 = omega(np.abs(getVelGrad(config2, data2.rp))/(2*np.pi))
v3 = velfit(config3, data3.rp)
w3 = omega(np.abs(getVelGrad(config3, data3.rp))/(2*np.pi))
if 0:
    plt.figure(2)
    plt.subplot(131)
    plt.plot(data2.rp, data2.velocity_gradient, "o", ms=1)
    plt.subplot(132)
    plt.plot(data2.rp, data2.velocity, "o", ms=1)
    plt.subplot(133)
    plt.plot(data2.rp, data2.velocity_fitted, "o", ms=1)
    #plt.plot(data2.rp, data2.velocity_fitted, "o")
    #plt.plot(data3.rp, data3.velocity_fitted, "o")

    plt.show()
    raise

v = np.hstack((v1, v2, v3))
w = np.hstack((w1, w2, w3))

indices = None
def fitfunc(x, p0, p1, p2):  # for stress versus strain
    k = p0#(p0 + x)
    alpha = p1
    #return x / (k * (np.abs(w)) ** alpha)
    #return np.log(x/k + 1) / ((np.abs(w) + (v/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2
    return ( x/(k * (np.abs(w[indices]) + (v[indices]/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2 )# * (x < 100)

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

indices = data.stress < 9999999
p = fitStiffness(data, config)
p1 = p
p2 = p
p3 = p
if 0:
    ps = []
    ii = np.arange(10, 300, 50)
    for i in ii:
        indices = data.stress < i
        p = fitStiffness(data[data.stress < i], config)
        ps.append(p)
        print(i, p)
    p1 = p
    p2 = p
    p3 = p

    ps = np.array(ps)
    plt.subplot(131)
    plt.plot(ii, ps[:, 0])
    plt.subplot(132)
    plt.plot(ii, ps[:, 1])
    plt.subplot(133)
    plt.plot(ii, ps[:, 2])
    plt.show()
    #raise

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


global_im = None
global_index = 0
cmaps = ["", "Blues", "Greens", "Reds"]
cmaps = ["", "Greens", "Blues", "Purples"]
datas = [0, data1, data2, data3]
fitfuncs = [0, fitfunc1, fitfunc2, fitfunc3]
ps = [0, p1, p2, p3]
for i in [3, 2, 1]:
    densityPlot(datas[i].stress, datas[i].strain, cmaps[i])
    plt.plot([], [], "o", color=plt.get_cmap(cmaps[i])(0.75), label=f"{i} bar")


    #f = plotPathList(paths[pressures==pressure], cmaps[index], 0.5 if index > 1 else 1)
    x = datas[i].stress
    y = fitfuncs[i](datas[i].stress, ps[i][0], ps[i][1], ps[i][2])
    indices = np.argsort(x)
    plt.plot(x[indices], y[indices], "k-")

print("parameter", p)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(5.800000/2.54, 5.430000/2.54, forward=True)
plt.figure(1).axes[0].legend(frameon=False, handletextpad=0.0, markerscale=0.7, fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[0].set_position([0.245870, 0.230186, 0.666410, 0.684177])
plt.figure(1).axes[0].set_xlim(0.0, 300.0)
plt.figure(1).axes[0].set_xticklabels(["0", "50", "100", "150", "200", "250", "300", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_xticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, np.nan])
plt.figure(1).axes[0].set_ylim(-0.0, 1.0)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend()._set_loc((0.677411, 0.100108))
plt.figure(1).axes[0].get_legend()._set_loc((0.606658, 0.043713))
plt.figure(1).axes[0].lines[1].set_color("#00000080")
plt.figure(1).axes[0].lines[3].set_color("#00000080")
plt.figure(1).axes[0].lines[3].set_markersize(2.0)
plt.figure(1).axes[0].lines[5].set_color("#00000080")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.500000, 1.026615])
plt.figure(1).axes[0].texts[0].set_text("")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("fluid shear stress $\\sigma$ (Pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("cell strain $\\epsilon$")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
