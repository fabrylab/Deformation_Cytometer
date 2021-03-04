# -*- coding: utf-8 -*-

#this programm ist to plot all needed informations for one experiment.
#the pressure needs to be set in line 62
#the folder needs to be set in line 21 or 28

import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, all_plots_same_limits, get_cell_properties#, load_all_data_new
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, plot_joint_density, split_axes, load_all_data_new
import numpy as np
import pylustrator
pylustrator.start()

if 0:
    """"""
    #\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\100 nM - Kontrolle\2021_03_01_12_21_37.tif
    #\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\100 nM - Kontrolle\2021_03_01_12_24_36.tif
    #\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\3.162 nM - Kontrolle\2021_03_01_14_20_57.tif
    #\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\316.2 nM - Kontrolle\2021_03_01_11_45_42.tif
    from deformationcytometer.evaluation.helper_functions import correctCenter
    data, config = load_all_data_new(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\100 nM - Kontrolle\2021_03_01_12_21_37_result.txt")
    correctCenter(data, config)
    d = data
    y_pos = d.rp
    vel = d.velocity
    valid_indices = np.isfinite(y_pos) & np.isfinite(vel)

    plt.plot(y_pos, vel, "o")
    plt.show()
    """"""

experiment = {}

#if parent-subfolders exist:
import glob
from pathlib import Path
import natsort

if 1:
    for folder in glob.glob(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\desmin_cells\2020_12_08_desmin_cytoD\*\*\\"):
        folder = Path(folder)
        name = str(folder.parent.name)+"/"+str(folder.name)
        experiment[name] = str(folder)+"\*_result.txt"

if 1:
    experiment = {}
    # if you want a specifiy order, you have to iterate manually over the folders
    for name1 in ["NIH3T3", "vim ko", "vim ko\n + hDesR\n350P#37", "vim ko\n + hDesR\n406W#34", "vim ko\n + hDes\nWT#39"]:
        # remove new lines \n from the filename, as they are only here for the plot
        name1_ = name1.replace("\n", "")
        for name2 in ["Kontrolle", "DMSO", "cytoD"]:
            p = Path(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\desmin_cells\2020_12_08_desmin_cytoD", name1_, name2, "*_result.txt")
            name = name1 + "/" + name2
            experiment[name] = str(p)

#if no parent-subfolders:
if 1:
    experiment = {}
    for folder in natsort.natsorted(glob.glob(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_02_08_NIH3T3_LatB_drugresponse\*\\")):
        folder = Path(folder)
        name = str(folder.name)
        if name != "old": #if a subfolder exists with no data, to avoid errors
            if name!="plots":
                experiment[name] = str(folder) + "\[2-9]\*_result.txt"

if 1:
    experiment = {}
    files = sorted(glob.glob(r"\\131.188.117.119\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_02_24_NIH3T3_LatrunculinB_doseresponse\*"))
    #files = sorted(glob.glob(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\*"))
    for folder in natsort.natsorted(files):
        folder = Path(folder)
        name = str(folder.name).replace(" - ", "/")
        #experiment[name] = str(folder) + "\*_evaluated_new.csv"
        experiment[name] = str(folder) + "\*_result.txt"

import os
outputfolder = Path(os.path.commonprefix(list(experiment.values()))) / "plots"
outputfolder.mkdir(exist_ok=True)

load_all_data = load_all_data_new

def get_mode_stats(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)]
    def get_mode(x):
        kde = stats.gaussian_kde(x)
        return x[np.argmax(kde(x))]
    mode = get_mode(x)
    err = bootstrap_error(x, get_mode, repetitions=2)
    def string(x):
        if x > 1:
            return str(round(x))
        else:
            return str(round(x, 2))
    plt.text(0.5, 1, string(mode)+"$\pm$"+string(err), transform=plt.gca().transAxes, ha="center", va="top")
    return mode, err, len(x)

N = len(experiment)
cols = 2#int(np.sqrt(N))
rows = int(np.ceil(N/cols))

import matplotlib as mpl
mpl.rc("figure.subplot", top=0.95, hspace=0.35, left=0.2, right=0.95)  # defaults top=0.88, hspace=0.2, left=0.125, right=0.9
mpl.rc("figure", figsize=[3.0, 4.8]) # default 6.4, 4.8
ax_k = []
ax_a = []
ax_Gp1 = []
ax_Gp2 = []
# iterate over all times
for index, name in enumerate(experiment.keys()):
    print(experiment[name])
    try:
        data, config = load_all_data(experiment[name], pressure=3) #set pressure to 0.5,1,2,3 or what was measured. for all pressures together, delete ", pressure=x"
    except (FileNotFoundError, ValueError):
        continue
    data = data[~np.isnan(data.tt)]
    print(index, name, len(data), np.sum(~np.isnan(data.w_k_cell)), np.sum(~np.isnan(data.k_cell)))

    row_title = ""
    if "/" in name:
        row_title, name = name.split("/", 1)
        row_title += "\n"
        if index >= cols:
            name = ""

    plt.figure(1) #histogram of alpha and k
    #plot k as histogram
    ax_k.append(plt.subplot(rows, cols, index + 1))
    plt.title(name, fontsize=10)
    plot_density_hist(np.log10(data.k_cell))
    stat_k = get_mode_stats(data.k_cell)
    plt.xlim(0, 4)
    plt.xticks(np.arange(5))
    plt.grid()
    plt.xlabel("k")
    plt.ylabel(row_title+"probability\ndensity")
    #plot histogram of alpha
    split_axes(join_x_axes=False, join_title=True)
    ax_a.append(plt.gca())
    plot_density_hist(data.alpha_cell, color="C1")
    stat_alpha = get_mode_stats(data.alpha_cell)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1, 0.2), ["0", "", "0.4", "", "0.8"])
    plt.grid()
    plt.xlabel("$\\alpha$")
    #plt.tight_layout()

    plt.figure(2) #alpha over k
#    plt.errorbar(stat_k[0], stat_alpha[0], xerr=stat_k[1], yerr=stat_alpha[1], label=name, color=f"C{int(index//3)}", alpha=[0.3, .6, 1][index%3]) #use this with parent-subfolder
    plt.errorbar(stat_k[0], stat_alpha[0], xerr=stat_k[1], yerr=stat_alpha[1], label=name, color=plt.get_cmap("viridis")(index/14)) #use this with no parent sub-folders
    plt.legend(fontsize=8)
    plt.xlabel("k")
    plt.ylabel("alpha")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    if name.endswith("nM"):
        plt.figure(3) #for dose response
        plt.plot(float(name[:-3]), stat_k[0], "o")
        plt.xscale("log")

    plt.figure(4) #strain versus stress
    plt.subplot(rows, cols, index + 1)
    # plot_joint_density(data.k_cell, data.alpha_cell, label=name, only_kde=True)
    plt.title(name, fontsize=10)
    plotDensityScatter(data.stress, data.strain)
    plt.xlim(0, 320)
    plt.ylim(0, 1.5)
    plt.xlabel("shear stress (Pa)")
    plt.ylabel(row_title+"strain")
    #plt.tight_layout()

    plt.figure(5) # velocity versus radial position
    plt.subplot(rows, cols, index + 1)
    plt.title(name, fontsize=10)
    plot_velocity_fit(data)
    plt.grid()
    plt.xlim(0, 100)
    all_plots_same_limits() #plt.ylim(0, 1.5)
    plt.xlabel("channel position (µm)")
    plt.ylabel(row_title+"velocity\n(cm/s)")
    #plt.tight_layout()

    plt.figure(6) #G' and G'' over frequency
    plt.subplot(rows, cols, index + 1)
    plt.title(name, fontsize=10)
    ax_Gp1.append(plt.gca())
    plt.loglog(data.omega, data.Gp1, "o", alpha=0.25, ms=1)
    plt.ylabel(row_title+"G' / G'' (Pa)")
    plt.xlabel("angular frequency")
    split_axes(join_x_axes=True, join_title=True)
    ax_Gp2.append(plt.gca())
    plt.loglog(data.omega, data.Gp2, "o", color="C1", alpha=0.25, ms=1)
    all_plots_same_limits()

    plt.gca().get_title()
    #plt.tight_layout()

    plt.figure(7) #alignement angular versus radial position
    plt.subplot(rows, cols, index + 1)
    plt.title(name, fontsize=10)
    plotDensityScatter(data.rp, data.angle)
    #plotBinnedData(data.rp, data.angle, bins=np.arange(-300, 300, 10))
    plt.xlabel("radial position (µm)")
    plt.ylabel(row_title+"angle (deg)")
    #plt.tight_layout()
    all_plots_same_limits()

pylustrator.helper_functions.axes_to_grid(ax_k)
pylustrator.helper_functions.axes_to_grid(ax_a)

plt.figure(1)
plt.savefig(outputfolder / "histogram.png")

plt.figure(2)
plt.savefig(outputfolder / "alpha_over_k.png")

fig = plt.figure(3)
fig.set_size_inches(6.4, 4.8)
plt.savefig(outputfolder / "dose-response_without-control.png")

plt.figure(4)
pylustrator.helper_functions.axes_to_grid()
plt.savefig(outputfolder / "strain-stress.png")

plt.figure(5)
pylustrator.helper_functions.axes_to_grid()
plt.savefig(outputfolder / "velocity.png")

plt.figure(6)
#pylustrator.helper_functions.axes_to_grid()
pylustrator.helper_functions.axes_to_grid(ax_Gp1)
pylustrator.helper_functions.axes_to_grid(ax_Gp2)
plt.savefig(outputfolder / "g'g''.png")

plt.figure(7)
pylustrator.helper_functions.axes_to_grid()
plt.savefig(outputfolder / "angle.png")

plt.show()