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
from deformationcytometer.includes.includes import getInputFile, read_args_evaluate
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data_new, get_cell_properties
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData
settings_name = "strain_vs_stress_clean"
""" loading data """

# reading commandline arguments if executed from terminal
file, irregularity_threshold, solidity_threshold = read_args_evaluate()

# get the results file (by command line parameter or user input dialog)
datafile = getInputFile(filetype="csv file (*_evaluated_new.csv)", settings_name=settings_name, video=file)
print("evaluate file", datafile)
# load the data and the config
data, config = load_all_data_new(datafile)

plt.figure(0, (10, 8))

plt.subplot(2, 3, 1)
plt.cla()
plot_velocity_fit(data)
plt.text(0.9, 0.9, f"$\\eta_0$ {data.eta0[0]:.2f}\n$\\delta$ {data.delta[0]:.2f}\n$\\tau$ {data.tau[0]:.2f}", transform=plt.gca().transAxes, va="top", ha="right")

plt.subplot(2, 3, 2)
plt.cla()
plotDensityScatter(data.stress, data.epsilon)
plotBinnedData(data.stress, data.epsilon, bins=np.arange(0, 300, 10))
plt.xlabel("stress (Pa)")
plt.ylabel("strain")

plt.subplot(2, 3, 3)
plt.cla()
plotDensityScatter(data.rp, data.angle)
plotBinnedData(data.rp, data.angle, bins=np.arange(-300, 300, 10))
plt.xlabel("radial position (µm)")
plt.ylabel("angle (deg)")

plt.subplot(2, 3, 4)
plt.loglog(data.omega, data.Gp1, "o", alpha=0.25)
plt.loglog(data.omega, data.Gp2, "o", alpha=0.25)
plt.ylabel("G' / G''")
plt.xlabel("angular frequency")

plt.subplot(2, 3, 5)
plt.cla()
plt.xlim(0, 4)
plot_density_hist(np.log10(data.k_cell), color="C0")
plt.xlabel("log10(k)")
plt.ylabel("relative density")
plt.text(0.9, 0.9, f"mean(log10(k)) {np.mean(np.log10(data.k_cell)):.2f}\nstd(log10(k)) {np.std(np.log10(data.k_cell)):.2f}\nmean(k) {np.mean(data.k_cell):.2f}\nstd(k) {np.std(data.k_cell):.2f}\n", transform=plt.gca().transAxes, va="top", ha="right")

plt.subplot(2, 3, 6)
plt.cla()
plt.xlim(0, 1)
plot_density_hist(data.alpha_cell, color="C1")
plt.xlabel("alpha")
plt.text(0.9, 0.9, f"mean($\\alpha$) {np.mean(data.alpha_cell):.2f}\nstd($\\alpha$) {np.std(data.alpha_cell):.2f}\n", transform=plt.gca().transAxes, va="top", ha="right")

plt.tight_layout()
try:
    plt.savefig(datafile[:-18] + '_evaluation.pdf')
except PermissionError:
    pass
#plt.show()

output = Path("all_data.csv")
if not output.exists():
    with output.open("w") as fp:
        fp.write("filename, seconds, pressure (bar), #cells, diameter, vmax, eta0, tau, delta, 10^logmean(k) (Pa), logstd(k), mean(alpha), std(alpha)\n")

date_time = str(Path(config["file_data"]).name).split('\\')
date_time = date_time[-1].split('_')
seconds = float(date_time[3]) * 60 * 60 + float(date_time[4]) * 60 + float(date_time[5])

d = data.iloc[0]
with open("all_data.csv", "a") as fp:
    fp.write(f"{Path(config['file_data'])}, {int(seconds)}, {config['pressure_pa']*1e-5}, {len(data)}, {np.mean(data.area)}, {np.max(data.vel)}, {d.eta0}, {d.tau}, {d.delta}, {10**np.mean(np.log10(data.k_cell))}, {np.std(np.log10(data.k_cell))}, {np.mean(data.alpha_cell)}, {np.std(data.alpha_cell)}\n")

