# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties#, load_all_data_new
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, plot_joint_density, split_axes, get_mode_stats, bootstrap_match_hist, load_all_data_new, get2Dhist_k_alpha, getGp1Gp2fit_k_alpha
import numpy as np

import pylustrator
#pylustrator.start()
import glob
import pandas as pd
from pathlib import Path
import natsort

data0, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2021\2021_08_11_HL60_latB",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2021\2021_08_12_HL60_latB",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2021\2021_08_13_HL60_latB",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\july_2021\2021_07_14_HL60_LatB",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\july_2021\2021_07_26_HL60_LatB",
])

data0["concentration"] = np.where((data0["treatment"] != "control")&(data0["treatment"] != "DMSO"), data0["treatment"], 0)
data0["treatment"] = np.where((data0["treatment"] != "control")&(data0["treatment"] != "DMSO"), "latB", data0["treatment"])
data0["concentration"] = data0["concentration"].astype(float)
data0["concentration"] = 10**np.round(np.log10(data0["concentration"]), 1)
data0["pressure"] = np.round(data0["pressure"], 1)
#data = data0.query("1.9 < pressure < 2.1")

new_data = []
for measurement_id, d in data0.groupby("measurement_id"):
    for (treatment, concentration, pressure), dd in d.groupby(["treatment", "concentration", "pressure"]):
        print(measurement_id, treatment)
        #k, alpha = getGp1Gp2fit_k_alpha(dd)
        k, alpha = get2Dhist_k_alpha(dd)
        new_data.append(dict(k=k, alpha=alpha, treatment=treatment, measurement_id=measurement_id, concentration=concentration, pressure=pressure))
new_data = pd.DataFrame(new_data)

new_data["k_rel"] = 0
new_data["k_rel2"] = 0
for (measurement_id, pressure), d in new_data.groupby(["measurement_id", "pressure"]):
    print((measurement_id, pressure))
    new_data['k_rel'] = np.where( (new_data['measurement_id'] == measurement_id) & (new_data['pressure'] == pressure), new_data.k/float(d[d.treatment == "control"].k), new_data['k_rel'])
    new_data['k_rel2'] = np.where( (new_data['measurement_id'] == measurement_id) & (new_data['pressure'] == pressure), new_data.k/float(d[d.treatment == "DMSO"].k), new_data['k_rel2'])

colors = {}
offsets = {}
for id, d in new_data.query("treatment == 'latB'").groupby(["measurement_id", "pressure"]):
    for p, d in d.groupby("pressure"):
        d = d.groupby("concentration")["k_rel"].agg(["mean", "sem"])
        if p not in colors:
            offsets[p] = 1+len(colors)/5
            colors[p] = f"C{len(colors)}"
        plt.plot(d.index*offsets[p], d["mean"].values, "o", color=colors[p], alpha=0.5)
        #plt.text(d.index[-1], d["mean"].values[-1], id)

for p in colors:
    plt.plot([], [], "o", alpha=0.5, label=p, color=colors[p])
plt.legend()

d = new_data.query("treatment == 'latB' and pressure == 1").groupby("concentration")["k_rel"].agg(["mean", "sem"])
plt.errorbar(d.index, d["mean"].values, d["sem"].values, fmt="o")
plt.semilogx()

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def hill_fit(x, b, m, s):
    return b + (1 - b) / (1 + (x/m)**s)

x = d.index
y = d["mean"].values
yerr = d["sem"].values
popt, pcov = curve_fit(hill_fit, x, y, [0.3, 30, 2], sigma=yerr) #(0.25, 0.15, 5., x)
print(popt)
xplot = 10 ** np.arange(-0.2, 3.5, 0.1)
plt.plot(xplot, hill_fit(xplot, *popt))
plt.ylim(bottom=0)
plt.xlabel("concentration")
plt.ylabel("stiffness change k/ko")

plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


