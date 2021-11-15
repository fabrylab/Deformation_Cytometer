# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, load_all_data_new, plot_joint_density
import numpy as np

import pylustrator
#pylustrator.start()


data, config = load_all_data_new(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\march_2021\2021_03_31_desmin_cytoD\vim ko + hDesWT#39\DMSO\2021_03_31_12_27_01.tif", do_group=False)
#data, config = load_all_data_new(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\april_2021\2021_04_09_desmin_cytoD\vim ko\cytoD\2021_04_09_12_39_09.tif", do_group=False)

plt.clf()
print("all", np.median(data.w_k_cell), np.mean(data.w_alpha_cell), len(data))
for name, d in data.groupby("manual_exclude"):
    print(name, np.median(d.w_k_cell), np.mean(d.w_alpha_cell), len(d))
    plt.plot(np.log10(d.w_k_cell), d.w_alpha_cell, "o", ms=1, label=name)

plt.legend()
plt.xlabel("rp")
plt.ylabel("angle")


plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


