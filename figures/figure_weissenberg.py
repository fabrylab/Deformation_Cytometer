# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, plotDensityScatter, plotBinnedData
import numpy as np
from scipy import stats

import pylustrator
pylustrator.start()


data, config = load_all_data([
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
], pressure=3)

def curve(x, x0, a):
    return 1 / 2 * 1 / (1 + (x / x0) ** a)


omega_weissenberg = curve(np.abs(data.vel_grad), (1 / data.tau) * 3, data.delta) * np.abs(data.vel_grad)  # * np.pi*2

plt.plot(data.omega, omega_weissenberg, "o")

plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")

plt.show()


