# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scripts.helper_functions import load_all_data, plotDensityScatter, plotBinnedData, all_plots_same_limits
import numpy as np
from scipy import stats

import pylustrator
pylustrator.start()

if 1:
    numbers = []
    for index, pressure in enumerate([0.5, 1, 1.5]):
        ax = plt.subplot(1, 3, index+1)

        #data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data" +
        #                             r"\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt", pressure=pressure)
        data, config = load_all_data([
        rf"\\131.188.117.96\biophysDS\meroles\2020.05.27_THP1_RPMI_2pc_Ag\THP1_27_05_2020_2replicate\*\*_result.txt",
#            rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\*\*_result.txt",
#            rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\*\*_result.txt",
            ], pressure=pressure)

        plotDensityScatter(data.rp, data.area)
        plotBinnedData(data.rp, data.area, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
        #plt.hist(data.area, bins=100, density=True, alpha=0.8)
        numbers.append(len(data.rp))

        #kde = stats.gaussian_kde(np.hstack((data.rp, -data.rp)))
        #xx = np.linspace(-110, 110, 1000)
        #plt.plot(xx, kde(xx), "--k", lw=0.8)

    print("numbers", numbers)

all_plots_same_limits()
plt.show()


