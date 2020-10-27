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

pylustrator.start()

data1, config1 = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
            ], pressure=1)

data = data1#pd.concat([data1, data2, data3])
config = config1

fits = []
frac = np.arange(0.1, 1.1, 0.1)
for i in frac:
    d = data[data.stress < i]
    ps = []
    for j in range(100):
        d0 = d.sample(frac=i, replace=True)
        p = fitStiffness(d0, config)
        #p = np.random.rand(3)
        ps.append(p)
    fits.append(ps)
fits = np.array(fits)
#np.save(__file__[:-3], fits)

print(fits)
print(fits.shape)
fit_means = np.mean(fits, axis=-2)
print(fit_means)
fit_stds = np.std(fits, axis=-2)
print(fit_stds)
for i in range(3):
    plt.subplot(1, 3, i+1)
    print(fit_means.shape)
    plt.fill_between(frac, fit_means[:, i]-fit_stds[:, i], fit_means[:, i]+fit_stds[:, i], alpha=0.5)
    plt.plot(frac, fit_means[:, i])

#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 4.390000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.0, 1.0)
plt.figure(1).axes[0].set_ylim(0.0, 219.62205480851765)
plt.figure(1).axes[0].set_xticks([0.0])
plt.figure(1).axes[0].set_xticklabels(["0"], fontsize=10)
plt.figure(1).axes[0].set_position([0.093888, 0.257929, 0.231421, 0.704405])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.318443, 0.968097])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("maximum shear stress (Pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).axes[1].set_xlim(0.0, 1.0)
plt.figure(1).axes[1].set_ylim(0.0, 1.0)
plt.figure(1).axes[1].set_position([0.415122, 0.257929, 0.231421, 0.704405])
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.330986, 0.968097])
plt.figure(1).axes[1].texts[0].set_text("b")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("maximum shear stress (Pa)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("alpha")
plt.figure(1).axes[2].set_xlim(0.0, 1.0)
plt.figure(1).axes[2].set_ylim(0.0, 0.2)
plt.figure(1).axes[2].set_position([0.736355, 0.257929, 0.231421, 0.704405])
plt.figure(1).axes[2].set_xticks([np.nan], minor=True)
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].yaxis.labelpad = 3.280000
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.334122, 0.968097])
plt.figure(1).axes[2].texts[0].set_text("c")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("maximum shear stress (Pa)")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("offset")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


