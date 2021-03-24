import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data_new
import numpy as np
import pylustrator

pylustrator.start()


""" NIH3T3 """

fit_data = []
x = []
# iterate over all times
for index, time in enumerate(range(2, 13, 1)):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\{time}\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data_new(path, pressure=1)
        data, config = load_all_data_new(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\**\*_result.txt", pressure=1)

        data1 = data[data.pressure == 1]
        #f.append([np.mean(data.alpha_cell), np.mean(np.log10(data.k_cell))])
        f.append([np.mean(data.alpha_cell), np.nanmedian(data.k_cell)])

    x.append(time*5)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.subplot(1, 2, i+1)
    l = plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="NIH 3T3",
                 )
    for j in range(fit_data.shape[1]):
        plt.plot(x + np.random.rand(len(x)) * 0.1 - 0.05, fit_data[:, j, i], "o", ms=3,
                 color=l[0].get_color(), alpha=0.5)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(15.980000/2.54, 5.970000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.0, 67.0)
plt.figure(1).axes[0].set_ylim(0.0, 0.5)
plt.figure(1).axes[0].set_position([0.585163, 0.202909, 0.381927, 0.689799])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.209737, 1.038176])
plt.figure(1).axes[0].texts[0].set_text("b")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("alpha")
plt.figure(1).axes[1].set_xlim(0.0, 67.0)
plt.figure(1).axes[1].set_ylim(0.0, 261.74081154926006)
plt.figure(1).axes[1].legend(handletextpad=0.7999999999999999, fontsize=8.0, title_fontsize=7.0)
plt.figure(1).axes[1].set_position([0.095965, 0.202909, 0.386609, 0.689799])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_legend()._set_loc((1.895941, 0.714523))
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.145620, 1.038176])
plt.figure(1).axes[1].texts[0].set_text("a")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("stiffness (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()
