import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data
import numpy as np
import pylustrator

pylustrator.start()

rows = 3
cols = 5

""" THP1 """

fit_data = []
x = []
# iterate over all times
for index, time in enumerate(range(1, 7, 1)):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\meroles\2020.07.07.Control.sync\Control2\T{time * 10}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\meroles\2020.06.26-Control\Control2\T{time * 10}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\meroles\2020.06.26-Control\Control1\T{time * 10}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\meroles\2020.07.09_control_sync_hepes\Control3\T{time * 10}\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, repetition=2)
        f.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

    x.append(time*10)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the three different plots
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="THP1",
                 )

""" K562 """

fit_data = []
x = []
# iterate over all times
for index, time in enumerate(range(2, 11, 1)):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\{time}\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, repetition=2)
        f.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

    x.append(time*5)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the three different plots
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="K562",
                 )

""" NIH3T3 """

fit_data = []
x = []
# iterate over all times
for index, time in enumerate(range(2, 14, 1)):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, repetition=1)
        f.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

    x.append(time*5)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the three different plots
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="NIH 3T3",
                 )

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).set_size_inches(15.980000/2.54, 5.970000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.062778, 0.202909, 0.241927, 0.689799])
plt.figure(1).axes[0].set_xlim(0.0, 67.0)
plt.figure(1).axes[0].set_ylim(0.0, 4.0)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.209737, 1.038176])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("alpha")
plt.figure(1).axes[1].set_position([0.416761, 0.202909, 0.226609, 0.710959])
plt.figure(1).axes[1].set_xlim(0.0, 67.0)
plt.figure(1).axes[1].set_ylim(0.0, 300.0)
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.350939, 1.007277])
plt.figure(1).axes[1].texts[0].set_text("b")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("prestress (Pa)")
plt.figure(1).axes[2].legend(handletextpad=0.7999999999999999, fontsize=8.0, title_fontsize=7.0)
plt.figure(1).axes[2].set_position([0.755425, 0.202909, 0.226609, 0.710959])
plt.figure(1).axes[2].set_xlim(0.0, 67.0)
plt.figure(1).axes[2].set_ylim(0.0, 150.0)
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_legend()._set_loc((0.440211, 0.043432))
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.343916, 1.007277])
plt.figure(1).axes[2].texts[0].set_text("c")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("stiffness (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()
