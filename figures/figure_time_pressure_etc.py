import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new
import numpy as np
import pylustrator

pylustrator.start()

""" NIH3T3 """

fit_data = []
x = []
# iterate over all times
for index, time in enumerate(range(2, 13, 1)):
    f = []
    if index == 9:
        continue
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_2\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_1\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\{time}\*_result.txt",
        #rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_1\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_3\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\{time}\*_result.txt",
    ]:
        # get the data and the fit parameters
        print(path)
        data, config = load_all_data_new(path, pressure=3)
        f.append([np.nanmean(data.alpha_cell), np.nanmedian(data.k_cell), np.sum(np.isfinite(data.k_cell))])

    x.append(time * 5)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.axes([0.2, .2, 0.5, 0.5], label="time_"+["k", "alpha"][i])
    l = plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                     np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3, color=f"C{1-i}",
                     label="NIH 3T3",
                     )
    print("###", "Time", "cellnumber", np.mean(fit_data[:, :, 2]), "experiments", fit_data.shape[1])
    for j in range(fit_data.shape[1]):
        plt.plot(x + np.random.rand(len(x)) * 0.1 - 0.05, fit_data[:, j, i], "o", ms=3,
                 color=l[0].get_color(), alpha=0.5)

""" NIH3T3 pressure """

fit_data = []
x = []
# iterate over all times
for index, pressure in enumerate([1, 2, 3]):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_2\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_1\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\*\*_result.txt",
        #rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_1\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_3\*\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\{time}\*_result.txt",
    ]:
        # get the data and the fit parameters
        print(path)
        data, config = load_all_data_new(path, pressure=pressure)
        f.append([np.nanmean(data.alpha_cell), np.nanmedian(data.k_cell), np.sum(np.isfinite(data.k_cell))])

    x.append(pressure)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.axes([0.2, .2, 0.5, 0.5], label="pressure_"+["k", "alpha"][i])
    l = plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                     np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3, color=f"C{1-i}",
                     label="NIH 3T3",
                     )
    print("###", "pressure", "cellnumber", np.mean(fit_data[:, :, 2]), "experiments", fit_data.shape[1])
    for j in range(fit_data.shape[1]):
        plt.plot(x + np.random.rand(len(x)) * 0.1 - 0.05, fit_data[:, j, i], "o", ms=3,
                 color=l[0].get_color(), alpha=0.5)

""" bioink """

print("*****************")
print("** alginate    **")
print("*****************")
# for better result, use measurements alg25: 1,2,4
alg25 = [
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_24_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_27_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_14_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_30_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",

]

alg20 = [
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_27_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_28_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_30_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
]

alg15 = [
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_30_alginate1.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_14_alginate1.5%_dmem_NIH_3T3\*\*_result.txt",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_14_alginate1.5%_dmem_NIH_3T3_2\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate1.5%_dmem_NIH_3T3\*\*_result.txt"
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_30_alginate1.5%_dmem_NIH_3T3\*\*_result.txt",
]

fit_data = []
x = []
# iterate over all times
for index, alg in enumerate([15, 20, 25]):
    f = []
    # iterate over the different experiment paths
    for path in [alg15, alg20, alg25][index]:
        # get the data and the fit parameters
        print(path)
        data, config = load_all_data_new(path, pressure=2)
        f.append([np.nanmean(data.alpha_cell), np.nanmedian(data.k_cell), np.sum(np.isfinite(data.k_cell))])

    x.append(alg/10)
    fit_data.append(f)

fit_data = np.array(fit_data)
print(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.axes([0.2, .2, 0.5, 0.5], label="bioink_"+["k", "alpha"][i])
    l = plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                     np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3, color=f"C{1-i}",
                     label="NIH 3T3",
                     )
    print("###", "alginate", "cellnumber", np.mean(fit_data[:, :, 2]), "experiments", fit_data.shape[1])
    for j in range(fit_data.shape[1]):
        plt.plot(x + np.random.rand(len(x)) * 0.1 - 0.05, fit_data[:, j, i], "o", ms=3,
                 color=l[0].get_color(), alpha=0.5)


""" inletmiddleoutlet """

print("****************************")
print("** inlet middle outlet    **")
print("****************************")

fit_data = []
x = []
# iterate over all times
for index, pos in enumerate(["inlet", "middle", "outlet"]):
    f = []
    # iterate over the different experiment paths
    for path in [
fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_14_alginate2%_diffxposition_2\{pos}\*\*_result.txt",
fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_21_alginate2%_NIH_xposition_1\{pos}\*\*_result.txt",
fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_21_alginate2%_NIH_xposition_2\{pos}\*\*_result.txt",
fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_21_alginate2%_NIH_xposition_3\{pos}\*\*_result.txt",
fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_alginate2%_NIH_diff_x_position_2\{pos}\*\*_result.txt",
fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_alginate2%_NIH_diff_x_position_3\{pos}\*\*_result.txt",
        
    ]:
        # get the data and the fit parameters
        print(path)
        data, config = load_all_data_new(path, pressure=2)
        f.append([np.nanmean(data.alpha_cell), np.nanmedian(data.k_cell), np.sum(np.isfinite(data.k_cell))])

    x.append(index)
    fit_data.append(f)

fit_data = np.array(fit_data)
print(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.axes([0.2, .2, 0.5, 0.5], label="pos_"+["k", "alpha"][i])
    l = plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                     np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3, color=f"C{1-i}",
                     label="NIH 3T3",
                     )
    print("###", "Pos", "cellnumber", np.mean(fit_data[:, :, 2]), "experiments", fit_data.shape[1])
    for j in range(fit_data.shape[1]):
        plt.plot(x + np.random.rand(len(x)) * 0.1 - 0.05, fit_data[:, j, i], "o", ms=3,
                 color=l[0].get_color(), alpha=0.5)

""" cell size """

paths = [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_2\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_1\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\*\*_result.txt",
        #rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_1\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\*\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_3\*\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\{time}\*_result.txt",
    ]

data, config = load_all_data_new(paths, pressure=3)
bins = np.percentile(data.area, [0, 33, 66, 100])
print("cell size, bins", bins)

fit_data = []
x = []
# iterate over all times
for index in range(3):
    f = []
    # iterate over the different experiment paths
    for path in paths:
        # get the data and the fit parameters
        print(path)
        data, config = load_all_data_new(path, pressure=3)
        data = data[data.area > bins[index]]
        data = data[data.area < bins[index+1]]
        f.append([np.nanmean(data.alpha_cell), np.nanmedian(data.k_cell), np.sum(np.isfinite(data.k_cell))])

    x.append(index)
    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.axes([0.2, .2, 0.5, 0.5], label="area_"+["k", "alpha"][i])
    l = plt.errorbar(x, np.mean(fit_data[:, :, i], axis=1),
                     np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3, color=f"C{1-i}",
                     label="NIH 3T3",
                     )
    print("###", "size", "cellnumber", np.mean(fit_data[:, :, 2]), "experiments", fit_data.shape[1])
    for j in range(fit_data.shape[1]):
        plt.plot(x + np.random.rand(len(x)) * 0.1 - 0.05, fit_data[:, j, i], "o", ms=3,
                 color=l[0].get_color(), alpha=0.5)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.980000/2.54, 5.910000/2.54, forward=True)
plt.figure(1).ax_dict["area_alpha"].set_xlim(-0.14763368056372056, 2.1384352600208723)
plt.figure(1).ax_dict["area_alpha"].set_ylim(0.0, 261.74081154926006)
plt.figure(1).ax_dict["area_alpha"].set_xticks([0.0, 1.0, 2.0])
plt.figure(1).ax_dict["area_alpha"].set_yticks([0.0, 200.0])
plt.figure(1).ax_dict["area_alpha"].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["area_alpha"].set_yticklabels(["", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["area_alpha"].set_position([0.845430, 0.594697, 0.146365, 0.329480])
plt.figure(1).ax_dict["area_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["area_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["area_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["area_alpha"].transAxes)  # id=plt.figure(1).ax_dict["area_alpha"].texts[0].new
plt.figure(1).ax_dict["area_alpha"].texts[0].set_position([-0.128998, 1.038176])
plt.figure(1).ax_dict["area_alpha"].texts[0].set_text("i")
plt.figure(1).ax_dict["area_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["area_k"].set_xlim(-0.12326438572207858, 2.1488649996704448)
plt.figure(1).ax_dict["area_k"].set_ylim(0.0, 0.5)
plt.figure(1).ax_dict["area_k"].set_xticks([0.0, 1.0, 2.0])
plt.figure(1).ax_dict["area_k"].set_yticks([0.0, 0.25, 0.5])
plt.figure(1).ax_dict["area_k"].set_xticklabels(["small", "medium", "big"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["area_k"].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["area_k"].set_position([0.845430, 0.196150, 0.146365, 0.329480])
plt.figure(1).ax_dict["area_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["area_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["area_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["area_k"].transAxes)  # id=plt.figure(1).ax_dict["area_k"].texts[0].new
plt.figure(1).ax_dict["area_k"].texts[0].set_position([-0.128998, 0.968015])
plt.figure(1).ax_dict["area_k"].texts[0].set_text("j")
plt.figure(1).ax_dict["area_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["area_k"].get_xaxis().get_label().set_text("cell size")
plt.figure(1).ax_dict["bioink_alpha"].set_xlim(1.3952195212681158, 2.6032812679138617)
plt.figure(1).ax_dict["bioink_alpha"].set_ylim(0.0, 261.74081154926006)
plt.figure(1).ax_dict["bioink_alpha"].set_yticks([0.0, 200.0])
plt.figure(1).ax_dict["bioink_alpha"].set_xticklabels([])
plt.figure(1).ax_dict["bioink_alpha"].set_yticklabels(["", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["bioink_alpha"].set_position([0.671119, 0.594697, 0.146365, 0.329480])
plt.figure(1).ax_dict["bioink_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["bioink_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["bioink_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["bioink_alpha"].transAxes)  # id=plt.figure(1).ax_dict["bioink_alpha"].texts[0].new
plt.figure(1).ax_dict["bioink_alpha"].texts[0].set_position([-0.165224, 1.038176])
plt.figure(1).ax_dict["bioink_alpha"].texts[0].set_text("g")
plt.figure(1).ax_dict["bioink_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["bioink_alpha"].get_xaxis().get_label().set_text('')
plt.figure(1).ax_dict["bioink_k"].set_xlim(1.3952195212681158, 2.6032812679138617)
plt.figure(1).ax_dict["bioink_k"].set_ylim(0.0, 0.5)
plt.figure(1).ax_dict["bioink_k"].set_yticklabels([])
plt.figure(1).ax_dict["bioink_k"].set_position([0.671119, 0.193851, 0.146365, 0.329480])
plt.figure(1).ax_dict["bioink_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["bioink_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["bioink_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["bioink_k"].transAxes)  # id=plt.figure(1).ax_dict["bioink_k"].texts[0].new
plt.figure(1).ax_dict["bioink_k"].texts[0].set_position([-0.165224, 0.974992])
plt.figure(1).ax_dict["bioink_k"].texts[0].set_text("h")
plt.figure(1).ax_dict["bioink_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["bioink_k"].get_xaxis().get_label().set_text("alginate %")
plt.figure(1).ax_dict["bioink_k"].get_yaxis().get_label().set_text('')
plt.figure(1).ax_dict["pos_alpha"].set_xlim(-0.14855274904467136, 2.1535578799008785)
plt.figure(1).ax_dict["pos_alpha"].set_ylim(0.0, 261.74081154926006)
plt.figure(1).ax_dict["pos_alpha"].set_xticks([0.0, 1.0, 2.0])
plt.figure(1).ax_dict["pos_alpha"].set_yticks([0.0, 200.0])
plt.figure(1).ax_dict["pos_alpha"].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["pos_alpha"].set_yticklabels(["", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["pos_alpha"].set_position([0.496807, 0.592783, 0.146365, 0.329480])
plt.figure(1).ax_dict["pos_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pos_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pos_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pos_alpha"].transAxes)  # id=plt.figure(1).ax_dict["pos_alpha"].texts[0].new
plt.figure(1).ax_dict["pos_alpha"].texts[0].set_position([-0.101886, 1.043986])
plt.figure(1).ax_dict["pos_alpha"].texts[0].set_text("e")
plt.figure(1).ax_dict["pos_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pos_k"].set_xlim(-0.1339748743572315, 2.135578353332946)
plt.figure(1).ax_dict["pos_k"].set_ylim(0.0, 0.5)
plt.figure(1).ax_dict["pos_k"].set_xticks([0.0, 1.0, 2.0])
plt.figure(1).ax_dict["pos_k"].set_yticks([0.0, 0.25, 0.5])
plt.figure(1).ax_dict["pos_k"].set_xticklabels(["inlet", "middle", "outlet"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["pos_k"].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["pos_k"].set_position([0.496807, 0.196150, 0.146365, 0.329480])
plt.figure(1).ax_dict["pos_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pos_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pos_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pos_k"].transAxes)  # id=plt.figure(1).ax_dict["pos_k"].texts[0].new
plt.figure(1).ax_dict["pos_k"].texts[0].set_position([-0.111824, 0.969543])
plt.figure(1).ax_dict["pos_k"].texts[0].set_text("f")
plt.figure(1).ax_dict["pos_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pos_k"].get_xaxis().get_label().set_text("channel pos")
plt.figure(1).ax_dict["pressure_alpha"].set_xlim(0.5, 3.5)
plt.figure(1).ax_dict["pressure_alpha"].set_ylim(0.0, 261.74081154926006)
plt.figure(1).ax_dict["pressure_alpha"].set_yticks([0.0, 200.0])
plt.figure(1).ax_dict["pressure_alpha"].set_xticklabels([])
plt.figure(1).ax_dict["pressure_alpha"].set_yticklabels(["", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["pressure_alpha"].set_position([0.322496, 0.594697, 0.146365, 0.329480])
plt.figure(1).ax_dict["pressure_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pressure_alpha"].transAxes)  # id=plt.figure(1).ax_dict["pressure_alpha"].texts[0].new
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_position([-0.257195, 1.038176])
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_text("c")
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pressure_alpha"].get_xaxis().get_label().set_text('')
plt.figure(1).ax_dict["pressure_alpha"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["pressure_k"].set_xlim(0.5, 3.5)
plt.figure(1).ax_dict["pressure_k"].set_ylim(0.0, 0.5)
plt.figure(1).ax_dict["pressure_k"].set_yticklabels([])
plt.figure(1).ax_dict["pressure_k"].set_position([0.322496, 0.193851, 0.146365, 0.329480])
plt.figure(1).ax_dict["pressure_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pressure_k"].transAxes)  # id=plt.figure(1).ax_dict["pressure_k"].texts[0].new
plt.figure(1).ax_dict["pressure_k"].texts[0].set_position([-0.257195, 0.974992])
plt.figure(1).ax_dict["pressure_k"].texts[0].set_text("d")
plt.figure(1).ax_dict["pressure_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pressure_k"].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).ax_dict["pressure_k"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["time_alpha"].set_xlim(0.0, 67.0)
plt.figure(1).ax_dict["time_alpha"].set_ylim(0.0, 261.74081154926006)
plt.figure(1).ax_dict["time_alpha"].set_xticklabels([])
plt.figure(1).ax_dict["time_alpha"].legend(handletextpad=0.7999999999999999, fontsize=8.0, title_fontsize=7.0)
plt.figure(1).ax_dict["time_alpha"].set_position([0.085637, 0.594181, 0.208913, 0.329996])
plt.figure(1).ax_dict["time_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].get_legend().set_visible(False)
plt.figure(1).ax_dict["time_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).ax_dict["time_alpha"].texts[0].set_position([-0.409025, 1.038116])
plt.figure(1).ax_dict["time_alpha"].texts[0].set_text("a")
plt.figure(1).ax_dict["time_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["time_alpha"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["time_alpha"].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).ax_dict["time_k"].set_xlim(0.0, 67.0)
plt.figure(1).ax_dict["time_k"].set_ylim(0.0, 0.5)
plt.figure(1).ax_dict["time_k"].set_position([0.085637, 0.193851, 0.208913, 0.329996])
plt.figure(1).ax_dict["time_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["time_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["time_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).ax_dict["time_k"].texts[0].set_position([-0.409025, 0.973466])
plt.figure(1).ax_dict["time_k"].texts[0].set_text("b")
plt.figure(1).ax_dict["time_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["time_k"].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).ax_dict["time_k"].get_yaxis().get_label().set_text("alpha")
#% end: automatic generated code from pylustrator
#plt.savefig(__file__[:-3]+".png", dpi=300)
#plt.savefig(__file__[:-3]+".pdf")
plt.show()
