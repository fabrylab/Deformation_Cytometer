import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist
import numpy as np
import pylustrator

pylustrator.start()

def get_mode_stats(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)&(x>0)]
    def get_mode(x):
        kde = stats.gaussian_kde(x)
        return x[np.argmax(kde(x))]
    mode = get_mode(x)
    #err = bootstrap_error(x, get_mode, repetitions=2)
    return mode


def get_mode_stats_log(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)&(x>0)]
    def get_mode(x):
        x = np.log(x)
        kde = stats.gaussian_kde(x)
        return np.exp(x[np.argmax(kde(x))])
    mode = get_mode(x)
    #err = bootstrap_error(x, get_mode, repetitions=2)
    return mode

def get_mode_stats_err(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)&(x>0)]
    def get_mode(x):
        x = np.log(x)
        kde = stats.gaussian_kde(x)
        return np.exp(x[np.argmax(kde(x))])
    #mode = get_mode(x)
    err = bootstrap_error(x, get_mode, repetitions=2)
    return err

def plotMeasurementOfParameter(data, parameter, value, agg="mean", color=None, label=None):
    agg_func = {"mean": np.mean, "median": np.median, "mode": get_mode_stats, "logmode": get_mode_stats_log}[agg]

    agg = data.groupby([parameter, "measurement_id"])[value].agg(agg_func).reset_index()
    #agg_err = data.groupby([parameter, "measurement_id"])[value].agg(get_mode_stats_err).reset_index()
    #agg = data.groupby([parameter, "measurement_id"])[value].mean().reset_index()
    #agg_err = data.groupby([parameter, "measurement_id"])[value].sem().reset_index()
    agg_mean = agg.groupby(parameter)[value].mean()
    agg_err_mean = agg.groupby(parameter)[value].sem()#agg(lambda x: 0.5*np.sqrt((x**2).sum()))
    l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2, label=label)
    for measurement_id, meas_data in agg.groupby("measurement_id"):
        plt.plot(meas_data[parameter], meas_data[value], "o", ms=3, color=l[0].get_color(), alpha=0.5)
        plt.plot(meas_data[parameter], meas_data[value], "-", ms=3, color="gray", alpha=0.5)
        #plt.text(meas_data[parameter].values[0], meas_data[value].values[0], measurement_id)
    plt.xlabel(parameter)
    plt.ylabel(value)


def plot_pair(data, parameter, color=["C0", "C1"], label=None):
    plt.axes([0.1, .2, 0.4, 0.5], label=f"{parameter}_alpha")
    plotMeasurementOfParameter(data, parameter, "w_k_cell", agg="logmode", color=color[0], label=label)
    plt.ylim(bottom=0)
    plt.axes([0.5, .2, 0.5, 0.5], label=f"{parameter}_k")
    plotMeasurementOfParameter(data, parameter, "w_alpha_cell", agg="mode", color=color[1], label=label)
    plt.ylim(bottom=0)


data_bleb, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\12.04.2021_THP1_CytoD_10µM_*\**\*_evaluated_new.csv"
], pressure=2)
data_dmso, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\24.03.2021_THP1_2%Alginate\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%\**\*_evaluated_new.csv",
#r"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%_2\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%_3\**\*_evaluated_new.csv",
], pressure=2)
print(data_bleb.pressure.unique(), data_dmso.pressure.unique())
plot_pair(data_bleb, "time", ["C0", "C0"], label="CytoD")
plot_pair(data_dmso, "time", ["C3", "C3"], label="Control")
plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 6.000000/2.54, forward=True)
plt.figure(1).ax_dict["time_alpha"].set_ylim(0.0, 250.0)
plt.figure(1).ax_dict["time_alpha"].set_position([0.125938, 0.289948, 0.325551, 0.538804])
plt.figure(1).ax_dict["time_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["time_alpha"].transAxes)  # id=plt.figure(1).ax_dict["time_alpha"].texts[0].new
plt.figure(1).ax_dict["time_alpha"].texts[0].set_position([-0.186747, 1.035135])
plt.figure(1).ax_dict["time_alpha"].texts[0].set_text("a")
plt.figure(1).ax_dict["time_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["time_alpha"].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).ax_dict["time_alpha"].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).ax_dict["time_k"].set_ylim(0.0, 0.4)
plt.figure(1).ax_dict["time_k"].set_position([0.556017, 0.289948, 0.325551, 0.538804])
plt.figure(1).ax_dict["time_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["time_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["time_k"].get_legend()._set_loc((0.354786, 0.891042))
plt.figure(1).ax_dict["time_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["time_k"].transAxes)  # id=plt.figure(1).ax_dict["time_k"].texts[0].new
plt.figure(1).ax_dict["time_k"].texts[0].set_position([-0.251855, 1.035135])
plt.figure(1).ax_dict["time_k"].texts[0].set_text("b")
plt.figure(1).ax_dict["time_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["time_k"].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).ax_dict["time_k"].get_yaxis().get_label().set_text("alpha")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
