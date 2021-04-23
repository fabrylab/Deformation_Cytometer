import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist, bootstrap_error
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
    agg_func_err = {"mean": lambda x: np.std(x)/len(x), "median": lambda x: bootstrap_error(x, np.median),
                    "mode": lambda x: bootstrap_error(x, get_mode_stats, repetitions=2), "logmode": lambda x: bootstrap_error(x, get_mode_stats_log, repetitions=2)}[agg]

    if 1:
        #agg = data.groupby([parameter, "measurement_id"])[value].agg(agg_func).reset_index()
        # agg_err = data.groupby([parameter, "measurement_id"])[value].agg(get_mode_stats_err).reset_index()
        # agg = data.groupby([parameter, "measurement_id"])[value].mean().reset_index()
        # agg_err = data.groupby([parameter, "measurement_id"])[value].sem().reset_index()
        agg_mean = data.groupby(parameter)[value].agg(agg_func)
        agg_err_mean = data.groupby(parameter)[value].agg(agg_func_err)  # agg(lambda x: 0.5*np.sqrt((x**2).sum()))
        l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2,
                         label=label)
    else:
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
    plotMeasurementOfParameter(data, parameter, "w_k_cell", agg="median", color=color[0], label=label)
    plt.ylim(bottom=0)
    plt.axes([0.5, .2, 0.5, 0.5], label=f"{parameter}_k")
    plotMeasurementOfParameter(data, parameter, "w_alpha_cell", agg="mean", color=color[1], label=label)
    plt.ylim(bottom=0)



data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.14\**\*_evaluated_new.csv"
])
data.pressure = np.round(data.pressure, 2)
plot_pair(data, "pressure", ["C0", "C1"])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 5.300000/2.54, forward=True)
plt.figure(1).ax_dict["pressure_alpha"].set_xlim(0.0, 0.52)
plt.figure(1).ax_dict["pressure_alpha"].set_ylim(0.0, 10.0)
plt.figure(1).ax_dict["pressure_alpha"].set_position([0.125368, 0.310687, 0.364634, 0.554430])
plt.figure(1).ax_dict["pressure_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pressure_alpha"].transAxes)  # id=plt.figure(1).ax_dict["pressure_alpha"].texts[0].new
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_position([-0.278460, 1.051865])
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_text("a")
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pressure_alpha"].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).ax_dict["pressure_alpha"].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).ax_dict["pressure_k"].set_xlim(0.0, 0.52)
plt.figure(1).ax_dict["pressure_k"].set_ylim(0.0, 0.6)
plt.figure(1).ax_dict["pressure_k"].set_position([0.614226, 0.310687, 0.364634, 0.554430])
plt.figure(1).ax_dict["pressure_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pressure_k"].transAxes)  # id=plt.figure(1).ax_dict["pressure_k"].texts[0].new
plt.figure(1).ax_dict["pressure_k"].texts[0].set_position([-0.230888, 1.051865])
plt.figure(1).ax_dict["pressure_k"].texts[0].set_text("b")
plt.figure(1).ax_dict["pressure_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pressure_k"].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).ax_dict["pressure_k"].get_yaxis().get_label().set_text("alpha")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
