import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist, bootstrap_error, plotDensityScatter, get2Dhist_k_alpha, plot_velocity_fit, plotBinnedData
import numpy as np
import pylustrator

#pylustrator.start()
#pylustrator.load("paa_image.py")

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

def getFitParameters(data):
    from scipy.special import gamma
    def fit(omega, k, alpha):
        omega = np.array(omega)
        G = k * (1j * omega) ** alpha * gamma(1 - alpha)
        return np.real(G), np.imag(G)

    def cost(p):
        Gp1, Gp2 = fit(data.omega_weissenberg, *p)
        return np.sum(np.abs(np.log10(data.w_Gp1) - np.log10(Gp1))) + np.sum(
            np.abs(np.log10(data.w_Gp2) - np.log10(Gp2)))

    from scipy.optimize import minimize
    res = minimize(cost, [np.median(data.w_k_cell), np.mean(data.w_alpha_cell)], bounds=([0, np.inf], [0, 1]),
                   options={"disp": False})

    return res.x


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
        l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2, label=label, fmt="o-")
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

rep = 0
def plot_stiffness(data, parameter, color=["C0", "C1"], label=None):
    global rep
    plt.axes([0.1, .2, 0.4, 0.5], label=f"{parameter}_stiffnes_stress_{rep}")
    plt.title(label)
    plotDensityScatter(data.stress, data.w_k_cell)
    plt.axes([0.1, .2, 0.4, 0.5], label=f"{parameter}_alpha_stress_{rep}")
    plotDensityScatter(data.stress, data.w_alpha_cell)
    plt.title(label)
    rep += 1

ax1 = plt.subplot(321, label="stiffness")
ax2 = plt.subplot(322, label="alpha")
ax3 = plt.subplot(323, label="Gp1")
ax4 = plt.subplot(324, label="Gp2")
ax5 = plt.subplot(325, label="AFM")
#ax6 = plt.subplot(326, label="alpha")

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.14",
])
data.pressure = np.round(data.pressure, 2)
from scipy.special import gamma

def func(w, k, a, eta):
    Gp1 = k * w **a * np.cos(np.pi / 2 *a)*gamma(1 - a)
    Gp2 = k * w **a * np.sin(np.pi / 2 *a)*gamma(1 - a) + w * eta# * data.strain
    return Gp1, Gp2

def cost(w, w_Gp1, w_Gp2):
    def c(p):
        Gp1, Gp2 = func(w, *p)
        #return np.sum((Gp1-data.w_Gp1)**2 + (Gp2-data.w_Gp2)**2)
        return np.mean((np.log(Gp1)-np.log(w_Gp1))**2) + np.mean((np.log(Gp2)-np.log(w_Gp2))**2)
    return c

from scipy.optimize import minimize

def plot_afm():
    afm = np.array([
    [0.628318531,	0.1	,114.3164556	,5.772011427],
    [3.141592654,	0.5	,119.2851729	,8.352389235],
    [6.283185307,	1	,119.3196243	,12.95380098],
    [18.84955592,	3	,128.4819193	,25.22817804],
    [31.41592654,	5	,137.1353971	,33.60356653],
    [62.83185307,	10	,149.8234193	,48.34501645],
    [125.6637061,	20,	158.632765	,77.60179818],
    [188.4955592,	30,	178.73844	,107.5468772],
    [314.1592654,	50	,178.6307385	,161.7222871],
    [439.8229715,	70,	188.201614,	213.985693],
    [628.3185307,	100,	161.652555	,273.1010434],
    [942.4777961,	150,	137.4191114,	393.786563],
    ])

    res = minimize(cost(afm[:, 0], afm[:, 2], afm[:, 3]), [120, 0.1, 13], method='Nelder-Mead')
    Gp1, Gp2 = func(afm[:, 0], *res["x"])

    plt.loglog(afm[:, 0], afm[:, 2], "C0o")
    plt.plot(afm[:, 0], Gp1)
    plt.loglog(afm[:, 0], afm[:, 3], "C1o")
    plt.plot(afm[:, 0], Gp2)
plt.sca(ax5)
plot_afm()

def plot_data(data, ax=None, ax2=None):
    data = data[data.w_Gp1 > 0]
    data = data[data.w_Gp2 > 0]

    res = minimize(cost(data.omega_weissenberg, data.w_Gp1, data.w_Gp2), [120, 0.1, 13], method='Nelder-Mead')
    #print(pressure, *res["x"])

    w = np.geomspace(data.omega_weissenberg.min(), data.omega_weissenberg.max(), 100)

    Gp1, Gp2 = func(w, *res["x"])

    if ax is None:
        ax = plt.subplot(121)
    plt.sca(ax)
    plotDensityScatter(data.omega_weissenberg, data.w_Gp1)
    #joined_hex_bin(data.omega_weissenberg, data.w_Gp1, data.omega_weissenberg, data.w_Gp2, loglog=True)
    plotBinnedData(data.omega_weissenberg, data.w_Gp1, np.linspace(0, 1.5, 10), xscale="log")
    plt.plot(w, Gp1, "-C0", lw=2)
    #plt.plot(afm[:, 0], afm[:, 2], "o")
    plt.loglog()

    if ax2 is None:
        ax2 = plt.subplot(122, sharex=ax, sharey=ax)
    plt.sca(ax2)
    plotDensityScatter(data.omega_weissenberg, data.w_Gp2)
    plotBinnedData(data.omega_weissenberg, data.w_Gp2, np.linspace(0, 1.5, 10), xscale="log")

    plt.plot(w, Gp2, "-C1", lw=2)
    #plt.plot(afm[:, 0], afm[:, 3], "o")
    plt.loglog()
    #plt.show()


#from deformationcytometer.evaluation.helper_functions import joined_hex_bin
#plt.clf()

d = np.array([list(get2Dhist_k_alpha(d))+[p, np.median(d.strain)] for p, d in data.groupby("pressure")])
plt.sca(ax1)
plt.plot(d[:,2],d[:,0],"o-",label="5")
plt.sca(ax2)
plt.plot(d[:,3], d[:, 1], "o-",label="5")


#plot_pair(data, "pressure", ["C0", "C0"], label="5")
#plot_stiffness(data, "pressure", ["C0", "C0"], label="5")

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.30\200 pa",
])
data.pressure = np.round(data.pressure, 2)

#plot_pair(data, "pressure", ["C1", "C1"], label="200")
#plot_stiffness(data, "pressure", ["C0", "C0"], label="5")
d = np.array([list(get2Dhist_k_alpha(d))+[p, np.median(d.strain)] for p, d in data.groupby("pressure")])
plt.sca(ax1)
plt.plot(d[:,2],d[:,0], "o-", label="200")
plt.sca(ax2)
plt.plot(d[:,3], d[:, 1], "o-", label="200")

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.30\700 pa",
])
data.pressure = np.round(data.pressure, 2)
#plot_pair(data, "pressure", ["C2", "C2"], label="700")
#plot_stiffness(data, "pressure", ["C0", "C0"], label="5")
d = np.array([list(get2Dhist_k_alpha(d))+[p] for p, d in data.groupby("pressure")])
d = np.array([list(get2Dhist_k_alpha(d))+[p, np.median(d.strain)] for p, d in data.groupby("pressure")])
plt.sca(ax1)
plt.plot(d[:,2],d[:,0], "o-",label="700")
plt.sca(ax2)
plt.plot(d[:,3], d[:, 1],"o-",label="700")
plt.legend()

plot_data(data, ax3, ax4)

#plot_afm()
#plot_data(data0[data0.pressure == 2])


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 5.300000/2.54, forward=True)
plt.figure(1).ax_dict["alpha"].set_xlim(0.0, 1.570166332156348)
plt.figure(1).ax_dict["alpha"].set_ylim(0.0, 0.7467686171617586)
plt.figure(1).ax_dict["alpha"].set_position([0.594337, 0.263846, 0.352273, 0.616154])
plt.figure(1).ax_dict["alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alpha"].get_xaxis().get_label().set_text("strain")
plt.figure(1).ax_dict["alpha"].get_yaxis().get_label().set_text("alpha")
plt.figure(1).ax_dict["stiffness"].set_yscale("log")
plt.figure(1).ax_dict["stiffness"].set_xlim(0.0, 2.095)
plt.figure(1).ax_dict["stiffness"].set_ylim(1.0, 1000.0)
plt.figure(1).ax_dict["stiffness"].set_yticks([1.0, 10.0, 100.0, 1000.0])
plt.figure(1).ax_dict["stiffness"].set_yticklabels(["$\mathdefault{10^{0}}$", "$\mathdefault{10^{1}}$", "$\mathdefault{10^{2}}$", "$\mathdefault{10^{3}}$"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["stiffness"].set_position([0.122881, 0.263846, 0.352273, 0.616154])
plt.figure(1).ax_dict["stiffness"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["stiffness"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["stiffness"].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).ax_dict["stiffness"].get_yaxis().get_label().set_text("stiffness (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
