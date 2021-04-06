# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, load_all_data_new, plot_joint_density
import numpy as np

import pylustrator
pylustrator.start()

def darken(color, f, a=None):
    from matplotlib import colors
    c = np.array(colors.to_rgba(color))
    if f > 0:
        c2 = np.zeros(3)
    else:
        c2 = np.ones(3)
    c[:3] = c[:3] * (1 - np.abs(f)) + np.abs(f) * c2
    if a is not None:
        c[3] = a
    return c

def colormap(base_color, f1, f2, a1=None, a2=None):
    return pylustrator.LabColormap((darken(base_color, f1, a1), darken(base_color, f2, a2)), 255)

def plot_joint_density(x, y, label=None, only_kde=False, color=None, growx=1, growy=1, offsetx=0):
    ax = plt.gca()
    x1, y1, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height

    wf, hf = ax.figure.get_size_inches()
    gap = 0.05
    fraction = 0.2
    width_of_hist = np.mean([(w*wf*fraction), (h*hf*fraction)])
    hist_w = width_of_hist/wf
    hist_h = width_of_hist/hf

    h *= growy
    w *= growx
    if getattr(ax, "ax2", None) is None:
        ax.ax2 = plt.axes([x1 + offsetx*w, y1 + h - hist_h + gap/hf, w - hist_w, hist_h], sharex=ax, label=ax.get_label()+"_top")
        #ax.ax2.set_xticklabels([])
        ax.ax2.spines['right'].set_visible(False)
        ax.ax2.spines['top'].set_visible(False)
        ax.ax2.tick_params(axis='y', colors='none', which="both", labelcolor="none")
        ax.ax2.tick_params(axis='x', colors='none', which="both", labelcolor="none")
        ax.ax2.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.sca(ax.ax2)
    plot_density_hist(x, color=color, only_kde=only_kde)
    if getattr(ax, "ax3", None) is None:
        ax.ax3 = plt.axes([x1 + offsetx*w + w - hist_w + gap/wf, y1, hist_w, h - hist_h], sharey=ax, label=ax.get_label()+"_right")
        #ax.ax3.set_yticklabels([])
        ax.set_position([x1 + offsetx*w, y1, w - hist_w, h - hist_h])
        ax.ax3.spines['right'].set_visible(False)
        ax.ax3.spines['top'].set_visible(False)
        ax.ax3.tick_params(axis='x', colors='none', which="both", labelcolor="none")
        ax.ax3.tick_params(axis='y', colors='none', which="both", labelcolor="none")
        #ax.ax3.spines['left'].set_visible(False)
        ax.ax3.spines['bottom'].set_visible(False)

        ax.spines['right'].set_visible(False)
    plt.sca(ax.ax3)
    l = plot_density_hist(y, color=color, orientation=u'horizontal', only_kde=only_kde)
    plt.sca(ax)
    plotDensityScatter(x, y, cmap=colormap("C2", -0.5, 0), s=1)
    #plt.plot(x, y, "o", c=color, alpha=0.5, ms=1)
    #plotDensityLevels(x, y, levels=1, colors=[l.get_color()], cmap=None)
    plt.plot([], [], color=l.get_color(), label=label)

alg20 = [
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_27_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_28_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_30_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
]
[
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
     r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\*\*_result.txt",
    ]

numbers = []
data_list = []
config_list = []
for index, pressure in enumerate([1, 2, 3]):
    data, config = load_all_data_new(alg20, pressure=pressure)
    data_list.append(data)
    config_list.append(config)

ranges = [
    np.arange(10, 80, 20),
    np.arange(10, 180, 20),
    np.arange(10, 280, 20),
]
colors = [
    "#d68800",
    "#d40000",
    "#b00038",
]

for index, pressure in enumerate([3]):
    ax = plt.subplot(2, 3, index+1)
    data = data_list[index]
    config = config_list[index]
    from scipy.special import gamma
    def fit(omega, k, alpha):
        omega = np.array(omega)
        G = k * (1j * omega)**alpha * gamma(1 - alpha)
        return np.real(G), np.imag(G)
    def cost(p):
        Gp1, Gp2 = fit(data.omega, *p)
        return np.sum((np.log10(data.Gp1)-np.log10(Gp1))**2) + np.sum((np.log10(data.Gp2)-np.log10(Gp2))**2)
    from scipy.optimize import minimize
    res = minimize(cost, [80, 0.5], bounds=([0, np.inf], [0, 1]))
    print(res)

    #plt.loglog(data.omega, data.Gp1, "o", alpha=0.25, label="G'", ms=1)
    plt.loglog([], [], "o", alpha=0.25, label="G'", ms=1)
    plotDensityScatter(data.omega, data.Gp1, colormap("C0", 0, 0.5), alpha=0.25, s=1)

    plt.plot([1e-1, 1e0, 1e1, 3e1], fit([1e-1, 1e0, 1e1, 3e1], *res.x)[0], "-", color=darken("C0", 0.3), lw=0.8)

    #plt.loglog(data.omega, data.Gp2, "o", alpha=0.25, label="G''", ms=1)
    plt.loglog([], [], "o", alpha=0.25, label="G''", ms=1)
    plotDensityScatter(data.omega, data.Gp2, colormap("C1", -0.75, 0), alpha=0.25, s=1)
    plt.plot([1e-1, 1e0, 1e1, 3e1], fit([1e-1, 1e0, 1e1, 3e1], *res.x)[1], "C1-", color=darken("C1", 0.3), lw=0.8)
    plt.ylabel("G' / G''")
    plt.xlabel("angular frequency")
    plt.legend()

    ax = plt.subplot(2, 3, 2)
    plot_joint_density(np.log10(data.k_cell), data.alpha_cell, color="C2")#, color=c, label=f"{pressure} bar")
    plt.axhline(res.x[1], color="k", ls="--", lw=0.5)
    plt.axvline(np.log10(res.x[0]), color="k", ls="--", lw=0.5)
    plt.title(f"{pressure} bar")
    plt.xlim(1, 3)
    plt.ylim(0, 1)
    plt.xlabel("k")
    plt.ylabel("$\\alpha$")

    numbers.append(len(data.rp))
    print(config)

print("numbers", numbers)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(11.000000/2.54, 8.000000/2.54, forward=True)
plt.figure(1).ax_dict["_right"].set_position([0.851678, 0.290569, 0.071372, 0.429595])
plt.figure(1).ax_dict["_top"].set_position([0.579966, 0.735887, 0.260343, 0.098705])
plt.figure(1).axes[0].set_xlim(0.1, 100.0)
plt.figure(1).axes[0].set_ylim(1.0, 1000.0)
plt.figure(1).axes[0].legend(handlelength=1.0, markerscale=3.0, fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[0].set_position([0.143253, 0.290569, 0.331715, 0.528300])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend()._set_loc((0.631390, 0.047891))
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.286730, 0.952117])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[1].set_xlim(1.0, 3.0)
plt.figure(1).axes[1].set_xticks([1.0, 2.0, 3.0])
plt.figure(1).axes[1].set_xticklabels(["$10^1$", "$10^2$", "$10^3$"])
plt.figure(1).axes[1].set_position([0.579966, 0.290569, 0.260343, 0.429595])
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.449183, 1.055997])
plt.figure(1).axes[1].texts[0].set_text("b")
plt.figure(1).axes[1].texts[0].set_weight("bold")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


