from deformationcytometer.includes.includes import getInputFile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData
from deformationcytometer.includes.fit_velocity import fit_velocity, fit_velocity_pressures, getFitXY, getFitXYDot
settings_name = "strain_vs_stress_clean"
""" loading data """

import pylustrator
pylustrator.start()

def plot_velocity_fit(data, color=None):
    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXY(config, np.mean(pressure), p)
        return x, y

    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        d = d.set_index(["eta0", "delta", "tau"])
        for p in d.index.unique():
            dd = d.loc[p]
            x, y = getFitLine(pressure, p)
            line, = plt.plot(np.abs(dd.rp), dd.velocity * 1e-3 * 1e2, "o", alpha=0.3, ms=2, color=color)
            plt.plot([], [], "o", ms=2, color=line.get_color(), label=f"{pressure:.1f}")
            l, = plt.plot(x[x>=0]* 1e+6, y[x>=0] * 1e2, color="k")
    plt.xlabel("position in channel (µm)")
    plt.ylabel("velocity (cm/s)")

def plot_velocity_fit_dot(data, color=None):
    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXYDot(config, np.mean(pressure), p)
        return x, y

    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        d = d.set_index(["eta0", "delta", "tau"])
        for p in d.index.unique():
            dd = d.loc[p]
            x, y = getFitLine(pressure, p)
            #line, = plt.plot(np.abs(dd.rp), dd.velocity * 1e-3 * 1e2, "o", alpha=0.3, ms=2, color=color)
            #plt.plot([], [], "o", ms=2, color=line.get_color(), label=f"{pressure:.1f}")
            l, = plt.plot(x[x>=0]* 1e+6, -y[x>=0], color=color)
    plt.xlabel("position in channel (µm)")
    plt.ylabel("shear rate (1/s)")

def plot_viscisity(data, color=None):
    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXYDot(config, np.mean(pressure), p)
        return x, y

    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        d = d.set_index(["eta0", "delta", "tau"])
        for p in d.index.unique():
            dd = d.loc[p]
            x, y = getFitLine(pressure, p)
            eta0 = data.iloc[0].eta0
            delta = data.iloc[0].delta
            tau = data.iloc[0].tau
            vel_grad = y
            eta = eta0 / (1 + tau ** delta * np.abs(vel_grad) ** delta)

            #line, = plt.plot(np.abs(dd.rp), dd.velocity * 1e-3 * 1e2, "o", alpha=0.3, ms=2, color=color)
            #plt.plot([], [], "o", ms=2, color=line.get_color(), label=f"{pressure:.1f}")
            l, = plt.plot(x[x>=0]* 1e+6, eta[x>=0], color=color)
    plt.xlabel("position in channel (µm)")
    plt.ylabel("viscosity (Pa s)")

def plot_omega(data, color=None):
    #omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        w = d.omega#omega[data.pressure == pressure]
        plt.plot(-d.vel_grad, w, "o", label=f"{pressure:.1f}", ms=1)
    plt.axline((0,0), slope=0.5, linestyle="dashed", color="k", lw=0.8)
    plt.xlabel("shear rate (1/s)")
    plt.ylabel("tank treading\nangular frequency (1/s)")

def plot_viscisoty_over_shear_rate(data):
    eta0 = data.iloc[0].eta0
    delta = data.iloc[0].delta
    tau = data.iloc[0].tau
    vel_grad = np.geomspace(0.01, 1100, 1000)
    eta = eta0 / (1 + tau ** delta * np.abs(vel_grad) ** delta)
    plt.plot(vel_grad, eta)
    plt.xlabel("shear rate (1/s)")
    plt.ylabel("viscosity (Pa s)")
    plt.loglog()

def plot_tt(ax1, ax2):
    import tifffile
    import skimage.registration
    import scipy.special
    import pandas as pd
    import imageio
    from pathlib import Path

    target_folder = Path(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\1\2020_09_16_14_32_59")
    data = pd.read_csv(target_folder / "output.csv")
    data2 = np.loadtxt(target_folder / "speeds_new.txt")
    # {cell_id} {data0.grad.mean()/(2*np.pi)} {speed} {data0.rp.mean()} {r2}
    print(np.max(data2[:, 4]))
    cell_id = int(data2[data2[:, 4] > 0.6, 0][1])
    factor_invert = -1
    #cell_id = int(data[data.rp > 50e-6].iloc[20].id)
    #cell_id = 781
    pixel_size = 0.34500000000000003

    data0 = data[data.id == cell_id]

    def getPerimeter(a, b):
        from scipy.special import ellipe

        # eccentricity squared
        e_sq = 1.0 - b ** 2 / a ** 2
        # circumference formula
        perimeter = 4 * a * ellipe(e_sq)

        return perimeter

    def getEllipseArcSegment(angle, a, b):
        e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
        perimeter = scipy.special.ellipeinc(2.0 * np.pi, e)
        return scipy.special.ellipeinc(angle, e) / perimeter * 2 * np.pi  # - sp.special.ellipeinc(angle-0.1, e)

    def getArcLength(points, major_axis, minor_axis, ellipse_angle, center):
        p = points - np.array(center)  # [None, None]
        alpha = np.deg2rad(ellipse_angle)
        p = p @ np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

        distance_from_center = np.linalg.norm(p, axis=-1)
        angle = np.arctan2(p[..., 0], p[..., 1])
        angle = np.arctan2(np.sin(angle) / (major_axis / 2), np.cos(angle) / (minor_axis / 2))
        angle = np.unwrap(angle)

        r = np.linalg.norm([major_axis / 2 * np.sin(angle), minor_axis / 2 * np.cos(angle)], axis=0)

        length = getEllipseArcSegment(angle, minor_axis / 2, major_axis / 2)
        return length, distance_from_center / r

    def getImageStack(cell_id):
        return np.array([im for im in imageio.get_reader(target_folder / f"{cell_id:05}.tif")])

    def getMask(d1, im):
        rr, cc = skimage.draw.ellipse(40, 60, d1.short_axis / pixel_size / 2, d1.long_axis / pixel_size / 2, im.shape,
                                      -d1.angle * np.pi / 180)
        mask = np.zeros(im.shape, dtype="bool")
        mask[rr, cc] = 1
        return mask

    def getCenterLine(x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        def func(x, m):
            return x * m

        import scipy.optimize
        p, popt = scipy.optimize.curve_fit(func, x, y, [1])
        return p[0], 0

    images = getImageStack(cell_id)
    times = np.array((data0.timestamp - data0.iloc[0].timestamp) * 1e-3)

    perimeter_pixels = getPerimeter(data0.long_axis.mean() / pixel_size / 2, data0.short_axis.mean() / pixel_size / 2)
    mask = getMask(data0.iloc[0], images[0])


    for i in range(len(images) - 1):
        dt = times[i + 1] - times[i]
        flow = skimage.registration.optical_flow_tvl1(images[i], images[i + 1], attachment=30)

        x, y = np.meshgrid(np.arange(flow[0].shape[1]), np.arange(flow[0].shape[0]), sparse=False, indexing='xy')
        x = x.flatten()
        y = y.flatten()
        flow = flow.reshape(2, -1)

        ox, oy = [60, 40]
        distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
        projected_speed = ((x - ox) * flow[0] - (y - oy) * flow[1]) / distance

        angle, distance_to_center = getArcLength(np.array([x, y]).T, data0.long_axis.mean() / pixel_size,
                                                 data0.short_axis.mean() / pixel_size,
                                                 data0.angle.mean(), [ox, oy])

        indices_middle = (distance_to_center < 1) & ~np.isnan(projected_speed)

        #data_x.extend(distance_to_center[indices_middle])
        #data_y.extend(projected_speed[indices_middle] / dt / perimeter_pixels)

        if i == 0:
            inside = mask[y, x]
            plt.sca(ax1)
            plt.imshow(images[i], cmap="gray")
            factor = 2
            skip = 3
            for i in np.arange(len(x))[inside]:
                if x[i] % skip == 0 and y[i] % skip == 0:
                    plt.plot([x[i], x[i] + flow[1][i]*factor], [y[i], y[i] + flow[0][i]*factor], "r", lw=0.8)
                #plt.plot([x[i]], [y[i]], "ro")
            #plt.axis("equal")
            if factor_invert == -1:
                plt.ylim(0, images[0].shape[0])
            plt.xticks([])
            plt.yticks([])

            plt.sca(ax2)
            x = distance_to_center[indices_middle]
            y = projected_speed[indices_middle] / dt / perimeter_pixels
            y = y*factor_invert
            plt.plot(x, y, "o", color="C3", ms=1, alpha=0.5)
            m, t = getCenterLine(x, y)
            x2 = np.linspace(0, max(x), 2)
            plt.plot(x2, x2*m, "-k")
            plt.axvline(0.7, linestyle="dashed", lw=0.8, color="k")
            plt.xlabel("relative radius")
            plt.ylabel("speed (µm/s)")
            break

    #m, t = getCenterLine(data_x, data_y)

    #cr = np.corrcoef(data_y, m * np.array(data_x))
    #r2 = np.corrcoef(data_y, m * np.array(data_x))[0, 1] ** 2

    #return m, r2

# load the data and the config
data, config = load_all_data(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\1\*_result.txt")

data = data[np.abs(data.velocity-data.velocity_fitted) < 1.5]

plt.subplot(241)
plot_velocity_fit(data)
#plt.legend(title="pressure (bar)")
plt.subplot(242)
plot_velocity_fit_dot(data)
plt.subplot(243)
plot_viscisity(data)
plt.subplot(244)
plot_viscisoty_over_shear_rate(data)

plot_tt(plt.subplot(245), plt.subplot(246))

plt.subplot(247)
plot_omega(data)
plt.legend(title="pressure (bar)")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.220000/2.54, 8.600000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(-4.5522522522522415, 100.0)
plt.figure(1).axes[0].set_xticks([0.0, 25.0, 50.0, 75.0, 100.0])
plt.figure(1).axes[0].set_xticklabels(["0", "25", "50", "75", "100"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.060498, 0.660336, 0.176478, 0.310475])
plt.figure(1).axes[0].set_zorder(1)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_fontname("Arial")
plt.figure(1).axes[0].texts[0].set_position([-0.210795, 0.970452])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([0.983853, 0.970452])
plt.figure(1).axes[0].texts[1].set_text("b")
plt.figure(1).axes[0].texts[1].set_weight("bold")
plt.figure(1).axes[1].set_xlim(-4.5522522522522415, 100.0)
plt.figure(1).axes[1].set_ylim(-55.86716058117165, 1173.4714408394984)
plt.figure(1).axes[1].set_xticks([0.0, 25.0, 50.0, 75.0, 100.0])
plt.figure(1).axes[1].set_yticks([0.0, 500.0, 1000.0])
plt.figure(1).axes[1].set_xticklabels(["0", "25", "50", "75", "100"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_yticklabels(["0k", ".5k", "1k"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_position([0.309705, 0.660336, 0.176478, 0.310475])
plt.figure(1).axes[1].set_zorder(1)
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].yaxis.labelpad = -0.769022
plt.figure(1).axes[2].set_xlim(-4.5522522522522415, 100.0)
plt.figure(1).axes[2].set_ylim(-0.3, 3.713736076371506)
plt.figure(1).axes[2].set_xticks([0.0, 25.0, 50.0, 75.0, 100.0])
plt.figure(1).axes[2].set_yticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[2].set_xticklabels(["0", "25", "50", "75", "100"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_yticklabels(["0", "1", "2", "3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[2].set_position([0.558913, 0.660476, 0.176478, 0.310475])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.242285, 0.970000])
plt.figure(1).axes[2].texts[0].set_text("c")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[3].set_xlim(0.01, 1000.0)
plt.figure(1).axes[3].set_ylim(0.3, 3.869813081501242)
plt.figure(1).axes[3].set_xticks([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
plt.figure(1).axes[3].set_yticks([0.3, 1.0, 3.0])
plt.figure(1).axes[3].set_xticklabels([".01", ".1", "1", "10", "100", "1k"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[3].set_yticklabels([".3", "1.", "3."], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[3].set_position([0.808120, 0.660336, 0.176478, 0.310334])
plt.figure(1).axes[3].set_yticks([np.nan], minor=True)
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([-0.257642, 0.970890])
plt.figure(1).axes[3].texts[0].set_text("d")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[4].set_xlim(20.0, 100.0)
plt.figure(1).axes[4].set_ylim(10.0, 70.0)
plt.figure(1).axes[4].set_position([0.020324, 0.161556, 0.221903, 0.314441])
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_position([-0.076818, 0.972343])
plt.figure(1).axes[4].texts[0].set_text("e")
plt.figure(1).axes[4].texts[0].set_weight("bold")
plt.figure(1).axes[5].set_position([0.332296, 0.137565, 0.221903, 0.362423])
plt.figure(1).axes[5].set_zorder(1)
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
#plt.figure(1).axes[5].lines[0].set_markeredgecolor("#1f77b40a")
#plt.figure(1).axes[5].lines[0].set_markerfacecolor("#1f77b40a")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.286155, 0.972343])
plt.figure(1).axes[5].texts[0].set_text("f")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[6].set_ylim(-5.223895317219428, 90.0)
plt.figure(1).axes[6].legend(handlelength=1.1, handletextpad=0.0, columnspacing=0.30000000000000004, ncol=3, title="pressure (bar)", fontsize=8.0, title_fontsize=8.0)
plt.figure(1).axes[6].set_position([0.655757, 0.137565, 0.295029, 0.362423])
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[6].get_legend()._set_loc((0.700119, 0.043286))
plt.figure(1).axes[6].get_legend()._set_loc((0.496545, 0.034610))
plt.figure(1).axes[6].get_legend()._set_loc((0.621234, 0.030272))
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].texts[0].set_position([-0.320562, 0.972343])
plt.figure(1).axes[6].texts[0].set_text("g")
plt.figure(1).axes[6].texts[0].set_weight("bold")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[1].new
plt.figure(1).axes[6].texts[1].set_position([0.182483, 0.849579])
plt.figure(1).axes[6].texts[1].set_rotation(67.0)
plt.figure(1).axes[6].texts[1].set_text("0.5")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()
