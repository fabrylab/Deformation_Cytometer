import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar as fzero
import scipy.integrate
import scipy.optimize
from form_factors import getFormFactorFunctions

from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data
from deformationcytometer.includes.fit_velocity import fit_velocity, fit_velocity_pressures, getFitXY
from deformationcytometer.evaluation.helper_functions import plotDensityScatter

from RoscoeCoreInclude import getAlpha1, getAlpha2, getMu1, getEta1, eq41

def apply_velocity_fit(data2):
    p = [3, 0, 1 / 23]
    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
    p0, vel, vel_grad = fit_velocity_pressures(data2, config)
    eta0, delta, tau = p0
    eta = eta0 / (1 + tau ** delta * np.abs(vel_grad) ** delta)
    data2["vel"] = vel
    data2["vel_grad"] = vel_grad
    data2["eta"] = eta
    data = data2
    return p0

def getFitLine(pressure, p):
    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
    x, y = getFitXY(config, np.mean(pressure), p)
    return x, y

def histplot(x):
    x = np.array(x)
    from scipy import stats
    kde = stats.gaussian_kde(x)
    xx = np.linspace(np.min(x), np.max(x), 1000)
    plt.hist(x, bins=50, density=True)
    plt.plot(xx, kde(xx))


if 0:
    data, config = load_all_data([
            rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\1\*_result.txt",
    #    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\1\*_result.txt",
        ])
    data = data.dropna()
    #data0 = data
    data2 = data
if 1:
    import pandas as pd
    data2 = pd.read_csv("../figures/panel3_tanktreading/output_rotation_NIH.csv")
    data2["velocity"] = data2.v
    data2["long_axis"] = data2.a
    data2["short_axis"] = data2.b
    data2["angle"] = data2.beta
    data = data2

p = apply_velocity_fit(data)

plt.subplot(131)
plt.cla()
for pressure in data.pressure.unique():
    d = data[data.pressure == pressure]
    x, y = getFitLine(pressure, p)
    l, = plt.plot(x, y)
    plt.plot(d.rp*1e-6, d.velocity*1e-3, "o", color=l.get_color())
plt.xlabel("position in channel (m)")
plt.ylabel("velocity (m/s)")

plt.subplot(132)
plt.cla()
for pressure in data.pressure.unique():
    d = data[data.pressure == pressure]
    x, y = getFitLine(pressure, p)
    y = np.diff(y)
    x = x[0:-1]+np.diff(x)/2
    l, = plt.plot(x, y)
plt.xlabel("position in channel (m)")
plt.ylabel("strain rate (1/s)")

plt.subplot(133)
plt.cla()
for pressure in data.pressure.unique():
    d = data[data.pressure == pressure]

    l, = plt.plot(d.rp*1e-6, d.stress, "o")
plt.xlabel("position in channel (m)")
plt.ylabel("strain rate (1/s)")

"""
alpha1 = getAlpha1(data.long_axis/data.short_axis)
alpha2 = getAlpha2(data.long_axis/data.short_axis)

mu1 = getMu1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.stress)
eta1 = getEta1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.eta)

ttfreq = - eq41(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), np.abs(vel_grad))

Gp1 = mu1
Gp2 = eta1*np.abs(omega)
alpha_cell = np.arctan(Gp2/Gp1) * 2/np.pi
k_cell = Gp1 / (omega**alpha * scipy.special.gamma(1-alpha) * np.cos(np.pi/2 * alpha_cell))


plt.figure(1)
plt.subplot(121)
histplot(np.log(k_cell)[i])
plt.subplot(122)
histplot(alpha_cell[i])


plt.figure(2)
plotDensityScatter(np.log(k_cell)[i], alpha_cell[i])
"""