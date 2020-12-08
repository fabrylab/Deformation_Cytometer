import numpy as np
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import getConfig, getData, load_all_data, plotBinnedData
from pathlib import Path
import pandas as pd
#data1 = np.loadtxt(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_15_alginate2%_NIH_tanktreading_1\2\2020_09_15_10_35_15\speeds.txt")

from deformationcytometer.includes.includes import getInputFile
from deformationcytometer.evaluation.helper_functions import getConfig, getData, getVelocity, correctCenter
from deformationcytometer.evaluation.helper_functions import fit_func_velocity

def removeNan(x, y):
    indices = ~np.isnan(x) & ~np.isnan(y)
    return x[indices], y[indices]

def getXY(pressure):
    data, config = load_all_data(
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\1\*_result.txt",
        pressure=pressure)
    x = np.abs(data.rp) * 1e-6
    y = data.velocity * 1e-3
    return removeNan(x, y)

data, config = load_all_data(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\1\*_result.txt", pressure=3)

data = data[np.abs(data.velocity-data.velocity_fitted) < 1.5]

plt.subplot(121)
plt.plot(data.rp, data.velocity, "o")
vel = fit_func_velocity(config)
dx = 1e-6
x = np.arange(-config["channel_width_m"]/2, config["channel_width_m"]/2, dx)
v = vel(x*1e6)*1e-3
plt.plot(x*1e6, v*1e3, "k--")
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("channel position (µm)")
plt.ylabel("velocity (mm/s)")

plt.subplot(122)
grad = np.diff(v)/np.diff(x)# * 1e3
#plt.plot(data.rp, data.velocity_gradient, "o")
plt.plot((x[:-1]+0.5*np.diff(x))*1e6, grad, "k--")
plt.xlabel("channel position (µm)")
plt.ylabel("share rate $\dot \gamma$ (1/s)")
#plt.plot(data.rp*1e-6, getVelGrad(data.rp), "s")
#plt.plot(x[:-1]+0.5*np.diff(x), grad, "-+")

plt.show()

plt.clf()
plt.subplot(121)
x = np.abs(data.rp)*1e-6
y = data.velocity*1e-3
#plt.plot(x, y, "o")

if 1: # Euler
    H = 200e-6
    W = 200e-6
    n = np.arange(1, 99, 2)[:, None]
    tau = 1 / 20
    alpha = 0.67
    eta0 = 3.65
    P = 3 * 100000
    L = 58.5e-3
    pi = np.pi

    x1, y1 = getXY(1)
    x2, y2 = getXY(2)
    x3, y3 = getXY(3)

    def getVelocity(y, eta0, alpha, tau, P, W, H, L):
        def euler(t, x0, y0, f):
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            x[0] = x0
            y[0] = y0
            dt = np.diff(t)
            for i, ti in enumerate(t[:-1]):
                y[i + 1] = f(ti, x[i])
                x[i + 1] = x[i] + y[i + 1] * dt[i]
            return t, x, y

        def getVDot(y, v):
            return 1 / tau * (np.abs(v * eta0 / getBeta(y) - 1)) ** (1 / alpha)

        def getBeta(y):
            return -(4 * (H ** 2) * P) / (L * (pi ** 3)) * np.sum(
                (-1) ** ((n - 1) / 2) / (n ** 3) * (1 - np.cosh((n * pi * y) / H) / np.cosh((n * pi * W) / (2 * H))),
                axis=0)

        t, v, vdot = euler(y, getBeta(0)/eta0, 0, getVDot)
        return t, -v, vdot

    import scipy.interpolate

    if 0:
        def getCostP(i, eta0, alpha, tau, W, H):
            yy = np.arange(0, H / 2, 1e-6)  # [:100]
            t, v, vdot = getVelocity(yy, eta0, alpha, tau, i*1e5, W, H, L)
            return np.sum(np.abs(scipy.interpolate.interp1d(t, v)(xl[i])-yl[i]))

        def getCost(p):
            print(p)
            return getCostP(1, p[0], p[3], p[4], p[5], p[5]) \
                 + getCostP(2, p[1], p[3], p[4], p[5], p[5]) \
                 + getCostP(3, p[2], p[3], p[4], p[5], p[5])

        res = scipy.optimize.minimize(getCost, [eta0, eta0, eta0, alpha, tau, W], bounds=[(0, None), (0, None), (0, None), (0, 1), (0, None), (0, None)])

    def curve(y, eta0, alpha, tau, W2):
        #W, H = W2, W2
        yy = np.arange(0, H / 2, 0.1e-6)  # [:100]
        t, v, vdot = getVelocity(yy, eta0, alpha, tau, P, W, H, L)
        return scipy.interpolate.interp1d(t, v)(y)


    def curve2(y):
        #W, H = W2, W2
        yy = np.arange(0, H / 2, 0.1e-6)  # [:100]
        t, v, vdot = getVelocity(yy, eta0, alpha, tau, P, W, H, L)
        return scipy.interpolate.interp1d(t, v)(y), scipy.interpolate.interp1d(t, vdot)(y)


    W = 190e-6
    H = 190e-6
    eta0 = 0.6
    alpha = 0.7
    tau = 1/2000
    xl = [0,x1, x2, x3]
    yl = [0,y1, y2, y3]
    for i in [1,2,3]:
        P = i*1e5
        p, popt = scipy.optimize.curve_fit(curve, xl[i], yl[i], [eta0, alpha, tau, W])
        print(i, p)

        eta0, alpha, tau, W2 = p
        yy = np.arange(0, W / 2, 0.1e-6)  # [:100]
        plt.plot(xl[i], yl[i], "o")
        t, v, vdot = getVelocity(yy, eta0, alpha, tau, P, W, W, L)
        plt.plot(t, v, "-")

    p, popt = scipy.optimize.curve_fit(curve, x3, y3, [eta0, alpha, tau, W])

    plt.clf()
    plt.subplot(121)
    #plt.plot(x1, y1, "o")
    #plt.plot(x2, y2, "o")
    plt.plot(x3, y3, "o")
    eta0, alpha, tau, W = p
    yy = np.arange(0, W / 2, 0.1e-6)  # [:100]
    if 0:
        t, v, vdot = getVelocity(yy, eta0, alpha, tau, 1e5, W, W, L)
        plt.plot(t, v, "-")

        t, v, vdot = getVelocity(yy, eta0, alpha, tau, 2e5, W, W, L)
        plt.plot(t, v, "-")

    t, v, vdot = getVelocity(yy, eta0, alpha, tau, 3e5, W, W, L)
    plt.plot(t, v, "-")

    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    plt.subplot(122)

    #tau = 1 / 20
    #alpha = 0.67
    #eta0 = 3.65
    plt.plot(t, eta0/(1+(tau*vdot)**alpha))
    plt.loglog([], [])
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e6)))
    plt.show()

if 0:
    import scipy.optimize

    def curve(x, x0, a1, a2):
        return x0 * (1 - (a1*(x/95e-6)**2 - (a2*(x/95e-6))**4))

    def derivative(x, x0, d1, a1, d2, a2):
        return -((d1 * (a1 * x) ** d1 + d2*(a2*x) ** d2) * x0) / x


    #def curve(x, x0, delta, delta2):
    #    return x0 * ((2) - (x)**delta - (x)**delta2)

    p, popt = scipy.optimize.curve_fit(curve, x, y, [25e-3, 1, 1/95e-6, 2, 1/95e-6])
    print(p)
    xx = np.linspace(0, 95e-6, 100)
    plt.plot(xx, curve(xx, *p))

    plt.subplot(122)
    yy = np.diff(curve(xx, *p)) / np.diff(xx)
    plt.plot(xx[:-1]+np.diff(xx)/2, yy)
    plt.plot(xx, derivative(xx, *p))

    plt.plot((x[:-1]+0.5*np.diff(x)), grad, "k--")

    plt.plot(xx, curve(xx, *[25e-3, 1, 1/95e-6, 2, 1/95e-6]))


    def stressfunc(R, P, L, H):  # imputs (radial position and pressure)
        G = P / L  # pressure gradient
        pre_factor = (4 * (H ** 2) * G) / np.pi ** 3
        # sum only over odd numbers
        n = np.arange(1, 100, 2)[None, :]
        u_primy = pre_factor * np.sum(((-1) ** ((n - 1) / 2)) * (np.pi / ((n ** 2) * H)) \
                         * (np.sinh((n * np.pi * R[:, None]) / H) / np.cosh(n * np.pi / 2)), axis=1)

        stress = np.abs(u_primy)
        return stress

    #def eta():
    #    return eta0 / (1-)