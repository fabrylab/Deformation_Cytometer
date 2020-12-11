import numpy as np
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import getConfig, getData, load_all_data, plotBinnedData
from pathlib import Path
import pandas as pd
import sys
#data1 = np.loadtxt(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_15_alginate2%_NIH_tanktreading_1\2\2020_09_15_10_35_15\speeds.txt")

from deformationcytometer.includes.includes import getInputFile
from deformationcytometer.evaluation.helper_functions import getConfig, getData, getVelocity, correctCenter
from deformationcytometer.evaluation.helper_functions import fit_func_velocity, getStressStrain

def getTorque(a, b, beta, count=120):
    phi2 = angles_in_ellipse(count, b, a, 0)
    return np.mean((-(a*np.sin(phi2)*np.sin(beta)) + (b*np.cos(phi2)*np.cos(beta)))**2)

def getPerimeter(a, b):
    from scipy.special import ellipe

    # eccentricity squared
    e_sq = 1.0 - b ** 2 / a ** 2
    # circumference formula
    perimeter = 4 * a * ellipe(e_sq)

    return perimeter

def angles_in_ellipse(num, a, b, offset):
    import scipy.special
    import scipy.optimize
    assert(num > 0)
    assert(a <= b)
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
        tot_size = scipy.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size + offset
        res = scipy.optimize.root(lambda x: (scipy.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
    return angles

def getPointsOnEllipse(a, b, beta):
    phi2 = angles_in_ellipse(18, b, a, 0)

    x = np.sin(phi2) * a
    y = np.cos(phi2) * b

    x, y = (np.array([x, y]).T @ np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])).T
    return np.array([x, y]).T

def removeNan(x, y):
    indices = ~np.isnan(x) & ~np.isnan(y)
    return x[indices], y[indices]

def dataWithGradient(filename):
    # %%
    config = getConfig(filename)
    config["channel_width_m"] = 0.00019001261833616293

    data = getData(filename)
    getVelocity(data, config)

    # take the mean of all values of each cell
    # data = data.groupby(['cell_id']).mean()

    correctCenter(data, config)
    # exit()

    data = data[(data.solidity > 0.96) & (data.irregularity < 1.06)]
    # data = data[(data.solidity > 0.98) & (data.irregularity < 1.02)]
    data.reset_index(drop=True, inplace=True)

    getStressStrain(data, config)

    def getVelGrad(r):
        p0, p1, p2 = config["vel_fit"]
        r = r
        p0 = p0 * 1e3
        r0 = config["channel_width_m"] * 0.5 * 1e6  # 100e-6
        return - (p1 * p0 * (np.abs(r) / r0) ** p1) / r

    vel = fit_func_velocity(config)

    import scipy.optimize

    def curve(x, x0, delta1, a1, delta2, a2, w):
        x = np.abs(x)
        delta1 = 2
        return x0 * ((a1+a2) - a1 * (x) ** delta1 - (a2 * x) ** delta2)

    def derivative(x, x0, d1, a1, d2, a2):
        d1 = 2
        return -((d1 * (a1 * np.abs(x)) ** d1 + d2 * (a2 * np.abs(x)) ** d2) * x0) / x

    # def curve(x, x0, delta, delta2):
    #    return x0 * ((2) - (x)**delta - (x)**delta2)
    return data

    x = data.rp * 1e-6
    y = data.velocity * 1e-3
    x, y = removeNan(x, y)
    p, popt = scipy.optimize.curve_fit(curve, x, y, [1.00512197e-02, 1.00000000e+00, 9.15342978e+03, 4.85959006e+00,
 1.15235584e+04])#[25e-3, 1, 1 / 95e-6, 2, 1 / 95e-6])
    getVelGrad = lambda x: derivative(x*1e-6, *p)
    vel = lambda x: curve(x*1e-6, *p) * 1e3

    data["grad"] = getVelGrad(data.rp)#/(2*np.pi)
    dx = data.short_axis/2
    #data["grad"] = (vel(data.rp+dx)-vel(data.rp))*1e-3/(dx*1e-6)
    if 1:
        print("vel fit", p)
        plt.figure(10)
        plt.subplot(131)
        plt.plot(data.rp * 1e-6, data.velocity * 1e-3, "o")
        dx = 1
        x = np.arange(-100, 100, dx) * 1e-6
        v = vel(x * 1e6) * 1e-3
        plt.plot(x, v, "r+")
        plt.axhline(0, color="k", lw=0.8)

        plt.subplot(132)
        grad = np.diff(v) / np.diff(x)  # * 1e3
        plt.plot(data.rp * 1e-6, data.velocity_gradient, "o")
        plt.plot(data.rp * 1e-6, getVelGrad(data.rp), "s")
        plt.plot(x[:-1] + 0.5 * np.diff(x), grad, "-+")
        plt.show()
    return data

if 0:
    for a in np.arange(1, 2, 0.1):
        b = 1/a
        c = getPerimeter(a, b)
        e = (a-b)/np.sqrt(a*b)
        plt.plot(e, b, "o")
        #plt.plot(e, c/(np.pi*2), "o")
    plt.show()

def plotFiles(files, label=""):
    files = list(files)
    print(files)

    xl = []
    yl = []
    
    slopes = []
    pressures = []

    plt.subplot(131)
    header = None
    with open(f"output_rotation_{label}.csv", "w") as fp:
        for file in files[::-1]:
            try:
                data2 = np.loadtxt(str(file)[:-4] + "\speeds_new.txt")
            except OSError as err:
                print(err, file=sys.stderr)
                continue
            print(str(file)[:-4] + "\speeds_new.txt")
            #print(data2)

            #print("load data")
            try:
                data = pd.read_csv(str(file)[:-4] + "/output.csv")
            except pd.errors.EmptyDataError:
                continue

            data00 = dataWithGradient(str(file)[:-4]+".tif")

            data2 = data2[data2[:, 4] > 0.2]

            r = []
            b = []
            a = []
            c = []
            beta = []
            stress = []
            stress_center = []
            #grad = []
            v = []
            rp = []
            print(data.columns)
            for cell in data2:
                data0 = data[data.id == cell[0]]
                data00_cell = data00[data00.cell_id == cell[0]]

                perimeter_pixels = getPerimeter(data0.long_axis.mean() / 2,
                                                data0.short_axis.mean() / 2)
                r.append(np.sqrt(data0.long_axis.mean() / 2 * data0.short_axis.mean() / 2))
                b.append(data0.short_axis.mean() / 2)
                a.append(data0.long_axis.mean() / 2)
                beta.append(data0.angle.mean())
                c.append(perimeter_pixels)
                stress.append(data00_cell.stress.mean())
                stress_center.append(data00_cell.stress_center.mean())
                rp.append(data00_cell.rp.mean())
                v.append(data0.velocity.mean())

                #grad.append(data00_cell.grad.mean())#/(2*np.pi))

                points = getPointsOnEllipse(data0.long_axis.mean() / 2, data0.short_axis.mean() / 2, data0.angle.mean())

                #data[:, 1]*2*np.pi*points[:, 1]
                #plt.plot(points[:, 0], points[:, 1], "-o")
                #plt.show()
                #print(2*np.pi*r[-1],c[-1], cell[0])

            print(data2.shape, len(c), len(r))
            b = np.array(b)
            a = np.array(a)
            r = np.array(r)
            c = np.array(c)
            v = np.array(v)
            rp = np.array(rp)
            stress = np.array(stress)
            #grad = np.array(grad)

            config = getConfig(file)
            pressure = config["pressure_pa"] / 100_000

            if header is None:
                fp.write("pressure,id,freq,a,b,r,c,beta,stress,stress_center,v,rp\n")
                header = True
            for i in range(len(data2)):
                fp.write(f"{pressure},{data2[i, 0]},{data2[i, 2]},{a[i]},{b[i]},{r[i]},{c[i]},{beta[i]},{stress[i]},{stress_center[i]},{v[i]},{rp[i]}\n")

            continue
            # print(data1.shape)
            print(data2.shape)
            # plt.plot(data1[:, 1], data1[:, 2])
            #plt.plot(data2[:, 1], data2[:, 2], "o", label=f"{pressure:.2f}")

            #plt.plot(data2[:, 1]*1e3, data2[:, 2], "o")
            #plt.plot(grad, data2[:, 2], "o")
            #continue


            strain = (a-b)/np.sqrt(a*b)
            ben_guess = 0.5 * data2[:, 1] * 0.5#1/(strain+1)
            #plt.plot(grad, data2[:, 2], "o", label=f"{pressure:.2f}")
            x = strain#data2[:, 1]*2*np.pi
            y = ben_guess#data2[:, 2]
            y = -data2[:, 2] / grad  # /(b*2*np.pi/c)
            plt.plot(x, y, "o", label=f"{pressure:.2f}")
            plt.subplot(132)
            plt.plot(grad, data2[:, 2], "o", label=f"{pressure:.2f}")

            def getCenterLine(x, y):
                x = np.asarray(x)
                y = np.asarray(y)

                def func(x, m):
                    return x * m

                import scipy.optimize
                x, y = removeNan(x, y)
                p, popt = scipy.optimize.curve_fit(func, x, y, [1])
                return p[0], 0

            try:
                m, t = getCenterLine(grad, data2[:, 2])
            except ValueError:
                continue
            x = np.linspace(np.min(grad), np.max(grad), 10)
            plt.plot(x, m * x, "-k")
            print(pressure, m)

            slopes.append(m)
            pressures.append(pressure)

            plt.subplot(131)

    plt.subplot(133)
    plt.plot(pressures, slopes, "o-")

    plt.subplot(132)
    #plt.plot(xl, yl, "-o", label=label)
    plt.subplot(131)


def getCenterLine(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    def func(x, m):
        return x * m

    import scipy.optimize
    x, y = removeNan(x, y)
    p, popt = scipy.optimize.curve_fit(func, x, y, [1])
    return p[0], 0


#files = plotFiles(Path(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\1").glob("*.tif"), label="NIH")
files = plotFiles(Path(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_15_alginate2%_NIH_tanktreading_1").glob("[1-9]\*.tif"), label="NIH2")
#files = plotFiles(Path(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_25_alginate2%_THP1_tanktreading").glob("[0-5]\*.tif"), label="THP1")
exit()
data = pd.read_csv("output_rotation_THP1.csv")
cols = 2

slopes = []
pressures = []

xl = []
yl = []
xl2 = []
yl2 = []
for pressure in data.pressure.unique():
    data0 = data[data.pressure == pressure]

    plt.subplot(1,cols,1)
    p, = plt.plot(data0.grad, -data0.freq, "o", label=f"{pressure:.2f}")

    plt.xlabel("shear rate $\dot \gamma$ (1/s)")
    plt.ylabel("rotation\nfrequency $f$ (1/s)")
    plt.legend()

    m, t = getCenterLine(data0.grad, data0.freq)
    x = np.linspace(np.min(data0.grad), np.max(data0.grad), 10)
    plt.plot(x, -m * x, "-k")

    slopes.append(m)
    pressures.append(pressure)

    plt.subplot(1,cols,2)
    strain = (data0.a - data0.b) / np.sqrt(data0.a * data0.b)
    x = data0.b/data0.r#data0.b/data0.r * np.pi*2 * data0.r / data0.c

    print("data0.beta.loc[i]", data0.beta)
    xx = np.array([getTorque(data0.a.loc[i]/data0.r.loc[i], data0.b.loc[i]/data0.r.loc[i], data0.beta.loc[i]/180*np.pi) for i in data0.index])

    individual_slope = -data0.freq / data0.grad
    p, = plt.plot(x, individual_slope, "o", ms=2)
    #plotBinnedData(x, individual_slope, np.arange(0, 1, 0.1), color=p.get_color())
    plt.xlabel("$\\frac{b}{r_0}$")
    plt.ylabel("slope $\\frac{f}{\dot \gamma}$")
    plt.ylim(0, 0.1)
    plt.xlim(0.6, 1)
    xl.extend(x)
    yl.extend(individual_slope)


xf, yf = plotBinnedData(xl, yl, np.arange(0, 1, 0.05))
index = ~np.isnan(xf) & ~np.isnan(yf)
xf = xf[index]
yf = yf[index]


def curve(x, a):
    return 1/(np.pi*4) * x **a

import scipy.optimize
if 0:
    p, popt = scipy.optimize.curve_fit(curve, xf, yf)
    x = np.arange(0.5, 1, 0.01)
    plt.plot(x, curve(x, *p), "-k")

plt.show()
