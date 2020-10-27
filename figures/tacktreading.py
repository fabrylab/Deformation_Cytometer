from qtpy import QtCore
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import skimage.draw
import imageio
import re
from pathlib import Path
import pylustrator
pylustrator.start()

from deformationcytometer.evaluation.helper_functions import getStressStrain, getConfig

import scipy.special


def getEllipseArcSegment(angle, a, b):
    e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
    perimeter = scipy.special.ellipeinc(2.0 * np.pi, e)
    return scipy.special.ellipeinc(angle, e)/perimeter*2*np.pi# - sp.special.ellipeinc(angle-0.1, e)

def getArcLength(points, major_axis, minor_axis, ellipse_angle, center):
    p = points - np.array(center)#[None, None]
    alpha = np.deg2rad(ellipse_angle)
    p = p @ np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    distance_from_center = np.linalg.norm(p, axis=-1)
    angle = np.arctan2(p[..., 0], p[..., 1])
    angle = np.arctan2(np.sin(angle) / (major_axis / 2), np.cos(angle) / (minor_axis / 2))
    angle = np.unwrap(angle)

    r = np.linalg.norm([major_axis / 2 * np.sin(angle), minor_axis / 2 * np.cos(angle)], axis=0)

    length = getEllipseArcSegment(angle, minor_axis/2, major_axis/2)
    return length, distance_from_center/r

video = r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_15_alginate2%_NIH_tanktreading_1\2\2020_09_15_10_35_15.tif"#getInputFile()
id = 8953
#video = r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\2\2020_09_16_14_36_11.tif"
#id = 4587

config = getConfig(video)

target_folder = Path(video[:-4])

data = pd.read_csv(target_folder / "output.csv")
print(data)
getStressStrain(data, config)

cell_ids = pd.unique(data.id)

pixel_size = 0.34500000000000003

#print("cell_ids", cell_ids)

def getImageStack(cell_id):
    return np.array([im for im in imageio.get_reader(target_folder / f"{cell_id:05}.tif")])

def getMask(d1, im):
    rr, cc = skimage.draw.ellipse(40, 60, d1.short_axis / pixel_size / 2, d1.long_axis / pixel_size / 2, im.shape,
                                  -d1.angle * np.pi / 180)
    mask = np.zeros(im.shape, dtype="bool")
    mask[rr, cc] = 1
    return mask

def doTracking(images, mask):
    lk_params = dict(winSize=(8, 8), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 0.03),
                     flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                     )

    for index, im in enumerate(images):
        # if it is the first image find features
        if index == 0:
            # find the features
            try:
                p0 = cv2.goodFeaturesToTrack(im, 20, 0.1, 5)[:, 0, :]
            except TypeError:
                print("error", cv2.goodFeaturesToTrack(im, 20, 0.1, 5))
                return None
            x, y = p0.astype(np.int).T
            inside = mask[y, x]
            p0 = p0[inside]

            # initialize the arrays
            tracks = np.ones((p0.shape[0], len(images), 2)) * np.nan
            active = np.ones(p0.shape[0], dtype=np.bool)
            # define the "results" of this tracking step
            st = np.ones(p0.shape[0], dtype=np.bool)
            p1 = p0
        else:
            # track the current points
            p1, st, err = cv2.calcOpticalFlowPyrLK(image_last, im, p0, None, **lk_params)
            st = st[:, 0].astype(np.bool)
            err = err[:, 0]

            # filter valid tracks (i.e. not out of bounds of the image)
            valid = (p1[:, 0] > 0) * (p1[:, 0] < im.shape[1]) * (p1[:, 1] > 0) * (p1[:, 1] < im.shape[0])
            x, y = p1.astype(np.int).T
            inside = mask[y, x]
            st = valid & st & inside #& (err < 0.2)
            active[active] = active[active] & st

        # add the found and active points to the track array
        tracks[active, index, :] = p1[st]
        # store the current points for the next iteration
        p0 = p1[st]
        image_last = im
        # if no points are left, stop
        if len(p0) == 0:
            break

    return tracks

def getLine(x, a):
    try:
        m, t = np.polyfit(x, a, deg=1)
    except (np.linalg.LinAlgError, TypeError):
        m, t = np.nan, np.nan
    return m, t

def joinImages(images):
    c, h, w = images.shape
    skip = c-1#int(np.ceil(c/10))
    c = images[::skip].shape[0]
    print("skpi", skip, c, images.shape)
    return images[::skip].transpose(1, 0, 2).reshape(h, w*c), c, skip


def makePlot(cell_id, data0, images, time, tracks, distance_to_center, angle, speeds, slopes):
    best_speeds = np.argsort(np.abs(speeds-np.nanmedian(speeds)))#[:10]
    best_indices = [best_speeds[0]]
    for i in best_speeds[1:]:
        if np.min(np.linalg.norm(tracks[np.array(best_indices), 0]-tracks[i, 0], axis=1)) > 10:
            best_indices.append(i)
        if len(best_indices) >= 10:
            break

    #tracks = tracks[::2]
    indices = [1, 0, 2, 14, 13, 10]
    #tracks = tracks[indices]

    tracks = tracks[best_indices]

    plt.subplot(231)
    im, c, skip = joinImages(images)
    plt.imshow(im, cmap="gray")
    #for index, track in enumerate(tracks.transpose(1, 0, 2)[::skip]):
    #    plt.plot(track[:, 0]+images[0].shape[1]*index, track[:, 1], "+", ms=1)
    from scipy.spatial import Delaunay
    tri = Delaunay(tracks[:, 0])
    print(tri.neighbors)

    for i, track in enumerate(tracks):
        points = track[::skip]
        index = np.arange(points.shape[0])
        print("index", index, track[:, 0], track[:, 1])
        plt.plot(points[:, 0]+images[0].shape[1]*index, points[:, 1], "o", ms=1)
        #plt.text(points[0, 0], points[0, 1], i)

    #for i in index:
    #    plt.triplot(tracks[:, ::skip][0, i, 0]+images[0].shape[1]*i, tracks[:, ::skip][0, i, 1], tri.simplices.copy())
    #plt.triplot(tracks[:, 0, 0], tracks[:, 0, 1], tri.simplices.copy())
    #plt.triplot(tracks[:, -1, 0]+images[0].shape[1], tracks[:, -1, 1], tri.simplices.copy())

    plt.subplot(234)
    la, sa, a = data0.long_axis.mean() / pixel_size, data0.short_axis.mean() / pixel_size, data0.angle.mean()
    cmap = plt.get_cmap("viridis")
    t = np.array(time)
    for j in range(tracks[0].shape[0]):
        for i in range(tracks.shape[0]):
            p = (t[j]-t[0])/(t[-1]-t[0])
            plt.plot(tracks[i, j:j+2, 0]-center[0], tracks[i, j:j+2, 1]-center[1], "o-", ms=2, lw=1.5, color=cmap(p))
    plt.plot([0], [0], "w+", ms=5)

    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=(0, 0), width=la, height=sa, angle=a, edgecolor='r', fc='None', lw=0.5, zorder=2)
    #plt.gca().add_patch(ellipse)
    plt.gca().axis("equal")
    h, w = images[0].shape
    plt.imshow(images[0], extent=[-w/2, w/2, h/2, -h/2], cmap="gray")

    plt.subplot(232)
    plt.subplot(233)
    i = -1
    for d, a, m, t in zip(distance_to_center, angle, speeds, slopes):
        i += 1
        if i not in best_indices:
            continue

        plt.subplot(232)
        plt.plot(time, (m * time + t) / np.pi * 180, "k-")
        plt.plot(time, a / np.pi * 180, "o", ms=2)
        plt.xlabel("time (ms)")
        plt.ylabel("angle (deg)")

        plt.subplot(233)
        plt.plot(d[0] * pixel_size, -m / (np.pi * 2), "o", ms=2)
        plt.xlabel("distance from center (µm)")
        plt.ylabel("rotation frequency (1/s)")

    plt.title(f"$\\gamma=${data0.grad.mean() / (2 * np.pi):.2} $\\omega=${-np.nanmedian(speeds)/(np.pi * 2):.2}")
    plt.axhline(-np.nanmedian(speeds) / (2 * np.pi), color="k", ls="--")

    from matplotlib import cm
    from matplotlib.colors import Normalize
    t = np.array(time)
    plt.colorbar(cm.ScalarMappable(norm=Normalize(t[0], t[-1]), cmap=cmap))

    pylustrator.load("tanktreading_speed.py")

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(16.210000/2.54, 10.830000/2.54, forward=True)
    plt.figure(1).ax_dict["<colorbar>"].set_position([0.262617, 0.538939, 0.007460, 0.223475])
    plt.figure(1).ax_dict["<colorbar>"].get_yaxis().get_label().set_text("time (ms)")
    plt.figure(1).axes[0].set_xlim(-0.5, 240.0)
    plt.figure(1).axes[0].set_ylim(79.5, -0.5)
    plt.figure(1).axes[0].set_xticks([np.nan])
    plt.figure(1).axes[0].set_yticks([np.nan])
    plt.figure(1).axes[0].set_xticklabels([""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[0].set_yticklabels([""], fontsize=10)
    plt.figure(1).axes[0].set_position([0.027702, 0.794044, 0.334258, 0.166540])
    plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
    plt.figure(1).axes[0].texts[0].set_position([-0.049660, 0.995106])
    plt.figure(1).axes[0].texts[0].set_text("a")
    plt.figure(1).axes[0].texts[0].set_weight("bold")
    plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
    plt.figure(1).axes[0].texts[1].set_ha("center")
    plt.figure(1).axes[0].texts[1].set_position([0.246667, 1.058626])
    plt.figure(1).axes[0].texts[1].set_text("0 ms")
    plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[2].new
    plt.figure(1).axes[0].texts[2].set_ha("center")
    plt.figure(1).axes[0].texts[2].set_position([0.780551, 1.058626])
    plt.figure(1).axes[0].texts[2].set_text("0.012 ms")
    plt.figure(1).axes[1].set_xlim(-55.45026033306826, 57.451800311593956)
    plt.figure(1).axes[1].set_ylim(39.40953773543042, -38.13245766156865)
    plt.figure(1).axes[1].set_xticks([np.nan])
    plt.figure(1).axes[1].set_yticks([np.nan])
    plt.figure(1).axes[1].set_xticklabels([""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[1].set_yticklabels([""], fontsize=10)
    plt.figure(1).axes[1].set_position([0.024415, 0.539373, 0.220267, 0.223138])
    plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
    plt.figure(1).axes[1].texts[0].set_position([-0.082711, 0.974828])
    plt.figure(1).axes[1].texts[0].set_text("b")
    plt.figure(1).axes[1].texts[0].set_weight("bold")
    plt.figure(1).axes[2].set_ylim(-230.0, 230.0)
    plt.figure(1).axes[2].set_position([0.482858, 0.613910, 0.169454, 0.333546])
    plt.figure(1).axes[2].spines['right'].set_visible(False)
    plt.figure(1).axes[2].spines['top'].set_visible(False)
    plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
    plt.figure(1).axes[2].texts[0].set_position([-0.467850, 0.976301])
    plt.figure(1).axes[2].texts[0].set_text("c")
    plt.figure(1).axes[2].texts[0].set_weight("bold")
    plt.figure(1).axes[3].set_xlim(0.0, 0.29676317677987996)
    plt.figure(1).axes[3].set_ylim(0.0, 12.040358072731685)
    plt.figure(1).axes[3].set_position([0.787367, 0.613910, 0.169454, 0.333546])
    plt.figure(1).axes[3].spines['right'].set_visible(False)
    plt.figure(1).axes[3].spines['top'].set_visible(False)
    plt.figure(1).axes[3].title.set_text("")
    plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
    plt.figure(1).axes[3].texts[0].set_position([-0.407895, 0.976301])
    plt.figure(1).axes[3].texts[0].set_text("d")
    plt.figure(1).axes[3].texts[0].set_weight("bold")
    plt.figure(1).axes[3].get_xaxis().get_label().set_text("distance\nfrom center (µm)")
    plt.figure(1).axes[3].get_yaxis().get_label().set_text("rotation\nfrequency (1/s)")
    plt.figure(1).axes[5].legend(frameon=False, borderpad=0.0, labelspacing=0.30000000000000004, handlelength=1.5999999999999999, handletextpad=0.30000000000000004, columnspacing=1.7999999999999998, markerscale=3.0, title="pressure", fontsize=10.0, title_fontsize=10.0)
    plt.figure(1).axes[5].set_position([0.116931, 0.072693, 0.415182, 0.397620])
    plt.figure(1).axes[5].spines['right'].set_visible(False)
    plt.figure(1).axes[5].spines['top'].set_visible(False)
    plt.figure(1).axes[5].get_legend()._set_loc((1.689024, 0.096641))
    plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
    plt.figure(1).axes[5].texts[0].set_position([0.950742, 0.931019])
    plt.figure(1).axes[5].texts[0].set_text("0.25")
    plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[1].new
    plt.figure(1).axes[5].texts[1].set_position([-0.122246, 0.984384])
    plt.figure(1).axes[5].texts[1].set_text("e")
    plt.figure(1).axes[5].texts[1].set_weight("bold")
    plt.figure(1).axes[5].get_yaxis().get_label().set_text("rotation\nfrequency (1/s)")
    plt.figure(1).axes[6].set_xlim(-10.0, 10.0)
    plt.figure(1).axes[6].set_ylim(-5.0, 5.0)
    plt.figure(1).axes[6].legend()
    plt.figure(1).axes[6].set_position([0.578170, 0.121344, 0.197687, 0.223475])
    plt.figure(1).axes[6].spines['right'].set_visible(False)
    plt.figure(1).axes[6].spines['top'].set_visible(False)
    plt.figure(1).axes[6].yaxis.labelpad = -9.333643
    plt.figure(1).axes[6].get_legend().set_visible(False)
    plt.figure(1).axes[6].get_xaxis().get_label().set_text("")
    plt.figure(1).axes[6].get_yaxis().get_label().set_text("")
    #% end: automatic generated code from pylustrator
    #plt.savefig(target_folder / f"fit_{cell_id}.png", dpi=300)
    print(target_folder / f"fit_{cell_id}.png")
    plt.savefig(__file__[:-3] + ".png", dpi=300)
    plt.savefig(__file__[:-3] + ".pdf")
    plt.show()
    plt.clf()
    exit()


import tqdm
with open(target_folder / "speeds.txt", "w") as fp:
    for cell_id in [id]:#tqdm.tqdm(cell_ids): # [15195]:#
    #for cell_id in [15195]:#
        data0 = data[data.id == cell_id]

        print(cell_id, len(data0))

        if len(data0) < 5:
            print("to short")
            continue

        image_stack = getImageStack(cell_id)

        #print(image_stack.shape)
        #plt.imshow(image_stack[0])
        #plt.show()

        tracks = doTracking(image_stack, mask=getMask(data0.iloc[0], image_stack[0]))

        if tracks is None:
            print("No features to track")
            continue

        center = np.array([60, 40])

        distance_to_center = np.linalg.norm(tracks - center, axis=2)
        angle = np.arctan2(tracks[:, :, 1] - center[1], tracks[:, :, 0] - center[0])
        angle = np.unwrap(angle, axis=1)

        angle, distance_to_center = getArcLength(tracks, data0.long_axis.mean() / pixel_size, data0.short_axis.mean() / pixel_size,
                                 data0.angle.mean(), center)

        time = (data0.timestamp - data0.iloc[0].timestamp) * 1e-3

        if time.shape[0] != angle.shape[1]:
            print("missmatch", len(time), len(angle), tracks.shape, angle.shape, time.shape)
            continue

        fits = np.array([getLine(time, a) for a in angle])
        speeds = fits[:, 0]
        slopes = fits[:, 1]

        try:
            makePlot(cell_id, data0, image_stack, time, tracks, distance_to_center, angle, speeds, slopes)
        except ValueError:
            pass

        fp.write(f"{cell_id} {data0.grad.mean()/(2*np.pi)} {-np.median(speeds)/(2*np.pi)} {data0.rp.mean()}\n")
