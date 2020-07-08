
import numpy as np
from skimage import feature
from skimage.filters import gaussian
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import os
import imageio
import json
from pathlib import Path
import time

def preprocess(img):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return (img - np.mean(img)) / np.std(img).astype(np.float32)

def getTimestamp(vidcap, image_index):
    if vidcap.get_meta_data(image_index)['description']:
        return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
    return "0"

def getRawVideo(filename):
    filename, ext = os.path.splitext(filename)
    raw_filename = Path(filename + "_raw" + ext)
    if raw_filename.exists():
        return imageio.get_reader(raw_filename)
    return imageio.get_reader(filename + ext)


def mask_to_cells(prediction_mask, im, config, r_min, frame_data):
    cells = []
    labeled = label(prediction_mask)

    # iterate over all detected regions
    for region in regionprops(labeled, im, coordinates='rc'):  # region props are based on the original image
        a = region.major_axis_length / 2
        b = region.minor_axis_length / 2
        r = np.sqrt(a * b)

        if region.orientation > 0:
            ellipse_angle = np.pi / 2 - region.orientation
        else:
            ellipse_angle = -np.pi / 2 - region.orientation

        Amin_pixels = np.pi * (r_min / config["pixel_size_m"] / 1e6) ** 2  # minimum region area based on minimum radius

        if region.area >= Amin_pixels:  # analyze only regions larger than 100 pixels,
            # and only of the canny filtered band-passed image returend an object

            # the circumference of the ellipse
            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

            # %% compute radial intensity profile around each ellipse
            theta = np.arange(0, 2 * np.pi, np.pi / 8)

            i_r = np.zeros(int(3 * r))
            for d in range(0, int(3 * r)):
                # get points on the circumference of the ellipse
                x = d / r * a * np.cos(theta)
                y = d / r * b * np.sin(theta)
                # rotate the points by the angle fo the ellipse
                t = ellipse_angle
                xrot = (x * np.cos(t) - y * np.sin(t) + region.centroid[1]).astype(int)
                yrot = (x * np.sin(t) + y * np.cos(t) + region.centroid[0]).astype(int)
                # crop for points inside the iamge
                index = (xrot < 0) | (xrot >= im.shape[1]) | (yrot < 0) | (yrot >= im.shape[0])
                x = xrot[~index]
                y = yrot[~index]
                # average over all these points
                i_r[d] = np.mean(im[y, x])

            # define a sharpness value
            sharp = (i_r[int(r + 2)] - i_r[int(r - 2)]) / 5 / np.std(i_r)

            # %% store the cells
            yy = region.centroid[0] - config["channel_width_px"] / 2
            yy = yy * config["pixel_size_m"] * 1e6

            data = {}
            data.update(frame_data)
            data.update({
                          "x_pos": region.centroid[1],  # x_pos
                          "y_pos": region.centroid[0],  # y_pos
                          "radial_pos": yy,                  # RadialPos
                          "long_axis": float(format(region.major_axis_length)) * config["pixel_size"] * 1e6,  # LongAxis
                          "short_axis": float(format(region.minor_axis_length)) * config["pixel_size"] * 1e6,  # ShortAxis
                          "angle": np.rad2deg(ellipse_angle),  # angle
                          "irregularity": region.perimeter / circum,  # irregularity
                          "solidity": region.solidity,  # solidity
                          "sharpness": sharp,  # sharpness
            })
            cells.append(data)
    return cells


def save_cells_to_file(result_file, cells):
    result_file = Path(result_file)
    output_path = result_file.parent

    with result_file.open('w') as f:
        f.write(
            'Frame' + '\t' + 'x_pos' + '\t' + 'y_pos' + '\t' + 'RadialPos' + '\t' + 'LongAxis' + '\t' + 'ShortAxis' + '\t' + 'Angle' + '\t' + 'irregularity' + '\t' + 'solidity' + '\t' + 'sharpness' + '\t' + 'timestamp' + '\n')
        f.write('Pathname' + '\t' + str(output_path) + '\n')
        for cell in cells:
            f.write("\t".join([
                str(cell["frame"]),
                str(cell["x_pos"]),
                str(cell["y_pos"]),
                str(cell["radial_pos"]),
                str(cell["long_axis"]),
                str(cell["short_axis"]),
                str(cell["angle"]),
                str(cell["irregularity"]),
                str(cell["solidity"]),
                str(cell["sharpness"]),
                str(cell["timestamp"]),
              ])+"\n")
    print(f"Save {len(cells)} cells to {result_file}")
