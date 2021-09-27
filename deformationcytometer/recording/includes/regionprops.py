
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
import pandas as pd

r_min = 6   #cells smaller than r_min (in um) will not be analyzed

def preprocess(img):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return (img - np.mean(img)) / np.std(img).astype(np.float32)

def getTimestamp(vidcap, image_index):
    if image_index >= len(vidcap):
        image_index = len(vidcap) - 1
    if vidcap.get_meta_data(image_index)['description']:
        return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
    return "0"

def getRawVideo(filename):
    filename, ext = os.path.splitext(filename)
    raw_filename = Path(filename + "_raw" + ext)
    if raw_filename.exists():
        return imageio.get_reader(raw_filename)
    return imageio.get_reader(filename + ext)


class Timeit:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass#print("TIMEIT", self.name, time.time()-self.time)

def mask_to_cells(prediction_mask, im, config, frame, timestamp):
    with Timeit("total mask_to_cells"):
        cells = []

        # iterate over all detected regions
        with Timeit("regionprops"):
            labeled = label(prediction_mask)
            rp = regionprops(labeled, im, coordinates='rc')

        Amin_pixels = np.pi * (r_min / config["pixel_size_m"] / 1e6) ** 2  # minimum region area based on minimum radius

        for region in rp:  # region props are based on the original image
            if region.area >= Amin_pixels:  # analyze only regions larger than 100 pixels,
                a = region.major_axis_length / 2
                b = region.minor_axis_length / 2

                # get the angle
                if region.orientation > 0:
                    ellipse_angle = np.pi / 2 - region.orientation
                else:
                    ellipse_angle = -np.pi / 2 - region.orientation

                # the circumference of the ellipse
                circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

                if 0:
                    # %% compute radial intensity profile around each ellipse
                    theta = np.arange(0, 2 * np.pi, np.pi / 8)

                    with Timeit("sharp"):
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

                else:
                    sharp = 0

                # %% store the cells
                yy = region.centroid[0] - config["channel_width_px"] / 2
                yy = yy * config["pixel_size_m"] * 1e6
                #print("rp", region.centroid[0], config["channel_width_px"], config["pixel_size_m"], yy)

                cells.append([frame,
                              region.centroid[1],  # x_pos
                              region.centroid[0],  # y_pos
                              yy,                  # RadialPos
                              float(format(region.major_axis_length)) * config["pixel_size_m"] * 1e6,  # LongAxis
                              float(format(region.minor_axis_length)) * config["pixel_size_m"] * 1e6,  # ShortAxis
                              np.rad2deg(ellipse_angle),  # angle
                              region.perimeter / circum,  # irregularity
                              region.solidity,  # solidity
                              sharp,  # sharpness
                              timestamp,
                              np.nan,  # velocity
                              np.nan,  # cell id
                ])
                #cells.append(data)
    return cells


def matchVelocities(last_frame_cells, new_cells, dt, next_cell_id, config):
    # print("new_cells")
    # print(new_cells)
    if last_frame_cells is not None and new_cells is not None:
        # print("last_frame_cells")
        # print(self.last_frame_cells)
        conditions = (
            # radial pos
                (np.abs(last_frame_cells[:, None, 3] - new_cells[None, :, 3]) < 1) &
                # long_axis
                (np.abs(last_frame_cells[:, None, 4] - new_cells[None, :, 4]) < 1) &
                # short axis
                (np.abs(last_frame_cells[:, None, 5] - new_cells[None, :, 5]) < 1) &
                # angle
                (np.abs(last_frame_cells[:, None, 6] - new_cells[None, :, 6]) < 5)# &
                # positive velocity
                #(last_frame_cells[:, None, 1] < new_cells[None, :, 1])
        )
        indices = np.argmax(conditions, axis=0)
        found = conditions[indices, np.arange(conditions.shape[1])]
        # print(conditions.shape, indices, conditions[indices].shape)
        # print("found", found)
        # print("indices", indices)
        for i in range(len(indices)):
            if found[i]:
                j = indices[i]
                c1 = new_cells[i]
                c2 = last_frame_cells[j]
                v = (c1[1] - c2[1]) * config["pixel_size_m"] / dt
                new_cells[i, -2] = v
                new_cells[i, -1] = c2[-1]
            else:
                new_cells[i, -2] = np.nan
                new_cells[i, -1] = next_cell_id
                next_cell_id += 1
    if last_frame_cells is None and new_cells is not None:
        for index in range(len(new_cells)):
            new_cells[index, -1] = next_cell_id
            next_cell_id += 1
    return new_cells, new_cells, next_cell_id



def saveResults(filename, cells_record):
    df = pd.DataFrame(cells_record,
                      columns=["Frame", "x_pos", "y_pos", "RadialPos", "LongAxis", "ShortAxis", "Angle", "irregularity",
                               "solidity", "sharpness", "timestamp", "velocity", "cell_id"])
    df.to_csv(filename.replace(".tif", ".csv"))
    del df['velocity']
    del df['cell_id']
    filename_old_tmp = filename.replace(".tif", "_result.tmp")
    filename_old = filename.replace(".tif", "_result.txt")
    df.to_csv(filename_old_tmp, sep="\t", index=False)
    with open(filename_old_tmp, "r") as fp:
        with open(filename_old, "w") as fp2:
            for i, line in enumerate(fp):
                if i == 1:
                    fp2.write("Pathname " + str(Path(filename_old).parent) + "\n")
                fp2.write(line)
    Path(filename_old_tmp).unlink()


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
