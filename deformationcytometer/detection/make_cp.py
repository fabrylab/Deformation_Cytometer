import clickpoints
import numpy as np
import pandas as pd
import imageio
import tqdm
from deformationcytometer.evaluation.helper_functions import load_all_data, getData, getVelocity, correctCenter, filterCells, getConfig, getStressStrain, apply_velocity_fit, get_cell_properties, match_cells_from_all_data
from pathlib import Path


def load_one_data(input_path):

    output_file = Path(str(file).replace("_result.txt", "_evaluated.csv"))
    output_config_file = Path(str(file).replace("_result.txt", "_evaluated_config.txt"))

    # load the data and the config
    data = getData(file)
    config = getConfig(file)
    config["channel_width_m"] = 0.00019001261833616293


    """ evaluating data"""
    getVelocity(data, config)
    # take the mean of all values of each cell
    #data = data.groupby(['cell_id'], as_index=False).mean()

    if 0:
        tt_file = Path(str(file).replace("_result.txt", "._tt.csv"))
        if tt_file.exists():
            data.set_index("cell_id", inplace=True)
            data_tt = pd.read_csv(tt_file)
            data["omega"] = np.zeros(len(data))*np.nan
            for i, d in data_tt.iterrows():
                if d.tt_r2 > 0.2:# and d.id in data.index:
                    data.at[d.id, "omega"] = d.tt * 2 * np.pi

            data.reset_index(inplace=True)
        else:
            print("WARNING: tank treading has not been evaluated yet")

    correctCenter(data, config)

    #data = filterCells(data, config, solidity_threshold, irregularity_threshold)
    # reset the indices
    data.reset_index(drop=True, inplace=True)

    getStressStrain(data, config)

    #data = data[(data.stress < 50)]
    data.reset_index(drop=True, inplace=True)

    data["area"] = data.long_axis * data.short_axis * np.pi
    data["pressure"] = config["pressure_pa"]*1e-5

    data, p = apply_velocity_fit(data)

    omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

    #fitStiffness(data, config)
    return data, config


from deformationcytometer.includes.fit_velocity import fit_velocity, fit_velocity_pressures, getFitXY
def getFitLine(pressure, p):
    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
    x, y = getFitXY(config, np.mean(pressure), p)
    return x, y

file = r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_02_08_NIH3T3_LatB_drugresponse\400 nM\2\2021_02_08_16_14_19.tif"

cp_file = file[:-4] + ".cdb"
evaluation_file = file[:-4] + "_result.txt"
data, config = load_one_data(evaluation_file)
print(data)
db = clickpoints.DataFile(cp_file, "r")

d = data.iloc[0]
from scipy.interpolate import interp1d
x, y = getFitLine(d.pressure, [d.eta0, d.delta, d.tau])
vel = interp1d(x, y)

image_reader = imageio.get_reader(file)
image_height, image_width = image_reader.get_data(0).shape
#for i in tqdm.tqdm(range(len(image_reader))):
print("add images")
#clickpoints.load(file, db)
#for i in tqdm.tqdm(range(len(image_reader))):
#    db.setImage(file, frame=i)
print("images added")

marker_type_cell = db.setMarkerType("cell", "#FF0000", mode=db.TYPE_Ellipse)
track_type = db.setMarkerType("cell_track", "#FF0000", mode=db.TYPE_Track)

import time
start_time = time.time()
match_cells_from_all_data(data, config)
print(time.time()-start_time)
print(len(data.cell_id.unique()))
# 56
# 844
db.deleteEllipses()
db.deleteMarkers()
db.deleteTracks()

tracks = {}
pixel_size = config["pixel_size"]
for i, d in data.iterrows():
    if d.cell_id not in tracks:
        tracks[d.cell_id] = db.setTrack(type=track_type)
    print(d.cell_id)
    db.setMarker(frame=int(d.frames), x=d.x, y=d.y, type=track_type, track=tracks[d.cell_id])
    db.setEllipse(frame=int(d.frames), x=d.x, y=d.y,
                  width=d.long_axis/pixel_size, height=d.short_axis/pixel_size,
                  text=f"{i} {d.cell_id}",
                  angle=d.angle, type=marker_type_cell)

