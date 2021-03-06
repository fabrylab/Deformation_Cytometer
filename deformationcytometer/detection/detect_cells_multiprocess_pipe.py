# -*- coding: utf-8 -*-
# this program reads the frames of an avi video file, averages all images,
# and stores the normalized image as a floating point numpy array
# in the same directory as the extracted images, under the name "flatfield.npy"
#
# The program then loops again through all images of the video file,
# identifies cells, extracts the cell shape, fits an ellipse to the cell shape,
# and stores the information on the cell's centroid position, long and short axis,
# angle (orientation) of the long axis, and bounding box width and height
# in a text file (result_file.txt) in the same directory as the video file.

r_min = 6
batch_size = 100
write_clickpoints_file = True
write_clickpoints_masks = False
write_clickpoints_markers = False


def log(name, name2, onoff, index=0):
    import os, time
    with open(f"log_{name}_{os.getpid()}.txt", "a") as fp:
        fp.write(f"{time.time()} \"{name2}\" {onoff} {index}\n")


def clear_logs():
    import glob, os
    files = glob.glob("log_*.txt")
    for file in files:
        os.remove(file)


class FileFinished: pass


def process_load_images(filename):
    """
    Loads an .tif file stack and yields all the images.
    """
    import imageio
    from deformationcytometer.detection import pipey
    from deformationcytometer.detection.includes.regionprops import preprocess, getTimestamp
    from deformationcytometer.includes.includes import getConfig
    import clickpoints

    print("start load images", filename)
    log("1load_images", "prepare", 1)

    # open the image reader
    reader = imageio.get_reader(filename)
    # get the config file
    config = getConfig(filename)
    # get the total image count
    image_count = len(reader)

    print("create cdb", filename[:-4]+".cdb")
    if write_clickpoints_file:
        cdb = clickpoints.DataFile(filename[:-4]+".cdb", "w")
        cdb.setMaskType("prediction", color="#FF00FF", index=1)

    yield dict(filename=filename, index=-1, type="start")
    log("1load_images", "prepare", 0)

    log("1load_images", "read", 1)
    # iterate over all images in the file
    for image_index, im in enumerate(reader):
        if image_index == image_count:
            break
        # ensure image has only one channel
        if len(im.shape) == 3:
            im = im[:, :, 0]
        # get the timestamp from the file
        timestamp = float(getTimestamp(reader, image_index))

        if write_clickpoints_file:
            cdb.setImage(filename, frame=image_index)#, timestamp=timestamp)
        log("1load_images", "read", 0, image_index)
        # return everything in a nicely packed dictionary
        yield dict(filename=filename, index=image_index, type="image", timestamp=timestamp, im=im, config=config,
                   image_count=image_count)
        if image_index < image_count - 1:
            log("1load_images", "read", 1, image_index + 1)

    yield dict(filename=filename, index=image_count, type="end")


class ProcessDetectMasksBatch:
    """
    Takes images and groups them into batches to feed them into the neural network to create masks.
    """
    unet = None
    batch = None

    def __init__(self, batch_size, network_weights):
        # store the batch size
        self.batch_size = batch_size
        self.network_weights = network_weights

    def __call__(self, data):
        from deformationcytometer.detection.includes.UNETmodel import UNet
        import numpy as np
        from deformationcytometer.detection.includes.regionprops import preprocess, getTimestamp

        # initialize the batch if necessary
        if self.batch is None:
            self.batch = []

        if data["type"] == "start":
            yield data
            return

        if data["type"] == "image":
            # add the new data
            self.batch.append(data)

        # if the batch is full or all images of the .tif file have been loaded
        if len(self.batch) == self.batch_size or (data["type"] == "end" and len(self.batch)):
            log("2detect", "prepare", 1, self.batch[0]["index"])
            batch = self.batch
            self.batch = []

            # initialize the unet if necessary
            if self.unet is None:
                im = batch[0]["im"]
                self.unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8, weights=self.network_weights)

            # predict cell masks from the image batch
            im_batch = np.dstack([data["im"] for data in batch])
            im_batch = preprocess(im_batch).transpose(2, 0, 1)
            prediction_mask_batch = self.unet.predict(im_batch[:, :, :, None])[:, :, :, 0] > 0.5

            import clickpoints
            if write_clickpoints_file and write_clickpoints_masks:
                with clickpoints.DataFile(data["filename"][:-4] + ".cdb") as cdb:
                    # iterate over all images and return them
                    for i in range(len(batch)):
                        data = batch[i]
                        data["mask"] = prediction_mask_batch[i]

                        cdb.setMask(frame=data["index"], data=data["mask"].astype(np.uint8))

                        data["config"].update({"network": self.network_weights})
                        log("2detect", "prepare", 0, data["index"])
                        yield data
                        if i < len(batch) - 1:
                            log("2detect", "prepare", 1, data["index"] + 1)
            else:
                # iterate over all images and return them
                for i in range(len(batch)):
                    data = batch[i]
                    data["mask"] = prediction_mask_batch[i]
                    data["config"].update({"network": self.network_weights})
                    log("2detect", "prepare", 0, data["index"])
                    yield data
                    if i < len(batch) - 1:
                        log("2detect", "prepare", 1, data["index"] + 1)

        if data["type"] == "end":
            return data


class ProcessFindCells:
    def __init__(self, irregularity_threshold, solidity_threshold):
        self.irregularity_threshold = irregularity_threshold
        self.solidity_threshold = solidity_threshold

    def __call__(self, data):
        import pandas as pd
        from pathlib import Path
        from deformationcytometer.detection.includes.regionprops import mask_to_cells_edge, mask_to_cells_edge2
        from deformationcytometer.evaluation.helper_functions import filterCells

        output_path = Path(data["filename"][:-4] + "_result_new.csv")

        if data["type"] != "image":
            if data["type"] == "start":
                # add ellipse marker type
                if write_clickpoints_file and write_clickpoints_markers:
                    import clickpoints
                    with clickpoints.DataFile(data["filename"][:-4] + ".cdb") as cdb:
                        cdb.setMarkerType("cell", "#FF0000", mode=cdb.TYPE_Ellipse)
                # delete an existing outputfile
                if output_path.exists():
                    output_path.unlink()
            return data

        log("3find_cells", "detect", 1, data["index"])

        new_cells = mask_to_cells_edge2(data["mask"], data["im"], data["config"], r_min,
                                       frame_data={"frames": data["index"], "timestamp": data["timestamp"]})
        new_cells = pd.DataFrame(new_cells,
                                 columns=["frames", "timestamp", "x", "y", "rp", "long_axis",
                                          "short_axis",
                                          "angle", "irregularity", "solidity", "sharpness", "velocity", "cell_id", "tt",
                                          "tt_r2"])

        if not output_path.exists():
            with output_path.open("w") as fp:
                new_cells.to_csv(fp, index=False, header=True)
        else:
            with output_path.open("a") as fp:
                new_cells.to_csv(fp, index=False, header=False)

        # filter cells according to solidity and irregularity
        new_cells = filterCells(new_cells, solidity_threshold=self.solidity_threshold,
                                irregularity_threshold=self.irregularity_threshold)

        if write_clickpoints_file and write_clickpoints_markers:
            import clickpoints
            with clickpoints.DataFile(data["filename"][:-4] + ".cdb") as cdb:
                for i, d in new_cells.iterrows():
                    cdb.setEllipse(frame=int(d.frames), x=d.x, y=d.y,
                                  width=d.long_axis / data["config"]["pixel_size"],
                                  height=d.short_axis / data["config"]["pixel_size"],
                                  angle=d.angle, type="cell")

        data["config"]["solidity"] = self.solidity_threshold
        data["config"]["irregularity"] = self.irregularity_threshold

        data["cells"] = new_cells
        del data["mask"]

        log("3find_cells", "detect", 0, data["index"])

        return data


class ProcessPairData:
    def init(self):
        self.filenames = {}

    def __call__(self, data):
        # print("vel", data["type"], data["index"])

        if data["filename"] not in self.filenames:
            self.filenames[data["filename"]] = dict(cached={-2: None}, next_index=-1, cell_index=0)

        file = self.filenames[data["filename"]]

        # cache the data
        i = data["index"]
        file["cached"][i] = data
        # try to find pairs. As we might have more detect cells processes the data is not guaranteed to come in in order
        while True:
            if file["next_index"] in file["cached"] and file["next_index"] - 1 in file["cached"]:
                yield self.match_velocities(file["cached"][file["next_index"] - 1], file["cached"][file["next_index"]])
                del file["cached"][file["next_index"] - 1]
                file["next_index"] += 1
            else:
                break
        # clear the cache when the file has been processed completely
        if "image_count" in data and file["next_index"] == data["image_count"]:
            del self.filenames[data["filename"]]

    def match_velocities(self, data1, data2):
        file = self.filenames[data2["filename"]]

        log("4vel", "prepare", 1, data2["index"])
        from deformationcytometer.detection.includes.regionprops import matchVelocities

        if data2["type"] == "end" or data2["type"] == "start":
            log("4vel", "prepare", 0, data2["index"])
            return data1, data2

        if data1["type"] == "start":
            new_cells = data2["cells"]
            for index in range(len(new_cells)):
                new_cells.iat[index, new_cells.columns.get_loc("cell_id")] = file["cell_index"]
                file["cell_index"] += 1
            log("4vel", "prepare", 0, data2["index"])
            return data1, data2

        dt = (data2["timestamp"] - data1["timestamp"])  # * 1e-3  # time is in ms

        data2["cells"], file["cell_index"] = matchVelocities(data1["cells"], data2["cells"], dt, file["cell_index"],
                                                             data2["config"])
        log("4vel", "prepare", 0, data2["index"])
        return data1, data2


class ProcessTankTreading:

    def __call__(self, data1, data2):
        from deformationcytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader
        import numpy as np
        import pandas as pd

        if data2["type"] == "start":
            yield data2
            return
        if data2["type"] == "end":
            yield data2
            return

        log("5tt", "prepare", 1, data2["index"])

        # if it is the first image of the file, just pass it along
        if data1["type"] == "start":
            log("5tt", "prepare", 0, data2["index"])
            yield data2
            return

        class CachedImageReader:
            def get_data(self, index):
                if index == data1["index"]:
                    return data1["im"]
                if index == data2["index"]:
                    return data2["im"]

        image_reader = CachedImageReader()

        for i, d2 in data2["cells"].iterrows():
            if np.isnan(d2.velocity):
                continue
            d1 = data1["cells"][data1["cells"].cell_id == d2.cell_id].iloc[0]
            d = pd.DataFrame([d2, d1])

            crops, shifts, valid = getCroppedImages(image_reader, d)

            if len(crops) <= 1:
                continue

            crops = crops[valid]
            shifts = shifts[valid]

            time = (d.timestamp - d.iloc[0].timestamp) * 1e-3

            speed, r2 = doTracking(crops, data0=d, times=np.array(time), pixel_size=data2["config"]["pixel_size"])

            data2["cells"].at[i, "tt"] = speed * 2 * np.pi
            data2["cells"].at[i, "tt_r2"] = r2
            if r2 > 0.2:
                data2["cells"].at[i, "omega"] = speed * 2 * np.pi

        log("5tt", "prepare", 0, data2["index"])
        yield data2


class ResultCombiner:

    def init(self):
        self.filenames = {}

    def __call__(self, data, data2=None):
        if data is None:
            return
        if data["filename"] not in self.filenames:
            self.filenames[data["filename"]] = dict(cached={}, next_index=-1, cell_count=0, cells=[], progressbar=None,
                                                    config=dict())

        file = self.filenames[data["filename"]]

        # cache the data
        i = data["index"]
        file["cached"][i] = data

        # try to find pairs. As we might have more detect cells processes the data is not guaranteed to come in in order
        while True:
            if file["next_index"] in file["cached"]:
                self.data(file["cached"][file["next_index"]])
                del file["cached"][file["next_index"]]
                file["next_index"] += 1
            else:
                break
        # clear the cache when the file has been processed completely
        if "image_count" in data and file["next_index"] == data["image_count"]:
            del self.filenames[data["filename"]]

    def data(self, data):
        # if the file is finished, store the results
        if data["type"] == "start":
            return

        file = self.filenames[data["filename"]]

        if data["type"] == "end":
            self.save(data)
            file["progressbar"].close()
            del self.filenames[data["filename"]]
            return

        if file["progressbar"] is None:
            import tqdm
            file["progressbar"] = tqdm.tqdm(total=data["image_count"], smoothing=0)
        file["cells"].append(data["cells"])
        file["cell_count"] += len(data["cells"])
        file["config"] = data["config"]
        file["progressbar"].update(1)
        file["progressbar"].set_description(f"cells {file['cell_count']}")

    def save(self, data):
        evaluation_version = 8
        from pathlib import Path

        import pandas as pd
        import numpy as np
        import json
        from deformationcytometer.evaluation.helper_functions import correctCenter, filterCells, getStressStrain, \
            apply_velocity_fit, get_cell_properties

        file = self.filenames[data["filename"]]

        data = pd.concat(file["cells"])
        data.reset_index(drop=True, inplace=True)

        config = file["config"]

        # take the mean of all values of each cell
        data = data.groupby(['cell_id'], as_index=False).mean()

        correctCenter(data, config)

        # data = filterCells(data, config)
        # reset the indices
        data.reset_index(drop=True, inplace=True)

        getStressStrain(data, config)

        data["area"] = data.long_axis * data.short_axis * np.pi
        data["pressure"] = config["pressure_pa"] * 1e-5

        data, p = apply_velocity_fit(data)

        omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

        output_file = Path(str(data["filename"])[:-4] + "_evaluated_new.csv")
        output_config_file = Path(str(data["filename"])[:-4] + "_evaluated_config_new.txt")
        data.to_csv(output_file, index=False)
        config["evaluation_version"] = evaluation_version
        data.to_csv(output_file, index=False)
        # print("config", config, type(config))
        with output_config_file.open("w") as fp:
            json.dump(config, fp, indent=0)


def to_filelist(paths):
    import glob
    if not isinstance(paths, list):
        paths = [paths]
    files = []
    for path in paths:
        if path.endswith(".tif"):
            if "*" in path:
                files.extend(glob.glob(path, recursive=True))
            else:
                files.append(path)
        else:
            files.extend(glob.glob(path + "/**/*.tif", recursive=True))
    return files


def get_items(d):
    from deformationcytometer.detection import pipey
    d = to_filelist(d)
    for x in d:
        yield x
    yield pipey.STOP


if __name__ == "__main__":
    from deformationcytometer.detection import pipey
    from deformationcytometer.includes.includes import getInputFile, read_args_pipeline
    import sys

    # reading commandline arguments if executed from terminal
    file, network_weight, irregularity_threshold, solidity_threshold = read_args_pipeline()

    clear_logs()

    print(sys.argv)
    video = getInputFile(settings_name="detect_cells.py")
    print(video)

    pipeline = pipey.Pipeline()

    pipeline.add(get_items)

    # one process reads the documents
    pipeline.add(process_load_images)

    pipeline.add(ProcessDetectMasksBatch(batch_size, network_weight))

    # One process combines the results into a file.
    pipeline.add(ProcessFindCells(irregularity_threshold, solidity_threshold), 2)

    pipeline.add(ProcessPairData())

    pipeline.add(ProcessTankTreading(), 2)

    pipeline.add(ResultCombiner())

    pipeline.run(video)
