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


def process_load_images(filename):
    import imageio
    from deformationcytometer.detection import pipey
    from deformationcytometer.detection.includes.regionprops import preprocess,  getTimestamp
    from deformationcytometer.includes.includes import getConfig

    reader = imageio.get_reader(filename)
    config = getConfig(filename)
    image_count = len(reader)

    for image_index, im in enumerate(reader):
        preprocess(im)
        timestamp = float(getTimestamp(reader, image_index))
        yield dict(index=image_index, timestamp=timestamp, im=im, config=config, image_count=image_count)

    yield pipey.STOP


class DetectMasksBatch:
    unet = None
    batch = None

    def __init__(self, batch_size):
        # you can record some arguments here, still within the parent process.
        self.batch_size = batch_size

    def __call__(self, data):
        if self.batch is None:
            self.batch = []
        self.batch.append(data)
        if len(self.batch) == self.batch_size:
            batch = self.batch
            self.batch = None
            from deformationcytometer.detection.includes.UNETmodel import UNet
            import numpy as np

            if self.unet is None:
                im = batch[0]["im"]
                self.unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8)

            # predict the images
            prediction_mask_batch = self.unet.predict(np.array([data["im"] for data in batch])[:, :, :, None])[:, :, :, 0] > 0.5

            for i in range(len(batch)):
                data = batch[i]
                data["mask"] = prediction_mask_batch[i]
                yield data


class ResultCombiner:
    def __init__(self, filename):
        # you can record some arguments here, still within the parent process.
        self.filename = filename

    def init(self):
        self.cells = []
        self.progressbar = None

    def __call__(self, data):
        if self.progressbar is None:
            import tqdm
            self.progressbar = tqdm.tqdm(total=data["image_count"], smoothing=0)
        self.cells.append(data["cells"])
        self.config = data["config"]
        self.progressbar.update(1)

    def shutdown(self):
        evaluation_version = 7
        from pathlib import Path

        import pandas as pd
        import numpy as np
        import json
        from deformationcytometer.evaluation.helper_functions import correctCenter, filterCells, getStressStrain, apply_velocity_fit, get_cell_properties
        data = pd.concat(self.cells)
        data.reset_index(drop=True, inplace=True)

        config = self.config

        # take the mean of all values of each cell
        #data = data.groupby(['cell_id'], as_index=False).mean()

        correctCenter(data, config)

        #data = filterCells(data, config)
        # reset the indices
        data.reset_index(drop=True, inplace=True)

        getStressStrain(data, config)

        data["area"] = data.long_axis * data.short_axis * np.pi
        data["pressure"] = config["pressure_pa"] * 1e-5

        data, p = apply_velocity_fit(data)

        omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

        output_file = Path(str(self.filename)[:-4]+"_evaluated.csv")
        output_config_file = Path(str(self.filename)[:-4]+"_evaluated_config.txt")
        data.to_csv(output_file, index=False)
        config["evaluation_version"] = evaluation_version
        data.to_csv(output_file, index=False)
        # print("config", config, type(config))
        with output_config_file.open("w") as fp:
            json.dump(config, fp)


def process_find_cells(data):
    import pandas as pd
    from deformationcytometer.detection.includes.regionprops import mask_to_cells_edge

    new_cells = mask_to_cells_edge(data["mask"], data["im"], data["config"], r_min,
                                   frame_data={"frames": data["index"], "timestamp": data["timestamp"]})
    new_cells = pd.DataFrame(new_cells, columns=["frames", "timestamp", "x_pos", "y_pos", "radial_pos", "long_axis", "short_axis",
                          "angle", "irregularity", "solidity", "sharpness", "velocity", "cell_id", "tt", "tt_r2"])

    for pair in [["x", "x_pos"], ["y", "y_pos"], ["rp", "radial_pos"]]:
        new_cells[pair[0]] = new_cells[pair[1]]
        del new_cells[pair[1]]

    data["cells"] = new_cells
    del data["mask"]

    return data


class pair_data:
    def init(self):
        self.cached = {}
        self.next_index = 1
        self.cell_index = 0

    def __call__(self, data):
        # cache the data
        i = data["index"]
        self.cached[i] = data
        # try to find pairs. As we might have more detect cells processes the data is not guaranteed to come in in order
        while True:
            if self.next_index in self.cached and self.next_index-1 in self.cached:
                yield self.match_velocities(self.cached[self.next_index-1], self.cached[self.next_index])
                del self.cached[self.next_index-1]
                self.next_index += 1
            else:
                break

    def match_velocities(self, data1, data2):
        from deformationcytometer.detection.includes.regionprops import matchVelocities

        dt = (data2["timestamp"] - data1["timestamp"]) * 1e-3  # time is in ms

        data2["cells"], self.cell_index = matchVelocities(data1["cells"], data2["cells"], dt, self.cell_index,
                                                          data2["config"])

        return data1, data2


class get_tt_speed:

    def __call__(self, data1, data2):
        from deformationcytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader
        import numpy as np
        import pandas as pd

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
            data2["cells"].at[i, "omega"] = speed * 2 * np.pi

        yield data2


if __name__ == "__main__":
    from deformationcytometer.detection import pipey
    from deformationcytometer.includes.includes import getInputFile
    video = getInputFile(settings_name="detect_cells.py")
    print(video)

    pipeline = pipey.Pipeline()

    # one process reads the documents
    pipeline.add(process_load_images)

    pipeline.add(DetectMasksBatch(batch_size))

    # One process combines the results into a file.
    pipeline.add(process_find_cells, 2)

    pipeline.add(pair_data())

    pipeline.add(get_tt_speed(), 2)

    pipeline.add(ResultCombiner(video))

    pipeline.run(video)
