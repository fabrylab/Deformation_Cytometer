# -*- coding: utf-8 -*-

#
r_min = 6
batch_size = 100
write_clickpoints_file = False
write_clickpoints_masks = False
write_clickpoints_markers = False
copy_images = True
shared_memory_size = 3000


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

class ProcessCopyImages:
    def __init__(self, data_storage: "JoinedDataStorage"):
        self.data_storage = data_storage

    def __call__(self, filename):
        """
        Loads an .tif file stack and yields all the images.
        """
        import imageio
        from deformationcytometer.detection import pipey
        from deformationcytometer.detection.includes.regionprops import preprocess, getTimestamp
        from deformationcytometer.includes.includes import getConfig
        from pathlib import Path
        import clickpoints
        import numpy as np

        import shutil

        Path("tmp").mkdir(exist_ok=True)
        target_file = Path("tmp") / Path(filename).name
        shutil.copy(filename, target_file)

        return filename, target_file


class ProcessLoadImages:
    def __init__(self, data_storage: "JoinedDataStorage"):
        self.data_storage = data_storage

    def __call__(self, filename, copy_of_file=None):
        """
        Loads an .tif file stack and yields all the images.
        """
        import imageio
        import sys
        from deformationcytometer.detection import pipey
        from deformationcytometer.detection.includes.regionprops import preprocess, getTimestamp
        from deformationcytometer.includes.includes import getConfig
        import clickpoints
        import numpy as np

        class reader2:
            def __init__(self, reader):
                self.reader = reader

            def __len__(self):
                return len(self.reader)//2

            def __iter__(self):
                for i, im in enumerate(self.reader):
                    if i % 2 == 0:
                        yield im

            def get_meta_data(self, index):
                return self.reader.get_meta_data(index*2)

            def close(self):
                self.reader.close()

        log("1load_images", "prepare", 1)

        # open the image reader
        #reader = reader2(imageio.get_reader(copy_of_file or filename))
        try:
            reader = imageio.get_reader(copy_of_file or filename)
        except Exception as err:
            print(err, file=sys.stderr)
            return
        # get the config file
        config = getConfig(filename)
        # get the total image count
        image_count = len(reader)

        if write_clickpoints_file:
            cdb = clickpoints.DataFile(filename[:-4]+".cdb", "w")
            cdb.setMaskType("prediction", color="#FF00FF", index=1)

        yield dict(filename=filename, index=-1, type="start", image_count=image_count)
        log("1load_images", "prepare", 0)

        data_storage_numpy = None

        log("1load_images", "read", 1)
        images = []
        timestamps = []
        start_batch_index = 0
        timestamp_start = None
        log("1load_images", "read", 1, 0)

        # iterate over all images in the file
        for image_index, im in enumerate(reader):
            # ensure image has only one channel
            if len(im.shape) == 3:
                im = im[:, :, 0]
            # get the timestamp from the file
            timestamp = float(getTimestamp(reader, image_index))
            if timestamp_start is None:
                timestamp_start = timestamp
            timestamp -= timestamp_start

            if write_clickpoints_file:
                cdb.setImage(filename, frame=image_index)#, timestamp=timestamp)

            images.append(im)
            timestamps.append(timestamp)

            if image_index == image_count-1 or len(images) == batch_size:

                info = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.float32)
                info_mask = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.uint8)
                data_storage_numpy = self.data_storage.get_stored(info)
                for i, im in enumerate(images):
                    data_storage_numpy[i] = im

                log("1load_images", "read", 0, start_batch_index)

                yield dict(filename=filename, index=start_batch_index, end_index=start_batch_index+len(images), type="image", timestamps=timestamps,
                           data_info=info, mask_info=info_mask,
                           config=config, image_count=image_count)

                if image_index != image_count-1:
                    log("1load_images", "read", 1, start_batch_index+len(images))
                images = []
                timestamps = []
                start_batch_index = image_index+1

            if image_index == image_count - 1:
                break

        reader.close()
        if copy_of_file is not None:
            copy_of_file.unlink()

        yield dict(filename=filename, index=image_count, type="end")


class ProcessDetectMasksBatch:
    """
    Takes images and groups them into batches to feed them into the neural network to create masks.
    """
    unet = None
    batch = None

    def __init__(self, batch_size, network_weights, data_storage, data_storage_mask):
        # store the batch size
        self.batch_size = batch_size
        self.network_weights = network_weights
        self.data_storage = data_storage
        self.data_storage_mask = data_storage_mask

    def __call__(self, data):
        import time
        predict_start_first = time.time()
        from deformationcytometer.detection.includes.UNETmodel import UNet
        import numpy as np
        from deformationcytometer.detection.includes.regionprops import preprocess, getTimestamp

        if data["type"] == "start" or data["type"] == "end":
            yield data
            return

        log("2detect", "prepare", 1, data["index"])

        def preprocess(img):
            img = img - np.mean(img, axis=(1, 2))[:, None, None]
            img = img / np.std(img, axis=(1, 2))[:, None, None]
            return img.astype(np.float32)

        data_storage_numpy = self.data_storage.get_stored(data["data_info"])
        data_storage_mask_numpy = self.data_storage.get_stored(data["mask_info"])

        # initialize the unet if necessary
        im = data_storage_numpy[0]  # batch[0]["im"]
        if self.unet is None or self.unet.shape[:2] != im.shape:
            im = data_storage_numpy[0]#batch[0]["im"]
            if self.network_weights is not None and self.network_weights != "":
                self.unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8, weights=self.network_weights)
            else:
                self.unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8)

        # predict cell masks from the image batch
        im_batch = preprocess(data_storage_numpy)
        import time
        predict_start = time.time()
        import tensorflow as tf
        with tf.device('/GPU:0'):
            prediction_mask_batch = self.unet.predict(im_batch[:, :, :, None])[:, :, :, 0] > 0.5
        dt = time.time() - predict_start
        data_storage_mask_numpy[:] = prediction_mask_batch

        import clickpoints
        if write_clickpoints_file and write_clickpoints_masks:
            with clickpoints.DataFile(data["filename"][:-4] + ".cdb") as cdb:
                # iterate over all images and return them
                for mask, index in zip(data_storage_mask_numpy, range(data["index"], data["end_index"])):
                    cdb.setMask(frame=index, data=mask.astype(np.uint8))


        data["config"].update({"network": self.network_weights})

        log("2detect", "prepare", 0, data["index"])
        yield data


class ProcessFindCells:
    def __init__(self, irregularity_threshold, solidity_threshold, data_storage):
        self.irregularity_threshold = irregularity_threshold
        self.solidity_threshold = solidity_threshold
        self.data_storage = data_storage

    def __call__(self, data):
        import time
        predict_start_first = time.time()
        import pandas as pd
        from pathlib import Path
        from deformationcytometer.detection.includes.regionprops import mask_to_cells_edge, mask_to_cells_edge2
        from deformationcytometer.evaluation.helper_functions import filterCells
        import numpy as np

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

        data_storage_mask_numpy = self.data_storage.get_stored(data["mask_info"])

        log("3find_cells", "detect", 1, data["index"])
        new_cells = []
        row_indices = [0]
        for mask, timestamp, index in zip(data_storage_mask_numpy, data["timestamps"], range(data["index"], data["index"]+data_storage_mask_numpy.shape[0])):
            cells = mask_to_cells_edge2(mask, None, data["config"], r_min, frame_data={"frames": index, "timestamp": timestamp})
            row_indices.append(row_indices[-1]+len(cells))
            new_cells.extend(cells)

        new_cells = pd.DataFrame(new_cells,
                                 columns=["frames", "timestamp", "x", "y", "rp", "long_axis",
                                          "short_axis",
                                          "angle", "irregularity", "solidity", "sharpness", "velocity", "cell_id", "tt",
                                          "tt_r2", "omega"])

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

        #new_cells.set_index("frames", inplace=True)

        data["cells"] = new_cells
        data["row_indices"] = row_indices
        #del data["mask"]

        log("3find_cells", "detect", 0, data["index"])
        return data


class ProcessPairData:
    def __call__(self, data):
        from deformationcytometer.detection.includes.regionprops import matchVelocities

        if data["type"] == "end" or data["type"] == "start":
            return data

        log("4vel", "prepare", 1, data["index"])

        cells = data["cells"]
        row_indices = data["row_indices"]
        next_id = 1+data["index"]*1000

        _, next_id = matchVelocities(cells.iloc[0:0],
                                     cells.iloc[row_indices[0]:row_indices[1]],
                                     next_id, data["config"])

        for i, index in enumerate(range(data["index"], data["end_index"]-1)):
            _, next_id = matchVelocities(cells.iloc[row_indices[i]:row_indices[i+1]],
                                         cells.iloc[row_indices[i+1]:row_indices[i+2]],
                                         next_id, data["config"])

        log("4vel", "prepare", 0, data["index"])
        return data


class ProcessTankTreading:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def __call__(self, data):
        from deformationcytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader
        import numpy as np
        import pandas as pd
        pd.options.mode.chained_assignment = 'raise'

        if data["type"] == "start" or data["type"] == "end":
            return data

        log("5tt", "prepare", 1, data["index"])

        data_storage_numpy = self.data_storage.get_stored(data["data_info"])

        class CachedImageReader:
            def get_data(self, index):
                return data_storage_numpy[int(index)-data["index"]]

        image_reader = CachedImageReader()
        cells = data["cells"]
        row_indices = data["row_indices"]

        for i, index in enumerate(range(data["index"], data["end_index"]-1)):
            cells1 = cells.iloc[row_indices[i+0]:row_indices[i+1]]
            cells2 = cells.iloc[row_indices[i+1]:row_indices[i+2]]

            for i, (index, d2) in enumerate(cells2.iterrows()):
                if np.isnan(d2.velocity):
                    continue
                try:
                    d1 = cells1[cells1.cell_id == d2.cell_id].iloc[0]
                except IndexError:
                    continue
                d = pd.DataFrame([d2, d1])

                crops, shifts, valid = getCroppedImages(image_reader, d)

                if len(crops) <= 1:
                    continue

                crops = crops[valid]
                shifts = shifts[valid]

                time = (d.timestamp - d.iloc[0].timestamp) * 1e-3

                speed, r2 = doTracking(crops, data0=d, times=np.array(time), pixel_size=data["config"]["pixel_size"])

                cells2.iat[i, cells2.columns.get_loc("tt")] = speed * 2 * np.pi
                cells2.iat[i, cells2.columns.get_loc("tt_r2")] = r2
                if r2 > 0.2:
                    cells2.iat[i, cells2.columns.get_loc("omega")] = speed * 2 * np.pi

        log("5tt", "prepare", 0, data["index"])
        return data


class ResultCombiner:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def init(self):
        self.filenames = {}

    def __call__(self, data):
        import sys
        # if the file is finished, store the results

        if data["filename"] not in self.filenames:
            self.filenames[data["filename"]] = dict(cached={}, next_index=-1, cell_count=0, cells=[], progressbar=None,
                                                    config=dict())

        file = self.filenames[data["filename"]]

        if file["progressbar"] is None and data["type"] != "end":
            import tqdm
            file["progressbar"] = tqdm.tqdm(total=data["image_count"], smoothing=0)
            file["progress_count"] = 0

        if data["type"] == "start" or data["type"] == "end":
            return

        log("6combine", "prepare", 1, data["index"])

        file["cells"].append(data["cells"])
        file["cell_count"] += len(data["cells"])
        file["progress_count"] += data["end_index"]-data["index"]
        file["config"] = data["config"]
        file["progressbar"].update(data["end_index"]-data["index"])
        file["progressbar"].set_description(f"cells {file['cell_count']}")

        self.data_storage.deallocate(data["data_info"])
        self.data_storage.deallocate(data["mask_info"])

        if file["progress_count"] == data["image_count"]:
            try:
                self.save(data)
            except Exception as err:
                print(err, file=sys.stderr)
            file["progressbar"].close()
            del self.filenames[data["filename"]]

        log("6combine", "prepare", 0, data["index"])

    def save(self, data):
        evaluation_version = 8
        from pathlib import Path

        import pandas as pd
        import numpy as np
        import json
        from deformationcytometer.evaluation.helper_functions import correctCenter, filterCells, getStressStrain, \
            apply_velocity_fit, get_cell_properties, match_cells_from_all_data

        filename = data["filename"]
        image_width = data["data_info"]["shape"][2]

        file = self.filenames[filename]

        data = pd.concat(file["cells"])
        data.reset_index(drop=True, inplace=True)

        config = file["config"]

        try:
            correctCenter(data, config)
        except Exception as err:
            print("WARNING: could not fit center for", filename, err)

        if 0:
            try:
                # take the mean of all values of each cell
                data = data.groupby(['cell_id'], as_index=False).mean()
            except pd.core.base.DataError:
                pass

        # data = filterCells(data, config)
        # reset the indices
        data.reset_index(drop=True, inplace=True)

        getStressStrain(data, config)

        data["area"] = data.long_axis * data.short_axis * np.pi
        data["pressure"] = config["pressure_pa"] * 1e-5

        data, p = apply_velocity_fit(data)

        # do matching of velocities again
        match_cells_from_all_data(data, config, image_width)

        omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

        output_file = Path(str(filename)[:-4] + "_evaluated_new.csv")
        output_config_file = Path(str(filename)[:-4] + "_evaluated_config_new.txt")
        data.to_csv(output_file, index=False)
        config["evaluation_version"] = evaluation_version
        data.to_csv(output_file, index=False)

        with output_config_file.open("w") as fp:
            json.dump(config, fp, indent=0)


def to_filelist(paths, reevaluate=False):
    import glob
    from pathlib import Path
    from deformationcytometer.includes.includes import getConfig

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
    files2 = []
    for filename in files:
        if reevaluate or not Path(str(filename)[:-4] + "_evaluated_config_new.txt").exists():
            # check if the config file exists
            try:
                config = getConfig(filename)
            except (OSError, ValueError):
                continue
            files2.append(filename)
        else:
            print(filename, "already evaluated")
    return files2


class get_items:
    def __init__(self, reevaluate):
        self.reevaluate = reevaluate

    def __call__(self, d):
        from deformationcytometer.detection import pipey
        d = to_filelist(d, self.reevaluate)
        for x in d:
            yield x
        yield pipey.STOP


import numpy as np
class JoinedDataStorage:
    def __init__(self, count):
        self.image_data = DataBlock(count, dtype=np.float32)
        self.mask_data = DataBlock(count, dtype=np.uint8)

    def get_stored(self, info):
        if info["dtype"] == np.float32:
            return self.image_data.get_stored(info)
        return self.mask_data.get_stored(info)

    def allocate(self, shape, dtype=np.float32):
        import time
        while True:
            try:
                if dtype == np.float32:
                    return self.image_data.allocate(shape)
                return self.mask_data.allocate(shape)
            except ValueError:
                time.sleep(1)

    def deallocate(self, info):
        if info["dtype"] == np.float32:
            return self.image_data.deallocate(info)
        return self.mask_data.deallocate(info)


class DataBlock:
    def __init__(self, count, dtype):
        self.max_size = 540 * 720 * count
        self.default_dtype = dtype
        self.data_storage = multiprocessing.Array({np.float32: "f", np.uint8: "B"}[dtype], 540 * 720 * count)
        self.data_storage_allocated = multiprocessing.Array("L", 100*2)

    def get_stored(self, info):
        return np.frombuffer(self.data_storage.get_obj(), count=int(np.prod(info["shape"])), dtype=info["dtype"], offset=info["offset"]).reshape(info["shape"])

    def allocate(self, shape, dtype=None):
        dtype = dtype or self.default_dtype

        def getOverlap(a, b):
            return max(0, min(a[1], b[1]) - max(a[0], b[0]))

        allocated_blocks = np.frombuffer(self.data_storage_allocated.get_obj(), dtype=np.uint32).reshape(-1, 2)
        start_frames = sorted([b[0] for b in allocated_blocks if b[1]])
        end_frames = sorted([b[1] for b in allocated_blocks if b[1]])

        count = np.prod(shape)
        bytes = np.dtype(dtype).itemsize

        start = 0
        for s, e in zip(start_frames, end_frames):
            if not getOverlap([start, start+count*bytes], [s, e]):
                break
            start = e
        if start+count*bytes > self.max_size:
            raise ValueError()
        for i, (s, e) in enumerate(allocated_blocks):
            if e == 0:
                allocated_blocks[i] = [start, start+count*bytes]
                break

        return dict(shape=shape, offset=start, dtype=dtype, name="data_storage", allocation_index=i)

    def deallocate(self, info):
        allocated_blocks = np.frombuffer(self.data_storage_allocated.get_obj(), dtype=np.uint32).reshape(-1, 2)
        allocated_blocks[info["allocation_index"], :] = 0


if __name__ == "__main__":
    from deformationcytometer.detection import pipey
    from deformationcytometer.includes.includes import getInputFile, read_args_pipeline, getInputFolder
    import sys
    import multiprocessing
    import argparse

    data_storage = JoinedDataStorage(shared_memory_size)

    # reading commandline arguments if executed from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('file', default=None, help='specify an input file or folder')  # positional argument
    parser.add_argument('-n', '--network_weight', default="", help='provide an external the network weight file')
    parser.add_argument('-r', '--irregularity_filter', type=float, default=1.06, help='cells with larger irregularity (deviation from elliptical shape) are excluded')
    parser.add_argument('-s', '--solidity_filter', type=float, default=0.96, help='cells with smaller solidity are excluded')
    parser.add_argument('-f', '--force', type=bool, default=False, help='if True reevaluate already evaluated files')
    args = parser.parse_args()

    file = args.file
    network_weight = args.network_weight
    irregularity_threshold = args.irregularity_filter
    solidity_threshold = args.solidity_filter

    clear_logs()

    print(sys.argv)
    video = getInputFolder(settings_name="detect_cells.py")
    print(video)

    pipeline = pipey.Pipeline(3)

    pipeline.add(get_items(args.force))

    if copy_images is True:
        pipeline.add(ProcessCopyImages(data_storage))

    # one process reads the documents
    #pipeline.add(process_load_images)
    pipeline.add(ProcessLoadImages(data_storage))

    pipeline.add(ProcessDetectMasksBatch(batch_size, network_weight, data_storage, None))

    # One process combines the results into a file.
    pipeline.add(ProcessFindCells(irregularity_threshold, solidity_threshold, data_storage), 1)

    pipeline.add(ProcessPairData())

    pipeline.add(ProcessTankTreading(data_storage), 3)

    pipeline.add(ResultCombiner(data_storage))

    pipeline.run(video)
