from deformationcytometer.detection.includes.pipe_helpers import *


class ResultCombiner:
    def __init__(self, data_storage, output="_evaluated_new.csv"):
        self.data_storage = data_storage
        self.output = output

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
        try:
            match_cells_from_all_data(data, config, image_width)
        except AttributeError:
            pass

        omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

        output_file = Path(str(filename)[:-4] + self.output)
        output_config_file = Path(str(filename)[:-4] + "_evaluated_config_new.txt")
        config["evaluation_version"] = evaluation_version
        data.to_csv(output_file, index=False)

        with output_config_file.open("w") as fp:
            json.dump(config, fp, indent=0)

