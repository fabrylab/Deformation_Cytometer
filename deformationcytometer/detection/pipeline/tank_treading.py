from deformationcytometer.detection.includes.pipe_helpers import *


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
