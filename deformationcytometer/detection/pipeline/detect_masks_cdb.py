from deformationcytometer.detection.includes.pipe_helpers import *


class ProcessReadMasksBatch:
    def __init__(self, batch_size, network_weights, data_storage, data_storage_mask):
        # store the batch size
        self.batch_size = batch_size
        self.network_weights = network_weights
        self.data_storage = data_storage
        self.data_storage_mask = data_storage_mask

    def __call__(self, data):
        import numpy as np
        import skimage.draw
        import clickpoints
        from pathlib import Path

        if data["type"] == "start" or data["type"] == "end":
            yield data
            return

        data_storage_mask_numpy = self.data_storage.get_stored(data["mask_info"])
        #with clickpoints.DataFile(r"E:\FlowProject\2021.4.14\0.1 atm\2021_04_14_11_37_36_ellipse.cdb") as cdb: # + 10000
        if Path(data["filename"][:-4]+"_ellipse.cdb").exists():
         with clickpoints.DataFile(data["filename"][:-4]+"_ellipse.cdb") as cdb:
            path_entry = cdb.getPath(".")#Path(data["filename"]).parent)
            for i, index in enumerate(range(data["index"], data["end_index"])):
                data_storage_mask_numpy[i][:] = 0
                img = cdb.table_image.get(cdb.table_image.filename==str(Path(data["filename"]).name), cdb.table_image.frame==index)#, path=path_entry)
                for ellipse in img.ellipses:
                    data_storage_mask_numpy[i][skimage.draw.ellipse(ellipse.y, ellipse.x, ellipse.width / 2, ellipse.height / 2,
                                                                 data_storage_mask_numpy[i].shape, np.pi / 2 - np.deg2rad(ellipse.angle))] = 1
                    data_storage_mask_numpy[i][
                        skimage.draw.ellipse(ellipse.y, ellipse.x, ellipse.width / 2 - 3, ellipse.height / 2 - 3,
                                             data_storage_mask_numpy[i].shape,
                                             np.pi / 2 - np.deg2rad(ellipse.angle))] = 0
        yield data
