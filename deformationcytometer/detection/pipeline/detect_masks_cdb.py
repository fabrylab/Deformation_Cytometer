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

        print("Mask", data["type"])

        if data["type"] == "start" or data["type"] == "end":
            yield data
            return

        print("get")
        data_storage_mask_numpy = self.data_storage.get_stored(data["mask_info"])
        #with clickpoints.DataFile(r"E:\FlowProject\2021.4.14\0.1 atm\2021_04_14_11_37_36_ellipse.cdb") as cdb: # + 10000
        with clickpoints.DataFile(data["filename"][:-4]+"_ellipse.cdb") as cdb: # + 30000
        #with clickpoints.DataFile(r"E:\FlowProject\2021.4.14\0.2 atm\2021_04_14_13_44_55_Fl_ellipse.cdb") as cdb: # + 40000
        #with clickpoints.DataFile(r"E:\FlowProject\2021.4.14\0.5 atm\2021_04_14_13_04_12_Fl.cdb") as cdb: # + 0
            print("open")
            path_entry = cdb.getPath(".")#Path(data["filename"]).parent)
            print("path_entry", path_entry)
            for i, index in enumerate(range(data["index"], data["end_index"])):
                img = cdb.table_image.get(cdb.table_image.filename==str(Path(data["filename"]).name), cdb.table_image.frame==index)#, path=path_entry)
                print("i", i, img)
                for ellipse in img.ellipses:
                    print("ellipse", data_storage_mask_numpy.shape)
                    data_storage_mask_numpy[i][skimage.draw.ellipse(ellipse.y, ellipse.x, ellipse.width / 2, ellipse.height / 2,
                                                                 data_storage_mask_numpy[i].shape, np.pi / 2 - np.deg2rad(ellipse.angle))] = 1
                    data_storage_mask_numpy[i][
                        skimage.draw.ellipse(ellipse.y, ellipse.x, ellipse.width / 2 - 3, ellipse.height / 2 - 3,
                                             data_storage_mask_numpy[i].shape,
                                             np.pi / 2 - np.deg2rad(ellipse.angle))] = 0
        #print("yield cdb", data["index"], data_storage_mask_numpy[i].max(), data_storage_mask_numpy[i].min(), data_storage_mask_numpy[i].mean())
        yield data
