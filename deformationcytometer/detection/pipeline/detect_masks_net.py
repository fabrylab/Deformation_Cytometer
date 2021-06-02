from deformationcytometer.detection.includes.pipe_helpers import *



class ProcessDetectMasksBatch:
    """
    Takes images and groups them into batches to feed them into the neural network to create masks.
    """
    unet = None
    batch = None

    def __init__(self, batch_size, network_weights, data_storage, data_storage_mask, write_clickpoints_masks):
        # store the batch size
        self.batch_size = batch_size
        self.network_weights = network_weights
        self.data_storage = data_storage
        self.data_storage_mask = data_storage_mask
        self.write_clickpoints_masks

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
        if self.write_clickpoints_masks:
            with clickpoints.DataFile(data["filename"][:-4] + ".cdb") as cdb:
                # iterate over all images and return them
                for mask, index in zip(data_storage_mask_numpy, range(data["index"], data["end_index"])):
                    cdb.setMask(frame=index, data=mask.astype(np.uint8))


        data["config"].update({"network": self.network_weights})

        log("2detect", "prepare", 0, data["index"])
        yield data