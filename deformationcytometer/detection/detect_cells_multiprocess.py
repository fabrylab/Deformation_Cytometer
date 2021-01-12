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

def queue_iterator(queue):
    while True:
        res = queue.get()
        if res == 0:
            break
        yield res


def process_load_images(filename, image_batch_queue):
    import imageio
    from deformationcytometer.detection.includes.regionprops import preprocess, batch_iterator

    vidcap = imageio.get_reader(filename)

    # iterate over image batches
    for batch_images, batch_image_indices in batch_iterator(vidcap, batch_size, preprocess):
        # update the description of the progressbar
        image_batch_queue.put([batch_images.copy(), batch_image_indices])
    image_batch_queue.put(0)


def process_detect_masks(video, image_batch_queue, mask_queue):
    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    from deformationcytometer.detection.includes.UNETmodel import UNet

    unet = None
    for batch_images, batch_image_indices in queue_iterator(image_batch_queue):

        # initialize the unet in the first iteration
        if unet is None:
            im = batch_images[0]
            unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8)

        # predict the images
        prediction_mask_batch = unet.predict(batch_images[:, :, :, None])[:, :, :, 0] > 0.5

        mask_queue.put([batch_images, batch_image_indices, prediction_mask_batch])
    mask_queue.put(0)


def process_find_cells(video, mask_queue):
    import tqdm
    import imageio
    from pathlib import Path
    from deformationcytometer.includes.includes import getConfig
    from deformationcytometer.detection.includes.regionprops import save_cells_to_file, mask_to_cells_edge, getTimestamp

    # get image and config
    vidcap = imageio.get_reader(video)
    config = getConfig(video)

    cells = []

    # initialize the progressbar
    with tqdm.tqdm(total=len(vidcap)) as progressbar:
        # update the description of the progressbar
        progressbar.set_description(f"{len(cells)} good cells")

        for batch_images, batch_image_indices, prediction_mask_batch in queue_iterator(mask_queue):
            # iterate over the predicted images
            for batch_index in range(len(batch_image_indices)):
                image_index = batch_image_indices[batch_index]
                im = batch_images[batch_index]
                prediction_mask = prediction_mask_batch[batch_index]

                # get the images in the detected mask
                cells.extend(mask_to_cells_edge(prediction_mask, im, config, r_min,
                                frame_data={"frame": image_index, "timestamp": getTimestamp(vidcap, image_index)}))

            # update the count of the progressbar with the current batch
            progressbar.update(len(batch_image_indices))

    # save the results
    save_cells_to_file(Path(video[:-3] + '_result.txt'), cells)


if __name__ == "__main__":
    from multiprocessing import Process, Queue
    from deformationcytometer.includes.includes import getInputFile

    video = getInputFile(settings_name="detect_cells.py")
    print(video)

    # initialize the queues
    image_batch_queue = Queue(2)
    mask_queue = Queue(2)

    # initialize the processes
    processes = [
        Process(target=process_load_images, args=(video, image_batch_queue)),
        Process(target=process_detect_masks, args=(video, image_batch_queue, mask_queue)),
        Process(target=process_find_cells, args=(video, mask_queue)),
    ]
    # start the processes
    for p in processes:
        p.deamon = True
        p.start()
    # wait for all processes
    for p in processes:
        p.join()
