import numpy as np
from datetime import datetime, timedelta
from includes.MemMap import MemMap
from includes.dotdict import dotdict


## generic RBF for data
class genericRBF():
    def __init__(self, dims, dtype, nr_slots):
        self.buffer = np.zeros((nr_slots,*dims), dtype=dtype)
        self.dims = dims
        self.nr_slots = nr_slots
        self.idx = 0

    def _increment_slot(self):
        if self.idx >= self.nr_slots - 1:
            self.idx = 0
        else:
            self.idx += 1

        return self.idx

    def setNextSlot(self, data):
        self.buffer[self.idx] = data
        self._increment_slot()

# sample = genericRBF([10, 20], np.int, 5)
# sample.setNextSlot(np.ones(sample.dims))

## memmory map input based RBF to extend generic RBF
class memmapRBF(genericRBF):
    def __init__(self,mmap_xml):
        self.mmap = MemMap(mmap_xml)
        self.nr_slots = len(self.mmap.rbf)
        self.idx = -1

        self.counter_last = -1

    def getIdxNewest(self):
        """
        access mode: retrieve newest image
        use this if you do not care about dropped images, reduce lag time
        e.g. live display
        """

        # get newest counter
        counters = [slot.counter for slot in self.mmap.rbf]
        counter_max = np.max(counters)
        counter_max_idx = np.argmax(counters)

        # return if there is no new one
        if counter_max == self.counter_last:
            self.idx = None
        else:
            self.idx = counter_max_idx

        frames_skipped = counter_max - self.counter_last - 1
        if frames_skipped > 0:
            print("Warning - %d frames skipped")

        self.counter_last = counter_max

        return self.idx

    def getIdxOldestUnused(self, safety_margin=2, verbose=False):
        """
        access mode: retrieve oldest unused frame
        use this to process all images in the mmap if possible
        e.g. image storage

        """

        # get newest and oldest counter, use copy to prevent modifications in mmap
        counters = np.array([slot.counter for slot in self.mmap.rbf]).copy().astype(np.float)
        # remove already used counters by setting them to nan
        counters[counters <= self.counter_last] = np.nan

        # all frames processed
        if np.all(np.isnan(counters)):
            return None

        counter_max = np.nanmax(counters)
        counter_max_idx = np.nanargmax(counters)
        counter_min = np.nanmin(counters)
        counter_min_idx = np.nanargmin(counters)

        # NOTE: never use the min position without a safety offset while the cam is recordings
        # the min idx slot will be the next one to be overwritten!
        save_idx = counter_min_idx
        # use safety
        if (not (counter_min_idx <= counter_max_idx) and
                ((counter_max_idx + safety_margin) % len(self.mmap.rbf) >= counter_min_idx)):
            save_idx = (counter_max_idx + safety_margin) % len(self.mmap.rbf)


        # get current count
        counter_now = counters[save_idx]

        # return if there is no new frame
        if counter_now == self.counter_last:
            self.idx = None
        else:
            self.idx = save_idx

        # calc frames skipped
        frames_skipped = counter_now - self.counter_last - 1
        if frames_skipped > 0:
            print("Warning - %d frames skipped" % frames_skipped)

        # calc lag time (nr of leading frames between write and read process)
        if verbose:
            lag_count = counter_max_idx - save_idx
            print("lag count: %d" % lag_count)


        self.counter_last = counter_now

        return self.idx



# camera class to retrieve images
class GigECam():
    def __init__(self,mmap_xml):
        self.mmap = MemMap(mmap_xml)
        self.counter_last = -1


    # def getSlotOldestUnused(self):
    #     # access mode: retrieve oldest unused
    #     # use this to process all images in the mmap if possible
    #     # e.g. image storage
    #
    #     # get newest and oldest counter
    #     counters = [slot.counter for slot in self.mmap.rbf]
    #
    #     counter_max = np.max(counters)
    #     counter_max_idx = np.argmax(counters)
    #     counter_min = np.min(counters)
    #     counter_min_idx = np.argmin(counters)
    #
    #     # NOTE: never use the min position without a safety offset,
    #     # as this slot will be the next one to be overwritten!
    #
    #
    #
    #     # return if there is no new one
    #     if counter_max == self.counter_last:
    #         # print("not new!")
    #         return None
    #
    #     return counter_max_idx


    def getNewestImage(self, return16bit=False, return_meta=False):
        # access mode: retrieve newest image
        # use this if you do not care about droped image but want to reduce lag time
        # e.g. live display

        # get newest counter
        counters = [ slot.counter for slot in self.mmap.rbf]

        counter_max = np.max(counters)
        counter_max_idx = np.argmax(counters)

        # return if there is no new one
        if counter_max == self.counter_last:
            # print("not new!")
            return None

        image = self.mmap.rbf[counter_max_idx].image


        try:
            im_channels = image.shape[2]
        except:
            im_channels = 1

        im_rsize = self.mmap.rbf[counter_max_idx].width * self.mmap.rbf[counter_max_idx].height * im_channels

        # check for ROI
        if not (im_rsize == len(image.flatten())):
           im_roi = image.flatten()[0: im_rsize ]

           image = im_roi.reshape([self.mmap.rbf[counter_max_idx].height,self.mmap.rbf[counter_max_idx].width,im_channels])

        # check for 16bit data
        if image.dtype == 'uint16' and not return16bit:
            # calculate 8 bit representation
            min = np.percentile(image,1)
            max = np.percentile(image,99)

            image = ((image - min) / (max - min) * 255)
            image [image < 0] = 0
            image [image > 255] = 255
            image = image.astype('uint8')

        self.counter_last = counter_max

        if return_meta:
            meta = dotdict({'timestamp': datetime.fromtimestamp(self.mmap.rbf[counter_max_idx].time_unix) +
                                 timedelta(milliseconds=int(self.mmap.rbf[counter_max_idx].time_ms))})
            return image, meta
        else:
            return image


# Rolling Counter
class RollCounter(object):
    """ RollOver counter to keep track of slot in mmap rbfs - stop rewriting code all the time
        Usage:
            a = RollCounter(0,5)    # start values, max values
            a()                     # returns the current values
            a.step()                # increments and rolls over if max value is reached
        """
    def __init__(self,start,max):
        self.counter = start
        self.max = max

    def __call__(self):
        return self.counter

    def step(self):
        self.counter += 1
        if self.counter >= self.max:
            self.counter = 0

class ImageStackHandler():
    """ utility handler for unsorted image stacks (provided by wsFilter) """
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.mmap = MemMap(self.xml_file)
        self.current_slot = None        # currently newest slot in the rbf
        self.oldest_slot = None         # currently oldes slot in the rbf
        self.current_frame = None       # frame number of the newest frame
        self.indexes = None             # index / frame number list
        self.stack_shape = None         # shape of the stack
        self.sorted_stack = None        # the stack sorted by frame numbers
        self.sorted_ids = None          # the sorted index / frame number list

    def update(self):
        """ update to current mmap state """
        self.indexes = np.array([i.counter for i in self.mmap.rbf[:]])
        self.current_slot = np.argmax(self.indexes)
        self.oldest_slot = np.argmin(self.indexes)
        self.current_frame = self.indexes[self.current_slot]

        # get image stack buffer
        istack = np.array([i.image for i in self.mmap.rbf[:]])
        self.stack_shape = istack.shape

        # sort by ids and rearrange stack
        self.sorted_ids = np.argsort(self.indexes)
        self.sorted_stack = istack[self.sorted_ids, :, :]

