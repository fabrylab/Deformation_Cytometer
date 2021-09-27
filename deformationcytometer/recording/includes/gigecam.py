from includes.MemMap import MemMap
import numpy as np
import configparser

class GigECam():
    last_slot_index = None
    counter_last = None

    def __init__(self, mmap_xml=None):
        if mmap_xml is None:
            config_cam = configparser.ConfigParser()
            config_cam.read("config.txt")
            mmap_xml = config_cam["camera"]["output_mmap"]
        self.mmap = MemMap(mmap_xml)

    def getNewestImage(self):
        # get newest counter
        counters = [slot.counter for slot in self.mmap.rbf]
        counter_max = np.max(counters)
        counter_max_idx = np.argmax(counters)
        # return if there is no new one
        if counter_max == self.counter_last:
            return None, None
        image = self.mmap.rbf[counter_max_idx].image
        timestamp = self.mmap.rbf[counter_max_idx].time_unix * 1000 + self.mmap.rbf[counter_max_idx].time_us/1000
        self.counter_last = counter_max
        self.last_slot_index = counter_max_idx
        return image, timestamp

    def getNextImage(self):
        if self.counter_last is None or self.last_slot_index is None:
            return self.getNewestImage()

        next_slot_index = (self.last_slot_index + 1) % len(self.mmap.rbf)
        if self.counter_last < self.mmap.rbf[next_slot_index].counter:
            diff = self.mmap.rbf[next_slot_index].counter - self.counter_last
            if diff > 1:
                self.counter_last = self.mmap.rbf[next_slot_index].counter
                raise ValueError("Skipped Frames", diff)
            self.last_slot_index = next_slot_index
            self.counter_last = self.mmap.rbf[next_slot_index].counter
            image = self.mmap.rbf[next_slot_index].image
            timestamp = self.mmap.rbf[next_slot_index].time_unix * 1000 + self.mmap.rbf[next_slot_index].time_us/1000
            return image, timestamp
        else:
            return None, None

    def getMaxValue(self):
        return 256#2**12
