import numpy as np
import sys

# camera libraries
from pypylon import pylon
from pypylon import genicam
# signal handling for grace full shutdown
import signal
# time
import datetime
# include
from includes.dotdict import dotdict
from includes.MemMap import MemMap
import configparser

import os
import psutil

# set the priority for this process
psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)

config = configparser.ConfigParser()
config.read("config.txt")

# setup ctrl + c handler
run = True
def signal_handler(sig, frame):
    global run  # we need to have this as global to work
    print("Ctrl+C received")
    print("Shutting down PylonStreamer ...")
    run = False
signal.signal(signal.SIGINT, signal_handler)

# TODO: add codes for color cameras!
fmt2depth = {'Mono8': 8,
             'Mono12': 12,
             'Mono12p': 12}

transpose_image = config["camera"]["transpose_image"] == "True"
flip_x = config["camera"]["flip_x"] == "True"
flip_y = config["camera"]["flip_y"] == "True"

""" PARAMETER """
# select the correct camera mmap cfg here
#output_mmap = "cfg/camera_acA5472-17um.xml"
#output_mmap = "cfg/camera_acA4112-30um.xml"
output_mmap = config["camera"]["output_mmap"]
settings_mmap = config["camera"]["settings_mmap"]

# Setup USB Camera access via Pylon
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# activate chunkmode for frame timestamps, gain and exposure values
camera.ChunkModeActive = True
camera.ChunkSelector = "Timestamp"
camera.ChunkEnable = True
camera.ChunkSelector = "Gain"
camera.ChunkEnable = True
camera.ChunkSelector = "ExposureTime"
camera.ChunkEnable = True

# set camera parameters for Exposure and Gain control
# camera.ExposureAuto = "Continuous"
# TODO we need to expose these values if we want to change them during runtime
camera.AutoExposureTimeLowerLimit = 30.0     # micro seconds
camera.AutoExposureTimeUpperLimit = 1800     # micro seconds
camera.GainAuto = "Continuous"
camera.AutoTargetBrightness = 0.5
camera.AutoGainLowerLimit = 0
camera.AutoGainUpperLimit = 36
camera.AcquisitionFrameRateEnable = True
#camera.AcquisitionFrameRate = 500
camera.ExposureTime = 30

# init memmory map output
mmap = MemMap(output_mmap)
smap = MemMap(settings_mmap)
# start continious acquisition, default as Freerun
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
while not camera.IsGrabbing():
    print('waiting for camera to start')
if (smap.framerate)>0:
    framerate = float(smap.framerate)
else:
    framerate = camera.ResultingFrameRate.Value
    smap.framerate = framerate
target_framerate = framerate
camera.AcquisitionFrameRate = 500#framerate
##
timings = []
counter = 0
dropped_counter = 0
slot = 0
gain = 0
run = True
index  = 0
last_framerate = None
buffer_size = len(mmap.rbf)
recording_index = 0
burst_frames = 1
while camera.IsGrabbing() and run:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        recording_index += 1
        TimeStampCam = grabResult.GetTimeStamp()
        image = grabResult.Array

        if recording_index % int(np.round(camera.ResultingFrameRate.Value/target_framerate)) >= burst_frames:
            continue
        index += 1
        slot = (slot+1) % buffer_size

        if transpose_image is True:
            image = image.T
        if flip_y is True:
            image = image[::-1, ::]
        if flip_x is True:
            image = image[:, ::-1]

        mmap.rbf[slot].image[:, :, :] = image[:,:,None]
        mmap.rbf[slot].time_unix = TimeStampCam//1000000000  # floor to remove microseconds
        mmap.rbf[slot].time_us   = TimeStampCam//1000 % 1000000 # microseconds timestamp
        mmap.rbf[slot].counter = index
                   
        
        if index % framerate == 0:
            gain = grabResult.ChunkGain.Value
            smap.gain = gain
            # only set the framerate if it is different
            if float(smap.framerate) != last_framerate:
                last_framerate = float(smap.framerate)
                target_framerate = last_framerate
                #camera.AcquisitionFrameRate = last_framerate
            print('skipped: ', grabResult.GetNumberOfSkippedImages(), \
				'\t gain: ', gain, \
				'\t temperature: ',camera.DeviceTemperature.Value, \
                '\t framerate: ',camera.ResultingFrameRate.Value, \
                '\t target framerate: ', target_framerate, \
                  )
            
        continue
        
    else:
        # in case of an error
        print("Err: %d\t - %s" % (grabResult.GetErrorCode(), grabResult.GetErrorDescription()))
        dropped_counter +=1

    grabResult.Release()
