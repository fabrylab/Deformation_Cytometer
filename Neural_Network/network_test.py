import os
import tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0' # makes sure debug info is printed

from deformationcytometer.detection.includes.UNETmodel import UNet
from Neural_Network.includes.training_functions import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

u = UNet((1000,1000,1), 1, 8)

rand_n = np.random.randint(0,10,(1, 1000, 1000,1))
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
print(get_available_gpus())

for i in range(4):
    print(np.sum(u.predict(rand_n)))
ns = []
dts = []
for n in [1, 5, 10, 20, 30, 40, 100]:
    try:
        print(n)
        rand_n = np.random.randint(0, 10, (n, 1000, 1000, 1))
        get_available_gpus()
        t1 = time.time()
        for i in range(4):
            np.sum(u.predict(rand_n))
        t2 = time.time()
        dt = t2 - t1
        ns.append(n)
        dts.append(dt)
    except:
        pass
dts_norm = np.array(dts)/(4*np.array(ns))
plt.figure()
plt.plot(ns, dts_norm, "o-")
plt.xlabel("batch size")
plt.ylabel("calculation time/image")
plt.savefig("cuda11_2_tf24.png")
