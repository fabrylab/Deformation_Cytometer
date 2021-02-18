
from tqdm import tqdm
from deformationcytometer.detection.includes.UNETmodel import UNet
from Neural_Network.includes.training_functions import *
import numpy as np
get_available_gpus()
u = UNet((1000,1000,1), 1, 8)
for i in tqdm(range(1000)):
     print(np.sum(u.predict(np.random.randint(0,10,(20,1000,1000,1)))))