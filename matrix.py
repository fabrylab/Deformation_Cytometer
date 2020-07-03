# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:29:16 2020

@author: Selina
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

X_data = []
files = [img for img in glob.glob ("//131.188.117.96/biophysDS/emirzahossein/data/Selina/Z-stacks/cell_01/cropped/*.tif")]


s = 0
files.sort() # ADD THIS LINE
for myFile in files:
    if s == 11: # the 11th image was missing in this set of images
        image = np.zeros((image.shape))
        #print('here')
        X_data.append (image)

    #print(myFile)
    image = cv2.imread (myFile)
    #print(np.shape(image))
    X_data.append (image)
    s+=1
    
print('X_data shape:', np.array(X_data).shape)

X_data = np.array(X_data)

fig = plt.figure(figsize=(18,6))
plt.xlabel('Distance from the center in mum')
plt.ylabel('Aperture setting')
#plt.yticks(np.arange(4, 14, step=2))
#plt.xticks(np.arange(-16, 20, step=1))
fig.tight_layout()
plt.axis('off')
for i in range(120):
    sub = fig.add_subplot(6, 20, i+1)    
    sub.imshow(X_data[2*i,:,:,0], cmap = 'gray', interpolation='bicubic',aspect = "auto")
    sub.axis('off')
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()
plt.savefig('//131.188.117.96/biophysDS/emirzahossein/data/Selina/Z-stacks/cell_01/Matrix_apertures_new.jpg')