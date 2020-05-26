# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:34:37 2019

"""
# this program converts the frames of an avi video file to individual jpg images
# it also averages all images and stores the normalized image as a floating point numpy array 
# in the same directory as the extracted images, under the name "flatfield.npy"

import cv2
import numpy as np

#source = r'\\131.188.117.96\biophysDS\emirzahossein\selina\free_p1_t9_vid000_2.avi'
#output=r"\\131.188.117.96\biophysDS\emirzahossein\selina\p1_t9\p1_"

source = r'\\131.188.117.96\biophysDS\emirzahossein\data_backup\20200318_alginate2%_3t3nih\p3_t3_vid000_2.avi'
output_path= r'G:\Ben\projects\Elham channel\20200318_alginate2%_3t3nih\\'
output= output_path + '\p3_'
flatfield = output_path + 'flatfield'

jpg = ".jpg"

vidcap = cv2.VideoCapture(source)

count = 0
im_av = []
while 1:
  success,image = vidcap.read()
  if success !=1:
      break
  # rotate counter clockwise
  image=cv2.transpose(image)
  image=cv2.flip(image,flipCode=0)  
  
  str_count = "{:04}".format(count)
  name = output + str_count + jpg
#  cv2.imwrite(name, image,[cv2.IMWRITE_JPEG_QUALITY, 100])     # save frame as JPEG file   
  if count == 0:
      im_av = np.zeros((image.shape[0], image.shape[1]))
  im_av = im_av + image[:,:,0]   
  print(count)
  count += 1 

im_av = im_av / np.mean(im_av)
#np.save(flatfield, im_av)