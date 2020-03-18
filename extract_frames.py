# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:34:37 2019

"""
# this program converts the frames of an avi video file to individual jpg images

import cv2

#source = r'\\131.188.117.96\biophysDS\emirzahossein\selina\free_p1_t9_vid000_2.avi'
#output=r"\\131.188.117.96\biophysDS\emirzahossein\selina\p1_t9\p1_"

source = r'\\131.188.117.96\biophysDS\emirzahossein\data_backup\20200318_alginate2%_3t3nih\p3_t3_vid000_2.avi'
output= r'G:\Ben\projects\Elham channel\20200318_alginate2%_3t3nih\p3_'

jpg = ".jpg"

vidcap = cv2.VideoCapture(source)

count = 0

while 1:
  success,image = vidcap.read()
  if success !=1:
      break
  # rotate counter clockwise
  image=cv2.transpose(image)
  image=cv2.flip(image,flipCode=0)  
  
  str_count = "{:04}".format(count)
  name = output + str_count + jpg
  cv2.imwrite(name, image,[cv2.IMWRITE_JPEG_QUALITY, 100])     # save frame as JPEG file   

  print(count)
  count += 1 