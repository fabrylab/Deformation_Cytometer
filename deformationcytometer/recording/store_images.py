# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:10:40 2020

@author: Ben
"""
from includes.MemMap import MemMap
import numpy as np
import os
# import time, sys
import tkinter as tk
from tkinter import filedialog
import tifffile
from datetime import datetime  # , timedelta
import configparser
import configparser
from configparser import ConfigParser

config_cam = configparser.ConfigParser()
config_cam.read("config.txt")
global config_setup
config_setup="config_setup.txt"

settings_mmap = config_cam["camera"]["settings_mmap"]
smap = MemMap(settings_mmap)
global gain, frame_rate
gain = smap.gain
frame_rate = smap.framerate
global finish, running
finish = False
running = False

config_0 = configparser.ConfigParser()
config_0.read(config_setup)
for section_name in config_0:
    #print('Section:', section_name)
    section = config_0[section_name]
    #print('  Options:', list(section.keys()))
    for name in section:
        #print('  {} = {}'.format(name, section[name]))
        if name=="pressure":
            pressure_1=(section[name])
        if name=="imaging position after inlet":
            imaging_position_1=(section[name])
        if name=="room temperature":
            room_temperature_1=(section[name])
        if name=="bioink":
            bioink_1=(section[name])
        if name=="condensor aperture":
            aperture_1=(section[name])
        if name=="cell type":
            cell_type_1=(section[name])
        if name=="cell passage number":
            cell_passage_number_1=(section[name])
        if name=="time after harvest":
            time_after_harvest_1=(section[name])
        if name=="treatment":
            treatment_1=(section[name])



    print()

def acquire_images():
    global tif_name_0, config_cam
    counter = 0
    counter_end = 0
    # dropped_counter = 0
    button_Start.configure(state='disabled')
    global running, finish
    running = True
    label_no_of_im.config(text=('opening Tiff Writer'))
    root.update()
    date_time = datetime.now()
    tif_name_0 = str(date_time.year) + '_' + \
                 '{0:02d}'.format(date_time.month) + '_' + \
                 '{0:02d}'.format(date_time.day) + '_' + \
                 '{0:02d}'.format(date_time.hour) + '_' + \
                 '{0:02d}'.format(date_time.minute) + '_' + \
                 '{0:02d}'.format(date_time.second)
    tif_name = folder_selected + '/' + tif_name_0 + '.tif'
    print(tif_name)
    tiffWriter = tifffile.TiffWriter(tif_name, bigtiff=True)
    label_no_of_im.config(text=('opening camera'))
    root.update()
    cam = GigECam(config_cam["camera"]["output_mmap"])
    label_no_of_im.config(text=('starting image aquisition'))
    root.update()
    im, timestamp = cam.getNewestImage()

    # counter_end = cam.counter_last + 10000
    counter_end = cam.counter_last + int(duration_1) * frame_rate
    print(counter_end)
    counter = cam.counter_last
    #start_time = time.time()


    while cam.counter_last < counter_end:
        im, timestamp = cam.getNextImage()
        if not (im is None):
            metad = {'timestamp': str(timestamp)}
            # value = str(timestamp)
            # print(metad)
            # tiffWriter.save(im, compress=5, metadata=metad)
            tiffWriter.save(im, compress=0, metadata=metad, contiguous=False)
            # tiffWriter.save(im, compress=0, extratags=[(270, 's', 1, value, False)])
            counter = counter + 1
            if (counter_end - counter) % frame_rate == 0:
                # print(timestamp, counter, time.time()-start_time)
                label_no_of_im.config(text=str(counter - counter_end + int(duration_1) * frame_rate))
                gain = smap.gain
                label_gain.config(text=str(gain))
                root.update()
        if running == False:
            break
    tiffWriter.close()
    del cam
    button_Start.configure(state='normal')
    running = False

    config = configparser.ConfigParser()
    config['Default'] = {'version': '1'}
    pressure_str = str(pressure_1)  # + ' kPa'
    imaging_pos_str = str(imaging_position_1)  # + ' cm'
    room_temperature_str = str(room_temperature_1)  # + ' deg C'
    cell_temperature_str = room_temperature_str
    bioink_str = str(bioink_1)
    config['SETUP'] = {'pressure': pressure_str, 'channel width': '200 um',
                       'channel length': '5.8 cm', 'imaging position after inlet': imaging_pos_str,
                       'bioink': bioink_str, 'room temperature': room_temperature_str,
                       'cell temperature': cell_temperature_str}
    aperture_str = str(aperture_1)
    config['MICROSCOPE'] = {'microscope': 'Leica DM 6000', 'objective': '40 x',
                            'na': '0.6', 'coupler': '0.5 x', 'condensor aperture': aperture_str}
    frame_rate_str = str(frame_rate) + ' fps'
    gain_str = str(gain_1)
    # print(gain_str)
    config['CAMERA'] = {'exposure time': '30 us', 'gain': gain_str, 'frame rate': frame_rate_str,
                        'camera': 'Basler acA20-520', 'camera pixel size': '6.9 um'}
    cell_type_str = str(cell_type_1)
    cell_passage_number_str = str(cell_passage_number_1)
    time_after_harvest_str = str(time_after_harvest_1)  # + ' min'
    treatment_str = str(treatment_1)
    config['CELL'] = {'cell type': cell_type_str, 'cell passage number': cell_passage_number_str,
                      'time after harvest': time_after_harvest_str, 'treatment': treatment_str}
    config_name = folder_selected + '/' + tif_name_0 + '_config' + '.txt'  # config_name
    with open(config_name, 'w') as configfile:
        config.write(configfile)
    with open(config_setup, 'w') as configfile:
        config.write(configfile)
    configfile.close()

    if finish == True:
        root.destroy()


def stop_images():
    global running
    running = False


def exit_program():
    global finish, running
    if running == False:
        root.destroy()
    finish = True
    running = False


def change_frame_rate():
    global frame_rate, gain_1
    try:
        frame_rate = int(framerate.get())
    except:
        frame_rate = smap.framerate
    if frame_rate > 500:
        frame_rate = 500
    smap.framerate = frame_rate
    framerate.delete(0, tk.END)
    framerate.insert(10, str(smap.framerate))
    print(frame_rate)
    gain = smap.gain
    gain_1 = gain
    label_gain.config(text=str(gain_1))
    root.update()


def change_pressure():
    global pressure, pressure_1
    pressure_1 = pressure.get()
    print(pressure_1)
    root.update()


def change_aperture():
    global aperture, aperture_1
    aperture_1 = aperture.get()
    print(aperture_1)
    root.update()


def change_imaging_position():
    global imaging_position, imaging_position_1
    imaging_position_1 = imaging_position.get()
    print(imaging_position_1)
    root.update()


def change_bioink():
    global bioink, bioink_1
    bioink_1 = bioink.get()
    print(bioink_1)
    root.update()


def change_room_temperature():
    global room_temperature, room_temperature_1
    room_temperature_1 = room_temperature.get()
    print(room_temperature_1)
    root.update()


def change_cell_type():
    global cell_type, cell_type_1
    cell_type_1 = cell_type.get()
    print(cell_type_1)
    root.update()


def change_cell_passage_number():
    global cell_passage_number, cell_passage_number_1
    cell_passage_number_1 = cell_passage_number.get()
    print(cell_passage_number_1)
    root.update()


def change_time_after_harvest():
    global time_after_harvest, time_after_harvest_1
    time_after_harvest_1 = time_after_harvest.get()
    print(time_after_harvest_1)
    root.update()


def change_treatment():
    global treatment, treatment_1
    treatment_1 = treatment.get()
    print(treatment_1)
    root.update()


def change_duration():
    global duration, duration_1
    duration_1 = duration.get()
    print(duration_1)
    root.update()


def change_dir():
    root1 = tk.Tk()
    root1.withdraw()
    global folder_selected
    folder_selected = filedialog.askdirectory()
    print(folder_selected)
    if folder_selected != '':
        label_path.config(text=folder_selected)
    root1.destroy()
    gain = smap.gain
    label_gain.config(text=str(gain))
    root.update()


class GigECam():
    def __init__(self, mmap_xml):
        self.mmap = MemMap(mmap_xml)
        self.counter_last = -1
        # percentil calc paramter
        self.pct_counter = 0
        self.pct_min = None
        self.pct_max = None

    def getNewestImage(self):
        # get newest counter
        counters = [slot.counter for slot in self.mmap.rbf]
        counter_max = np.max(counters)
        counter_max_idx = np.argmax(counters)
        # return if there is no new one
        if counter_max == self.counter_last:
            # print("not new!")
            return None, None
        image = self.mmap.rbf[counter_max_idx].image
        timestamp = self.mmap.rbf[counter_max_idx].time_unix * 1000 + self.mmap.rbf[counter_max_idx].time_us/1000
        # print(image.shape, image.dtype)
        self.counter_last = counter_max
        self.last_slot_index = counter_max_idx
        # print("img type:", image.dtype)
        return image, timestamp

    #    def getNextImage(self):
    #        next_slot_index = (self.last_slot_index + 1) % len(self.mmap.rbf)
    #        if self.counter_last < self.mmap.rbf[next_slot_index].counter:
    #            #if self.counter_last+1 != self.mmap.rbf[next_slot_index].counter:
    #                #raise ValueError("Skipped Frames")
    #            self.last_slot_index = next_slot_index
    #            self.counter_last = self.mmap.rbf[self.next_slot_index].counter
    #            image = self.mmap.rbf[self.last_slot_index].image
    #            timestamp = self.mmap.rbf[self.last_slot_index].time_unix*1000 + self.mmap.rbf[self.last_slot_index].time_ms
    #            return image, timestamp
    #        else:
    #            return None, None

    def getNextImage(self):
        next_slot_index = (self.last_slot_index + 1) % len(self.mmap.rbf)
        if self.counter_last < self.mmap.rbf[next_slot_index].counter:
            # if self.counter_last+1 != self.mmap.rbf[next_slot_index].counter:
            # raise ValueError("Skipped Frames")
            self.last_slot_index = next_slot_index
            self.counter_last = self.mmap.rbf[next_slot_index].counter
            image = self.mmap.rbf[next_slot_index].image
            timestamp = self.mmap.rbf[next_slot_index].time_unix * 1000 + self.mmap.rbf[next_slot_index].time_us/1000
            return image, timestamp
        else:
            return None, None


root = tk.Tk()
root.title("Acquire Images")
# frame = tk.Frame(root)
# frame.pack()

label_path = tk.Label(root, fg="dark green")
label_path.grid(row=0, column=1, columnspan=2)
label_path.config(text=os.getcwd())

button_ChangeDir = tk.Button(root, width=20,
                             text="Change Dir",
                             command=change_dir)
button_ChangeDir.grid(row=0, column=0, sticky='w', pady=2)

txt_no_of_im = tk.Label(root, text="# of images")
txt_no_of_im.grid(row=2, column=0, sticky='w', pady=2)

label_no_of_im = tk.Label(root, width=20, fg="dark green")
label_no_of_im.grid(row=2, column=1, sticky='w', pady=2)

txt_gain = tk.Label(root, text="gain")
txt_gain.grid(row=3, column=0, sticky='w', pady=2)

gain_1 = 0
label_gain = tk.Label(root, width=20, fg="dark green")
label_gain.grid(row=3, column=1, sticky='w', pady=2)
label_gain.config(text=str(gain_1))
root.update()

txt_framerate = tk.Label(root, text="framerate")
txt_framerate.grid(row=4, column=0, sticky='w', pady=2)

framerate = tk.Entry(root)
framerate.grid(row=4, column=1, sticky='w', pady=2)
framerate.insert(10, str(smap.framerate))

txt_pressure = tk.Label(root, text="Pressure")
txt_pressure.grid(row=6, column=0, sticky='w', pady=2)

pressure_1 = 100
pressure = tk.Entry(root)
pressure.grid(row=6, column=1, sticky='w', pady=2)
pressure.insert(10, pressure_1)

txt_imaging_position = tk.Label(root, text="Imaging position")
txt_imaging_position.grid(row=8, column=0, sticky='w', pady=2)

imaging_position_1 = 2.9
imaging_position = tk.Entry(root)
imaging_position.grid(row=8, column=1, sticky='w', pady=2)
imaging_position.insert(10, imaging_position_1)

txt_aperture = tk.Label(root, text="Aperture")
txt_aperture.grid(row=7, column=0, sticky='w', pady=2)

#aperture_1 = 8
aperture = tk.Entry(root)
aperture.grid(row=7, column=1, sticky='w', pady=2)
aperture.insert(10, 8)

txt_bioink = tk.Label(root, text="Bioink")
txt_bioink.grid(row=9, column=0, sticky='w', pady=2)

bioink_1 = 'alginate 2 pc'
bioink = tk.Entry(root)
bioink.grid(row=9, column=1, sticky='w', pady=2)
bioink.insert(10, bioink_1)

txt_room_temperature = tk.Label(root, text="Room temperature")
txt_room_temperature.grid(row=10, column=0, sticky='w', pady=2)

room_temperature_1 = 25
room_temperature = tk.Entry(root)
room_temperature.grid(row=10, column=1, sticky='w', pady=2)
room_temperature.insert(10, room_temperature_1)

txt_cell_type = tk.Label(root, text="Cell type")
txt_cell_type.grid(row=11, column=0, sticky='w', pady=2)

cell_type_1 = 'NIH-3T3'
cell_type = tk.Entry(root)
cell_type.grid(row=11, column=1, sticky='w', pady=2)
cell_type.insert(10, cell_type_1)

txt_cell_passage_number = tk.Label(root, text="Cell passage number")
txt_cell_passage_number.grid(row=12, column=0, sticky='w', pady=2)

cell_passage_number_1 = 14
cell_passage_number = tk.Entry(root)
cell_passage_number.grid(row=12, column=1, sticky='w', pady=2)
cell_passage_number.insert(10, cell_passage_number_1)

txt_time_after_harvest = tk.Label(root, text="Time after harvest")
txt_time_after_harvest.grid(row=13, column=0, sticky='w', pady=2)

time_after_harvest_1 = 20
time_after_harvest = tk.Entry(root)
time_after_harvest.grid(row=13, column=1, sticky='w', pady=2)
time_after_harvest.insert(10, time_after_harvest_1)

txt_treatment = tk.Label(root, text="Treatment")
txt_treatment.grid(row=14, column=0, sticky='w', pady=2)

treatment_1 = 'none'
treatment = tk.Entry(root)
treatment.grid(row=14, column=1, sticky='w', pady=2)
treatment.insert(10, treatment_1)

txt_duration = tk.Label(root, text="Duration in s")
txt_duration.grid(row=15, column=0, sticky='w', pady=2)

duration_1 = '20'
duration = tk.Entry(root)
duration.grid(row=15, column=1, sticky='w', pady=2)
duration.insert(10, duration_1)

'''
button_ChangeFrameRate = tk.Button(root, width=20,
                        text="Change Framerate",
                        fg="black",
                        command=change_frame_rate)
button_ChangeFrameRate.grid(row = 4, column = 2,sticky = 'w', pady = 2)
'''
button_Quit = tk.Button(root, width=20,
                        text="Quit",
                        fg="red",
                        command=exit_program)
button_Quit.grid(row=5, column=0, sticky='w', pady=2)

button_Start = tk.Button(root, width=20,
                         text="Start",
                         command=acquire_images)
button_Start.grid(row=5, column=1, sticky='w', pady=2)

button_Stop = tk.Button(root, width=20,
                        text="Stop",
                        command=stop_images)
button_Stop.grid(row=5, column=2, sticky='w', pady=2)

button_ChangeConfig = tk.Button(root, width=20,
                                text="Change Configuration",
                                fg="black",
                                command=lambda: [change_frame_rate(), change_duration(), change_pressure(),
                                                 change_aperture(), change_imaging_position(), change_bioink(),
                                                 change_room_temperature(), change_cell_type(),
                                                 change_cell_passage_number(), change_time_after_harvest(),
                                                 change_treatment()])

button_ChangeConfig.grid(row=6, column=2, sticky='w', pady=2)

'''
button_ChangeDuration = tk.Button(root, width=20,
                        text="Change Duration",
                        command=change_duration)
button_ChangeDuration.grid(row = 15, column = 2,sticky = 'w', pady = 2)
'''
# button_ChangePressure = tk.Button(root, width=20,
#                                  text="Change Pressure",
#                                  fg="black",
#                                  command=change_pressure)
# button_ChangePressure.grid(row = 6, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeAperture = tk.Button(root, width=20,
#                                  text="Change Aperture",
#                                  fg="black",
#                                  command=change_aperture)
# button_ChangeAperture.grid(row = 7, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeIamging_Position = tk.Button(root, width=20,
#                                  text="Change Imaging Position",
#                                  fg="black",
#                                  command=change_imaging_position)
# button_ChangeIamging_Position.grid(row = 8, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeBioink = tk.Button(root, width=20,
#                                  text="Change Bioink",
#                                  fg="black",
#                                  command=change_bioink)
# button_ChangeBioink.grid(row = 9, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeRoom_temperature = tk.Button(root, width=20,
#                                  text="Change Room Temperature",
#                                  fg="black",
#                                  command=change_room_temperature)
# button_ChangeRoom_temperature.grid(row = 10, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeCell_type = tk.Button(root, width=20,
#                                  text="Change Cell Type",
#                                  fg="black",
#                                  command=change_cell_type)
# button_ChangeCell_type.grid(row = 11, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeCell_passage_number = tk.Button(root, width=20,
#                                  text="Change Cell Passage Number",
#                                  fg="black",
#                                  command=change_cell_passage_number)
# button_ChangeCell_passage_number.grid(row = 12, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeTime_after_harvest = tk.Button(root, width=20,
#                                  text="Change Time after harvest",
#                                  fg="black",
#                                  command=change_time_after_harvest)
# button_ChangeTime_after_harvest.grid(row = 13, column = 2,sticky = 'w', pady = 2)
#
# button_ChangeTreatment = tk.Button(root, width=20,
#                                  text="Change Treatment",
#                                  fg="black",
#                                  command=change_treatment)
# button_ChangeTreatment.grid(row = 14, column = 2,sticky = 'w', pady = 2)


folder_selected = os.getcwd()
label_path.config(text=folder_selected)

root.mainloop()




