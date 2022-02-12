#%% Import the nescessary stuff
# basic OS stuff
import time, os, sys, shutil

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# small utilities
import csv
from colour import Color
from itertools import compress # for list selection with logical
from tqdm import tqdm

# for image manipulation
import cv2

# for recording and connecting to the intel realsense librar
#import pyrealsense as pyrs

#import multiprocessing
import multiprocessing
from multiprocessing import Process

import click
#%% helping functions to run stuff during recording!
def load_auto_roi(which_device):

    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:8] == 'led_mask' for x in folder_list]

    led_list = list(compress(folder_list,logic_list))
    led_list.sort()
    # get the last one
    newest_folder = led_list[-1]
    # and make a folder
    constant_folder = '/media/chrelli/Data0/'+newest_folder #TODO fix this shit!

    # now read the masks
    which_label = 'auto_led'
    led_mask = cv2.imread(constant_folder+'/dev'+str(which_device)+'_roi_frame_'+which_label+'.png',0)
    # also return a logical
    led_logic = led_mask > 0

    # and the central pixel of the largest binary region
    img, contours, hierarchy = cv2.findContours(led_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # calculate the momnets and centroids
    moments = [cv2.moments(cnt) for cnt in contours]
    # get the size of the regions
    region_size = np.empty(len(contours))
    for i in range(len(contours)):
        region_size[i] = np.shape(contours[i])[0]
    # biggest region
    biggest_region = np.argmax(region_size)
    M = moments[biggest_region]
    # this is in the cv2 coordinate system!!  not matrix notation
    # get the centroid of the LED like this:
    led_centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    return led_mask,led_logic,led_centroid

def load_auto_roi_with_background(which_device):

    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:8] == 'led_mask' for x in folder_list]

    led_list = list(compress(folder_list,logic_list))
    led_list.sort()
    # get the last one
    newest_folder = led_list[-1]
    # and make a folder
    constant_folder = '/media/chrelli/Data0/'+newest_folder #TODO fix this shit!

    # now read the masks
    which_label = 'auto_led'
    led_mask = cv2.imread(constant_folder+'/dev'+str(which_device)+'_roi_frame_'+which_label+'.png',0)
    # also return a logical
    led_logic = led_mask > 0

    # and the central pixel of the largest binary region
    img, contours, hierarchy = cv2.findContours(led_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # calculate the momnets and centroids
    moments = [cv2.moments(cnt) for cnt in contours]
    # get the size of the regions
    region_size = np.empty(len(contours))
    for i in range(len(contours)):
        region_size[i] = np.shape(contours[i])[0]
    # biggest region
    biggest_region = np.argmax(region_size)
    M = moments[biggest_region]
    # this is in the cv2 coordinate system!!  not matrix notation
    # get the centroid of the LED like this:
    led_centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    # also load the median depth
    which_label = 'median depth'
    d_median = np.load(constant_folder+'/dev'+str(which_device)+'_roi_frame_'+which_label+'.npy')

    return led_mask,led_logic,led_centroid,d_median

# function to read the roi file
def read_roi_file(which_device):
    # LOOK for the most recent folder
    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:8] == 'roi_mask' for x in folder_list]

    led_list = list(compress(folder_list,logic_list))
    led_list.sort()
    # get the last one
    newest_folder = led_list[-1]
    # and make a folder
    constant_folder = '/media/chrelli/Data0/'+newest_folder #TODO fix this shit!


    this_name = constant_folder+'/dev'+str(which_device)+'_cad_roi.csv'
    if os.path.exists(this_name):
        roi_values = np.genfromtxt(this_name, delimiter=',',dtype='int' )
    else:
        print('ERROR: '+ this_name+' not found!')
        sys.exit(0)

    return roi_values




def single_pixel_RGB2GRAY(rgb):
    # uses same weighting as cv2
    gray = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    return gray

# also define the firmata
def blink_using_firmata():
    # for getting the port
    import serial, sys
    # for running firmata
    from pyfirmata import Arduino, util
    import time
    import os

    # handle keyboard interrupt for quitting the program
    import signal
    import sys

    # define a graceful way to exit if no frames a being shown
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        board.digital[which_pin].write(False)
        print('cleaned headers!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop recording')

    # get the serial port for firmata
    # double - todo not good
    def get_serial_port():
        ser_devs = [dev for dev in os.listdir('/dev') if dev.startswith('ttyAC')]
        if len(ser_devs) > 0:
            return '/dev/'+ser_devs[0]
        return None


    # run the function to get the port
    port = get_serial_port()
    # and open connection using pyfirmata
    print('opening '+port+'...')
    board = Arduino(port)
    print(port+' is open.')
    # use the 12th pin for the blinking
    which_pin = 12

    # what is the time in s pr blink
    blink_time = 0.150
    # start by setting it down
    board.digital[which_pin].write(0)

    # and wait a moment for the cams to start running
    initial_waiting = 0
    print('arduino waiting for '+str(initial_waiting)+' seconds.')
    time.sleep(initial_waiting)

    # and now just run the blinking loop
    print('starting blink every '+str(blink_time) +' s on pin '+str(which_pin)+'.')
    while True:
        time.sleep(blink_time)
        board.digital[which_pin].write(0)
        time.sleep(blink_time)
        board.digital[which_pin].write(1)


# also define the firmata
def blink_using_firmata_random():
    # for getting the port
    import serial, sys
    # for running firmata
    from pyfirmata import Arduino, util
    import time
    import os

    # handle keyboard interrupt for quitting the program
    import signal
    import sys

    # define a graceful way to exit if no frames a being shown
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        board.digital[which_pin].write(False)
        print('cleaned headers!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop recording')

    # get the serial port for firmata
    # double - todo not good
    def get_serial_port():
        ser_devs = [dev for dev in os.listdir('/dev') if dev.startswith('ttyAC')]
        if len(ser_devs) > 0:
            return '/dev/'+ser_devs[0]
        return None


    # run the function to get the port
    port = get_serial_port()
    # and open connection using pyfirmata
    print('opening '+port+'...')
    board = Arduino(port)
    print(port+' is open.')
    # use the 12th pin for the blinking
    which_pin = 12

    # what is the time in s pr blink
    blink_time = 0.150
    # start by setting it down
    board.digital[which_pin].write(0)

    # and wait a moment for the cams to start running
    initial_waiting = 0
    print('arduino waiting for '+str(initial_waiting)+' seconds.')
    time.sleep(initial_waiting)

    # and now just run the blinking loop
    print('starting blink every '+str(blink_time) +' s on pin '+str(which_pin)+'.')
    while True:
        time.sleep(blink_time)
        board.digital[which_pin].write(0)
        time.sleep(blink_time+np.random.uniform(low = 0.0, high = .200))
        board.digital[which_pin].write(1)




#%% also define the firmata
def blink_using_firmata_random_sound():
    # for getting the port
    import serial, sys
    # for running firmata
    from pyfirmata import Arduino, util
    import time
    import os

    # handle keyboard interrupt for quitting the program
    import signal
    import sys

    # define a graceful way to exit if no frames a being shown
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        board.digital[which_pin].write(False)
        print('cleaned headers!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop recording')

    # get the serial port for firmata
    # double - todo not good
    def get_serial_port():
        ser_devs = [dev for dev in os.listdir('/dev') if dev.startswith('ttyAC')]
        if len(ser_devs) > 0:
            return '/dev/'+ser_devs[0]
        return None

    def buzz_piezo():
        audio_time = 1/100.
        # do four quick BEEPS
        for _ in range(40):
            board.digital[9].write(0)
            time.sleep(audio_time)
            board.digital[9].write(1)
            time.sleep(audio_time)
        board.digital[9].write(0)

    def beep_speaker():
        beep_time = .1
        # do four quick BEEPS
        board.digital[10].write(0)
        board.digital[10].write(1)
        time.sleep(beep_time)
        board.digital[10].write(0)


    # buzz_piezo()
    # beep_speaker()
    # run the function to get the port
    port = get_serial_port()

    # and open connection using pyfirmata
    print('opening '+port+'...')
    board = Arduino(port)
    print(port+' is open.')
    # use the 12th pin for the blinking
    which_pin = 12

    # what is the time in s pr blink
    blink_time = 0.150
    # start by setting it down
    board.digital[which_pin].write(0)

    # and wait a moment for the cams to start running
    initial_waiting = 0
    print('arduino waiting for '+str(initial_waiting)+' seconds.')
    time.sleep(initial_waiting)

    # and now just run the blinking loop
    print('starting blink every '+str(blink_time) +' s on pin '+str(which_pin)+'.')
    blink_counter = 0
    next_blink = 20
    while True:
    # for _ in range(300):
        # these are the LED blinks:
        time.sleep(blink_time)
        board.digital[which_pin].write(0)
        time.sleep(blink_time+np.random.uniform(low = 0.0, high = .200))
        board.digital[which_pin].write(1)

        # update the blink counter
        blink_counter += 1

        if blink_counter > next_blink:
            # Give a buzz every ~30 blinks
            # buzz_piezo()
            beep_speaker()
            # and reset the blinking counter
            blink_counter = 0
            # and pull a random next time for a buzz
            next_blink = np.random.randint(25,50)

    board.digital[which_pin].write(0)
