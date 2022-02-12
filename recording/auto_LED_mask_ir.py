#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:34:45 2018

@author: chrelli

USAGE python draw_depth_roi.py -ncams 4

AUTOMATIC function to generate the LED roi!

added a moe clever way to look for the blinking LED!
"""


#%% Import the nescessary stuff
# basic OS stuff
import time, os, sys, shutil
# add the realsense library
sys.path.append(r'/usr/local/lib')
# and load it!
import pyrealsense2 as rs


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
from multiprocessing import Process

# import handy Functions
from utils.common_utils import *

#%% Constants
frame_width,frame_height  = 640,480 # Should have as defaults only one place?
# frame_width,frame_height  = 848,480 # Should have as defaults only one place?

fps_choice = 30

# get a tring with the current time!

timestr = time.strftime("%Y%m%d-%H%M%S")

# reset the folder
data_folder = '/media/chrelli/Data0'
top_folder = data_folder + '/led_mask_' + timestr
reset_folder_if_present(top_folder)


#%% Parse some inputs

import argparse

parser = argparse.ArgumentParser(description='Atomatically blinks arduino LED and determines mask (where it is visible in the color camera view). Shows the plots for one second.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ncams', type=int, default = 4 , choices=[1,2,3,4],
                    help='number of cameras to stream')

parser.add_argument("--noplots", help="disables plotting",
                    action="store_false")


args = parser.parse_args()


#%%
# for running firmata
from pyfirmata import Arduino, util

# run the function to get the port
port = get_serial_port()
# and open connection using pyfirmata
print('opening '+port+'...')
board = Arduino(port)
print(port+' is open.')

# use the 12th pin for the blinking
which_pin = 12
# what is the time in s pr blink
blink_time = 0.050
# start by setting it down

# initial state of the led
led_state = False
# flip he led every n frames, e.g. every 2nd frame
led_flip = 2
board.digital[which_pin].write(led_state)

# and wait a moment for the cams to start running
initial_waiting = 0
print('arduino waiting for '+str(initial_waiting)+' seconds.')
time.sleep(initial_waiting)


# open up a realsense context and get a list of the devices!
ctx = rs.context()

devices = [ctx.devices[i] for i in range(args.ncams)]
# sort the devices by their serial numbers
serials = [devices[i].get_info(rs.camera_info.serial_number) for i in range(args.ncams)]
devices = [x for _,x in sorted(zip(serials,devices))]



# open each device sequentially

for which_device in range(args.ncams):

    # get the serial of that device
    device_serial = devices[which_device].get_info(rs.camera_info.serial_number)

    # first, open up a  config
    config = rs.config()

    # then open a pipeline
    pipeline = rs.pipeline()

    # enable the selected device and streams
    config.enable_device(device_serial);
#    config.enable_stream(rs.stream.depth, frame_width,frame_height, rs.format.z16, fps_choice)
    config.enable_stream(rs.stream.depth, frame_width,frame_height, rs.format.z16, fps_choice)

#    config.enable_stream(rs.stream.color, frame_width,frame_height, rs.format.bgr8, fps_choice)
    config.enable_stream(rs.stream.color, 848,480, rs.format.bgr8, fps_choice)

    # IR
    config.enable_stream(rs.stream.infrared, 1, frame_width,frame_height, rs.format.y8, fps_choice)

    # Start streaming
    cfg = pipeline.start(config)

    # create an align object
    # alternative is to align to color, faster but less precise: align_to = rs.stream.color
    align_to = rs.stream.depth
    align = rs.align(align_to)

    # wait 1 s for the cam to wake up
    time.sleep(1)

    # recieve 30 frames and
    frames_to_capture = 60

    # grayscale holder matrix to keep the grayscale images as a stack
    g_holder = np.empty((frame_height,frame_width,frames_to_capture))
    d_holder = np.empty((frame_height,frame_width,frames_to_capture))

#    g_holder = np.empty((480,848,frames_to_capture))

    # run the cam for 30 frames
    for ii in range(frames_to_capture):
        # flip the led every n frames
        if ii%led_flip == 0:
            led_state = not(led_state)
            board.digital[which_pin].write(led_state)

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # get the raw images like this:

        # run the alignment process
       # aligned_frames = align.process(frames)
#        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
#        color_frame = aligned_frames.get_color_frame()

        depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image

        infrared_frame = frames.get_infrared_frame()

#        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        d = np.asanyarray(depth_frame.get_data())
#        c = np.asanyarray(color_frame.get_data())
        c = np.asanyarray(infrared_frame.get_data())

        # convert to grayscale and save to matrix stack
#        g_holder[:,:,ii] = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        g_holder[:,:,ii] = c
        d_holder[:,:,ii] = d

        # blur a little bit
        g_holder[:,:,ii] = cv2.GaussianBlur(g_holder[:,:,ii],(5,5),0)


    # calculate the difference in time, so along the last axis:
    g_diff = np.diff(g_holder)

    # take the absolute value of the difference!
    g_diff = np.abs(g_diff)

    # and sum in along the time dimension. also rund to stay within the uint8 space
    g_diff = np.round(np.mean(g_diff,axis=2))

    # blur abit again and cut at 75 pct of max
    g_diff=cv2.GaussianBlur(g_diff,(15,15),0)
    g_half = .60 * np.max(g_diff)
    ret,auto_mask = cv2.threshold(g_diff,g_half,255,cv2.THRESH_BINARY)
    # convert to uint8!
    auto_mask=auto_mask.astype('uint8')

    # write the mask to the constants folder
    which_label = 'auto_led'
    cv2.imwrite(top_folder+'/dev'+str(which_device)+'_roi_frame_'+which_label+'.png', auto_mask)

    # calculate the median depth_frame
    d_median = np.median(d_holder,axis = 2)

    # and write the background to the constants folder
    which_label = 'background_depth'
    np.save(top_folder+'/dev'+str(which_device)+'_roi_frame_'+which_label+'.npy', d_median)

    if args.noplots:

        # c is already bgr now
        # make the mask into bgr as well
#        color_mask = cv2.cvtColor(auto_mask,cv2.COLOR_GRAY2BGR)
        color_mask = auto_mask

        # select color
        txt_color = (Color('Peru').rgb)

        # convert to 8 bit color
        txt_color=tuple(255*x for x in txt_color)

        cv2.putText(color_mask, 'detected LED region of dev'+str(which_device), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color)

        plt.imshow(np.concatenate((c,color_mask),axis=1))
        plt.savefig(top_folder+'/dev_'+str(which_device)+'_side_by_side'+'.png')

        cv2.imshow('masked', np.concatenate((c,color_mask),axis=1))
        cv2.waitKey(800) # show for one second

        # d_median_show = (d_median*100) % 65535
        # d_median_show=d_median_show.astype('uint16')
        # cv2.imshow('median depth', d_median_show)
        # cv2.waitKey(400) # show for one second




    # finally stop reading from the pipeline
    pipeline.stop()
