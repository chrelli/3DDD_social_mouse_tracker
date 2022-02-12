#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:11:58 2018

@author: chrelli

"""

#%% Import the nescessary stuff


import sys
sys.path.append(r'/usr/local/lib')

import pyrealsense2 as rs
import numpy as np
import cv2
import json

import time, os, shutil

import matplotlib.pyplot as plt
import csv

from colour import Color



import argparse

parser = argparse.ArgumentParser(description='Records cad and d images with roi cut to disk. Also records timestamps and led traces using the auto LED mask. Currently, with no ROI, the program maxes out disk write speed around 45 fps.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ncams', type=int, default = 4 , choices=[1,2,3,4],
                    help='number of cameras to stream')

parser.add_argument('--fps',type=int, default = 60 , choices=[30,60,90],
                    help='select fps to record')

parser.add_argument("--noroi", help="disables the roi cutting",
                    action="store_true")

parser.add_argument("--nocad", help="disables recording of the cad image",
                    action="store_true")

parser.add_argument("--plots", help="shows the live video while recording",
                    action="store_true")

parser.add_argument("--nolaser", help="turn off the laser",
                    action="store_true")

args = parser.parse_args()















#%% Constants
frame_width,frame_height  = 640,480 # Should have as defaults only one place?
# frame_width,frame_height  = 848,480 # Should have as defaults only one place?

fps_choice = args.fps
# fps_choice = 60

# colors of the
fps_color = (Color('pink').rgb)
ts_color = (Color('Peru').rgb)

# convert to 8 bit color
fps_color=tuple(255*x for x in fps_color)
ts_color=tuple(255*x for x in ts_color)

















# open a context
ctx = rs.context()

# theses are the four devices
# dev0 = ctx.devices[0]
# dev1 = ctx.devices[1]
# dev2 = ctx.devices[2]
# dev3 = ctx.devices[3]



devices = [ctx.devices[i] for i in range(args.ncams)]
# sort the devices by their serial numbers
serials = [devices[i].get_info(rs.camera_info.serial_number) for i in range(args.ncams)]
devices = [x for _,x in sorted(zip(serials,devices))]

# set the start time for the unix time stamp
start_time = time.time()


device_list = [ctx.devices[i] for i in range(4)]

print("Connected devices:")
for i in device_list:
    print(i)





# define a subfunction which runs everything
def sub_function_trick(device):
    # get the serial of that device
    device_serial = device.get_info(rs.camera_info.serial_number)

    #set the preset
    advnc_mode = rs.rs400_advanced_mode(device)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
    # run like
    # advnc_mode.load_json(json_string)

    # load the preset here!
    preset_folder = 'presets/'
    if device_serial[:3] == '740':
        preset_name = 'master60pp_640'
    else:
        preset_name = 'slave60pp_640'

    if args.nolaser:
        preset_name = preset_name + '_nolaser'


    jsonFile = preset_folder+preset_name+'.json'
    jsonObj = json.load(open(jsonFile))
    json_string = str(jsonObj).replace("'", '\"')
    print("Configuration " + jsonFile + " loaded");
    time.sleep(1.)
    advnc_mode.load_json(json_string)
    print("Configuration " + jsonFile + " applied!");

    # first, open up a  config
    config = rs.config()



    # then open a pipeline
    pipeline = rs.pipeline()

    # enable the selected device and streams
    config.enable_device(device_serial);
    config.enable_stream(rs.stream.depth, frame_width,frame_height, rs.format.z16, fps_choice)
    config.enable_stream(rs.stream.infrared, 1, frame_width,frame_height, rs.format.y8, fps_choice)

    config.enable_stream(rs.stream.color, frame_width,frame_height, rs.format.bgr8, fps_choice)

    # Start streaming
    cfg = pipeline.start(config)

    # make filters
    dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
    spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
    temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise


    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = cfg.get_device().first_depth_sensor()
    # color_sensor = cfg.get_device().first_color_sensor()

    # this is how to get the intrinsics
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    # align_to = rs.stream.color
    #align the color to the depth, is best, I think
    align_to = rs.stream.depth
    align = rs.align(align_to)

    # this is how to get the intrinsics, of the one which we align to!!
    # profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    # intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile

    # open a window for cv2
    window_title = "dev "+str(device_serial)
    cv2.namedWindow(window_title)

    # block for setting up a low-level fps estimation,
    cnt = 0 # a counter
    last = time.time() # start_time
    fps = 0 # initial fps value

    while True:

    # for counting frame rate
        cnt += 1
        if (cnt % 10) == 0:
            now = time.time() # after 10 frames
            dt = now - last # how long did it take?
            fps = 10/dt # calculate frame rate
            last = now # assign a new value to the 'last time'
        # NOW let's get the frames out

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # get the raw images like this:

        # run the alignment process
        depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        infrared_frame = frames.get_infrared_frame()

        color_frame = frames.get_color_frame()

        filtered = dec_filter.process(depth_frame)
        filtered = spat_filter.process(filtered)
        filtered = temp_filter.process(filtered)
        depth_frame = filtered

        # and time stamp
        ts = frames.get_timestamp()

        if not depth_frame:
            continue

        # Convert images to numpy arrays
        d = np.asanyarray(depth_frame.get_data())
        ir = np.asanyarray(infrared_frame.get_data())



        # convert image space from rgb to bgr, remove to decrease overhead
        #frame = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue
        c = np.asanyarray(color_frame.get_data())
        frame = c
        frame = (d%256).astype(np.uint8)



        frame = ir
        raw_frame = frame

        # add text and show the video
        cv2.putText(frame, window_title+', fps: '+str(fps)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, fps_color,2)
        cv2.putText(frame, str(round(ts)), (0, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, ts_color)
        cv2.imshow(window_title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()



            break

#%% NOW define helping funtions for the multiprocessing
# these functions have to not be iterable, otherwise multiprocessing will complain

def read_device_0():
    print('starting camera 0!')
    which_device = device_list[0]
    sub_function_trick(which_device)

def read_device_1():
    print('starting camera 1!')
    which_device = device_list[1]
    sub_function_trick(which_device)

def read_device_2():
    print('starting camera 2!')
    which_device = device_list[2]
    sub_function_trick(which_device)

def read_device_3():
    print('starting camera 3!')
    which_device = device_list[3]
    sub_function_trick(which_device)

#%% set up multiprocessing Pool
from multiprocessing import Process
if __name__ == '__main__':
    print('starting 4 cams, with multiprocessing!')
    # start 4 worker processes
    Process(target=read_device_0).start()
    time.sleep(3.)
    Process(target=read_device_1).start()
    Process(target=read_device_2).start()
    Process(target=read_device_3).start()
