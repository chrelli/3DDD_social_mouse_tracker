#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:36:15 2018

@author: chrelli
added unix time stamps to the first camera!

Ways to slim down the data:
    no unix time stamps?
    no color frame showing? - yes, helps a lot!
    no png compression? Totally fine at 30 fps!

Majow to do list:
    - use arduino to synchronize? Yes, could send out synchronization time code to another unit: Problem: doesn't account for delay of arriving frames

    - use depth roi to slim down writing footprint

    - use LED roi to get blinking time stamps

    -

## with connected device cam
from pyrealsense import offline
offline.save_depth_intrinsics(dev)
"""

#%% Import the nescessary stuff
# basic OS stuff
import time, os, sys, shutil
import json
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
# add the realsense library
sys.path.append(r'/usr/local/lib')
# and load it!
import pyrealsense2 as rs


#import multiprocessing
from multiprocessing import Process

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *



#%% Parse some inputs

import argparse

parser = argparse.ArgumentParser(description='Records cad and d images with no roi cut to disk. Also records timestamps and led traces using the auto LED mask. Currently, with no ROI, the program maxes out disk write speed around 45 fps.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ncams', type=int, default = 4 , choices=[1,2,3,4],
                    help='number of cameras to stream')

parser.add_argument('--fps',type=int, default = 30 , choices=[30,60],
                    help='select fps to stream')

#parser.add_argument("--singlecore", help="disables mult.proc. for debugging on macbook, overrides ncams to 1",
#                    action="store_true")

parser.add_argument("--plots", help="shows the live video while recording",
                    action="store_true")

args = parser.parse_args()



#%% Constants
# frame_width,frame_height  = 848,480
frame_width,frame_height  = 640,480


fps_choice = args.fps
# number of padding digits for the frame numbers
n_padding_digits = 8

print('# cameras: '+str(args.ncams))
print('Frame size is '+str(frame_width)+'x'+str(frame_height)+' pixels.')
print('Grabbing frames at '+str(fps_choice)+' fps')


# get the current timestring
timestr = time.strftime("%Y%m%d-%H%M%S")

# reset the folder
#data_folder = '/media/chrelli/Data0'
#top_folder = data_folder + '/calibration_' + timestr
#reset_folder_if_present(top_folder)
#
#top_folder_0 = top_folder
#top_folder_1 = top_folder

# reset the folders
top_folder_0 = '/media/chrelli/Data0' + '/calibration_' + timestr
top_folder_1 = '/media/chrelli/Data1' + '/calibration_' + timestr

reset_folder_if_present(top_folder_0)
reset_folder_if_present(top_folder_1)

# also make the numpy folders
npy_folder_0 = top_folder_0+'/npy_raw'
npy_folder_1 = top_folder_1+'/npy_raw'

reset_folder_if_present(npy_folder_0)
reset_folder_if_present(npy_folder_1)


#%% 8 bit color setup
fps_color = (Color('White').rgb)
ts_color = (Color('Peru').rgb)

# convert to 8 bit color
fps_color=tuple(255*x for x in fps_color)
ts_color=tuple(255*x for x in ts_color)

#%% Block for running
# open the pyrealsense server
#serv = pyrs.Service()

# set the start time for the unix time stamp
start_time = time.time()



# open up a realsense context and get a list of the devices!
ctx = rs.context()

devices = [ctx.devices[i] for i in range(args.ncams)]
# sort the devices by their serial numbers
serials = [devices[i].get_info(rs.camera_info.serial_number) for i in range(args.ncams)]
devices = [x for _,x in sorted(zip(serials,devices))]



def sub_function_trick(which_device,top_folder):
    show_frames = args.plots


    ####################
    #
    # DEVICE SETUP BLOCK
    #
    #####################

    # get the serial of that device
    device =  devices[which_device]
    device_serial = device.get_info(rs.camera_info.serial_number)

    #set the preset
    advnc_mode = rs.rs400_advanced_mode(device)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
    # run like
    # advnc_mode.load_json(json_string)

    # load the preset here!
    preset_folder = '/home/chrelli/git/3d_sandbox/mycetrack0p8/presets/'
    if device_serial[:3] == '740':
        preset_name = 'master60pp'
    else:
        preset_name = 'slave60pp'

    jsonFile = preset_folder+preset_name+'.json'
    jsonObj = json.load(open(jsonFile))
    json_string = str(jsonObj).replace("'", '\"')
    print("Configuration " + jsonFile + " loaded");
    time.sleep(1.)
    advnc_mode.load_json(json_string)
    print("Configuration " + jsonFile + " applied!");

    if device_serial[:3] == '740':
        # master
        targetSyncMode = 1
    else:
        # slave
        targetSyncMode = 2
    device.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, targetSyncMode)


    # first, open up a  config
    config = rs.config()

    # then open a pipeline
    pipeline = rs.pipeline()

    # enable the selected device and streams # RGB SPACE HERE
    config.enable_device(device_serial);
    config.enable_stream(rs.stream.depth, frame_width,frame_height, rs.format.z16, fps_choice)
#        config.enable_stream(rs.stream.color, frame_width,frame_height, rs.format.rgb8, fps_choice)

    config.enable_stream(rs.stream.color, frame_width,frame_height, rs.format.rgb8, fps_choice)
    config.enable_stream(rs.stream.infrared,1, frame_width,frame_height, rs.format.y8, fps_choice)

    print("PING after enabling the sync mode is {}".format(device.first_depth_sensor().get_option(rs.option.inter_cam_sync_mode)))


    # Start streaming, call the stream 'cfg' for some reason, as pr example
    cfg = pipeline.start(config)

    # create an align object
    # alternative is to align to color, faster but less precise: align_to = rs.stream.color
    align_to = rs.stream.depth
    align = rs.align(align_to)


    print('dev '+str(which_device)+' serial is ' + device_serial)
    # Use the first three digits of the serial as a string to tag the device:
    device_tag = device_serial[0:3]

    if show_frames:
        # open a window for cv2
        window_title = "dev"+str(which_device)+"(#" + device_tag + ")"
        cv2.namedWindow(window_title+'cad')

        # block for setting up a low-level fps estimation,
        cnt = 0 # a counter
        last = time.time() # start_time
        fps = 0 # initial fps value

    # save the camera intrinsics
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = cfg.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print ("Depth Scale is: " , depth_scale)

    # this is how to get the intrinsics
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile


    #% now make file and save time stamps and depth scaling and intrinsics etc
    # use the old naming convention

    parameternames = np.array(['cam_params.fx',
                       'cam_params.fy',
                       'cam_params.ppx',
                       'cam_params.ppy',
                       'd_scale',
                       'fps_choice',
                       'frame_width',
                       'frame_height'])
    parameters = np.array([intr.fx,
                       intr.fy,
                       intr.ppx,
                       intr.ppy,
                       depth_scale,
                       fps_choice,
                       intr.width,
                       intr.height])

    # open a file for writint the parameters
    with open(top_folder+'/parameters_'+str(which_device)+'.csv','w') as intrfile:
        writer = csv.writer(intrfile, delimiter=',')
        writer.writerow(parameternames)
        writer.writerow(parameters)


    # load the automatic led mask from the constants folder!
    led_mask,led_logic,led_centroid = load_auto_roi(which_device)


    # open a file for time stamps
    tsfile = open(top_folder+'/timestamps_'+str(which_device)+'.csv','w')


#    ## HF try to open an HF file
#    import h5py
#    #TODO input from somewhere
#    hf = h5py.File(top_folder+'/dev'+str(which_device)+'_d_'+'.h5', 'w')
#    # also open one for the cad
#    hf_cad = h5py.File(top_folder+'/dev'+str(which_device)+'_cad_'+'.h5', 'w')

    # NPY ADDITION
    npy_folder = top_folder+'/npy_raw'


    # open a file for led stamps
#        ledsfile = open(top_folder+'/ledstamps_'+str(which_device)+'.csv','w')

    print('starting to stream from device '+str(which_device)+'!')
    # wait for a bit for the cam to warm up
    # and loop over 30 frames
    warmup_time = 2 # seconds
    warmup = 0
    while warmup < fps_choice*warmup_time:
        frames = pipeline.wait_for_frames()
        warmup += 1
    print('device '+str(which_device)+' is warmed up!')


    # START A CLOCK FOR THE FRAMES!
    FRAME_CLOCK = 0
    try:
        while True:

            if show_frames:
                # for counting frame rate
                cnt += 1
                if (cnt % 10) == 0:
                    now = time.time() # after 10 frames
                    dt = now - last # how long did it take?
                    fps = 10/dt # calculate frame rate
                    last = now # assign a new value to the 'last time'

            #################################
            #
            # R E A D   B L O C K
            #
            #################################

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # get the frame numbers and time stamps
#            ts = round(frames.get,2)
            ts = frames.get_timestamp()
            fn = frames.get_frame_number()

            # get the unix time stamp
            ts_unix = time.time()-start_time

            # run the alignment process
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            cad_frame = aligned_frames.get_color_frame()
            # also get one for the LED
            # depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()
            infrared_frame = frames.get_infrared_frame()


            # Convert images to numpy arrays
            depth = np.asanyarray(depth_frame.get_data())
            cad = np.asanyarray(cad_frame.get_data())
            c = np.asanyarray(infrared_frame.get_data())


            # get the LED value, round it a bit, could be profiled
            led_stamp = c[led_centroid[1],led_centroid[0]]

            # this is the writing block for the csv file, frame number and time stamp!
#            tsfile.write(str(FRAME_CLOCK)+','+str(fn)+','+str(ts)+','+str(ts_unix)+','+str(single_pixel_RGB2GRAY(led_stamp))+'\n')
            tsfile.write(str(FRAME_CLOCK)+','+str(fn)+','+str(ts)+','+str(ts_unix)+','+str(led_stamp)+'\n')

            # this is the writing block for the csv file, frame number and time stamp!
            #TODO put led with the others in same file?
#            ledsfile.write(str(single_pixel_RGB2GRAY(led_stamp))+'\n')


            # write the depth frames to tiff (replace: send to queue)
#            cv2.imwrite(top_folder+'/dev'+str(which_device)+'_d_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.png', depth)
#            cv2.imwrite(top_folder+'/dev'+str(which_device)+'_cad_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.png', cad)

#            hf.create_dataset(str(FRAME_CLOCK), data=depth)
#            hf_cad.create_dataset(str(FRAME_CLOCK), data=cad)

            np.save(npy_folder+'/dev'+str(which_device)+'_d_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.npy',depth, allow_pickle = False)
            np.save(npy_folder+'/dev'+str(which_device)+'_cad_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.npy',cad, allow_pickle = False)


            # UPDATE CLOCK
            FRAME_CLOCK += 1

            #
            if show_frames:
                # add text and show the CAD frames
                cv2.putText(cad, window_title+', fps: '+str(fps)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, fps_color)
                cv2.putText(cad, str(round(ts)), (0, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, ts_color)
                cv2.imshow(window_title+'cad', cad)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                # looks for a small q to nbe pressed
                # close the time stamp file
                tsfile.close

                # close the hf file
#                hf.close()
#                hf_cad.close()
#                ledsfile.close

                # stop the device
                pipeline.stop()
                print('pipeline from device '+str(which_device)+' is now closed!')
                break

    finally:
        tsfile.close

        # close the hf file
#        hf.close()
#        hf_cad.close()

        # stop the device
        pipeline.stop()
        print('pipeline from device '+str(which_device)+' is now closed!')


#%% define helping funtions for the multiprocessing
# these functions have to not be iterable.

def read_device_0():
    print('starting camera 1!')
    which_device = 0
    top_folder = top_folder_0
    sub_function_trick(which_device,top_folder)

def read_device_1():
    print('starting camera 2!')
    which_device = 1
    top_folder = top_folder_0
    sub_function_trick(which_device,top_folder)

def read_device_2():
    print('starting camera 3!')
    which_device = 2
    top_folder = top_folder_1
    sub_function_trick(which_device,top_folder)

def read_device_3():
    print('starting camera 4!')
    which_device = 3
    top_folder = top_folder_1
    sub_function_trick(which_device,top_folder)


#%% run the processes on independent cores
from multiprocessing import Process
if __name__ == '__main__':
    if args.ncams == 4:
        print('starting 4 cams, with multiprocessing!')
        # start 4 worker processes
        Process(target=read_device_0).start()
        time.sleep(3.)
        Process(target=read_device_1).start()
        Process(target=read_device_2).start()
        Process(target=read_device_3).start()
        Process(target=blink_using_firmata_random).start()

    elif args.ncams == 3:
        print('starting 3 cams, with multiprocessing!')
        Process(target=read_device_0).start()
        Process(target=read_device_1).start()
        Process(target=read_device_2).start()
        Process(target=blink_using_firmata).start()

    elif args.ncams == 2:
        print('starting 2 cams, with multiprocessing!')
        Process(target=read_device_0).start()
        Process(target=read_device_1).start()
        Process(target=blink_using_firmata).start()

    elif args.ncams == 1:
        print('starting 1 cam, with multiprocessing!')
        Process(target=read_device_0).start()
        Process(target=blink_using_firmata).start()
