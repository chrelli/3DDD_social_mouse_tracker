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

NYP: TRYING out saving directly to npy file!
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

args = parser.parse_args()



#%% Constants
frame_width,frame_height  = 640,480
# frame_width,frame_height  = 848,480

fps_choice = args.fps
# number of padding digits for the frame numbers
n_padding_digits = 8

print('# cameras: '+str(args.ncams))
print('Frame size is '+str(frame_width)+'x'+str(frame_height)+' pixels.')
print('Grabbing frames at '+str(fps_choice)+' fps')

# OK, now set up timed folders
# get the current timestring
timestr = time.strftime("%Y%m%d-%H%M%S")

# reset the folders
top_folder_0 = '/media/chrelli/Data0' + '/recording_' + timestr
top_folder_1 = '/media/chrelli/Data1' + '/recording_' + timestr

reset_folder_if_present(top_folder_0)
reset_folder_if_present(top_folder_1)


# also make the numpy folders
npy_folder_0 = top_folder_0+'/npy_raw'
npy_folder_1 = top_folder_1+'/npy_raw'

reset_folder_if_present(npy_folder_0)
reset_folder_if_present(npy_folder_1)



#%% 8 bit color setup
fps_color = (Color('HotPink').rgb)
ts_color = (Color('Peru').rgb)

# convert to 8 bit color
fps_color=tuple(255*x for x in fps_color)
ts_color=tuple(255*x for x in ts_color)

#%% Block for running
# open the pyrealsense server
# open up a realsense context and get a list of the devices!
ctx = rs.context()


devices = [ctx.devices[i] for i in range(args.ncams)]
# sort the devices by their serial numbers
serials = [devices[i].get_info(rs.camera_info.serial_number) for i in range(args.ncams)]
devices = [x for _,x in sorted(zip(serials,devices))]

# set the start time for the unix time stamp
start_time = time.time()




def sub_function_trick(which_device,top_folder):
    # pull out the logicals, slightly silly, maybe rename, look up best practice
    show_frames = args.plots
    print('show frames')
    print(show_frames)
    print('show frames')

    save_cad = not args.nocad
    use_roi = not args.noroi
    # look for size of frames would alsso work
    if use_roi:
        np.savetxt(top_folder+'/use_roi.csv',[use_roi],delimiter=',')
    else:
        np.savetxt(top_folder+'/use_roi.csv',[use_roi],delimiter=',')

    ####################
    #
    # DEVICE SETUP BLOCK 2
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
    preset_folder = 'presets/'
    if device_serial[:3] == '740':
        preset_name = 'master60pp_640'
    else:
        preset_name = 'slave60pp_640'

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

    # enable the selected device and streams
    config.enable_device(device_serial);
    config.enable_stream(rs.stream.depth, frame_width,frame_height, rs.format.z16, fps_choice)
    config.enable_stream(rs.stream.infrared, 1, frame_width,frame_height, rs.format.y8, fps_choice)
    # config.enable_stream(rs.stream.color, frame_width,frame_height, rs.format.bgr8, fps_choice)


    print("PING after enabling the sync mode is {}".format(device.first_depth_sensor().get_option(rs.option.inter_cam_sync_mode)))

    # Start streaming
    cfg = pipeline.start(config)

    # make filters for depth
    dec_filter = rs.decimation_filter()   # Decimation - reduces depth frame density
    spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
    # temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise


    print('dev '+str(which_device)+' serial is ' + device_serial)
    # Use the first three digits of the serial as a string to tag the device:
    device_tag = device_serial[0:3]

    if show_frames:
        # open a window for cv2
        window_title = "dev"+str(which_device)
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

    with open(top_folder+'/parameters_d_'+str(which_device)+'.csv','w') as intrfile:
        writer = csv.writer(intrfile, delimiter=',')
        writer.writerow(parameternames)
        writer.writerow(parameters)


    # this is how to get the intrinsics
    profile = cfg.get_stream(rs.stream.infrared) # Fetch stream profile for depth stream
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
    with open(top_folder+'/parameters_ir_'+str(which_device)+'.csv','w') as intrfile:
        writer = csv.writer(intrfile, delimiter=',')
        writer.writerow(parameternames)
        writer.writerow(parameters)

    # also get the extrinsics between two streams
    print("extrinsics!")
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    extr = profile.as_video_stream_profile().get_extrinsics_to(cfg.get_stream(rs.stream.infrared)) # Downcast to video_stream_profile
    print(extr)
    print(extr.rotation)
    np.save(top_folder+'/extr_R_depth_to_color_'+str(which_device)+'.npy',extr.rotation)
    np.save(top_folder+'/extr_t_depth_to_color_'+str(which_device)+'.npy',extr.translation)
    print("extrinsics!")



    # load the automatic led mask from the constants folder!
    # led_mask,led_logic,led_centroid = load_auto_roi(which_device)


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
        which_label = 'background_depth'
        d_median = np.load(constant_folder+'/dev'+str(which_device)+'_roi_frame_'+which_label+'.npy')

        return led_mask,led_logic,led_centroid,d_median

    led_mask,led_logic,led_centroid,d_median = load_auto_roi_with_background(which_device)


    # NPY ADDITION
    npy_folder = top_folder+'/npy_raw'


    # open a file for time stamps
    tsfile = open(top_folder+'/timestamps_'+str(which_device)+'.csv','w')

    print('starting to stream from device '+str(which_device)+'!')

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
            # R E A D   B L O C K
            #################################

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # get the frame numbers and time stamps
            ts = frames.get_timestamp()
            fn = frames.get_frame_number()

            # get the unix time stamp
            ts_unix = time.time()-start_time


            depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            infrared_frame = frames.get_infrared_frame()


            filtered = dec_filter.process(depth_frame)
            filtered = spat_filter.process(filtered)
            # filtered = temp_filter.process(filtered)
            depth_frame = filtered

            # cad_frame = frames.get_color_frame()



            # Convert images to numpy arrays
            depth = np.asanyarray(depth_frame.get_data())
            # cad = np.asanyarray(cad_frame.get_data())
            c = np.asanyarray(infrared_frame.get_data())

            # get the LED value, round it a bit, could be profiled
            led_stamp = c[led_centroid[1],led_centroid[0]]

            # this is the writing block for the csv file, frame number and time stamp!
            tsfile.write(str(FRAME_CLOCK)+','+str(fn)+','+str(ts)+','+str(ts_unix)+','+str(led_stamp)+'\n')

            cv2.imwrite(npy_folder+'/dev'+str(which_device)+'_d_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.png', depth)
            cv2.imwrite(npy_folder+'/dev'+str(which_device)+'_ir_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.png', c)

            # np.save(npy_folder+'/dev'+str(which_device)+'_d_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.npy',depth, allow_pickle = False)
            # np.save(npy_folder+'/dev'+str(which_device)+'_cad_'+str(FRAME_CLOCK).rjust(n_padding_digits,'0')+'.npy',cad[::2,::2,:], allow_pickle = False)


            # UPDATE CLOCK
            FRAME_CLOCK += 1

            if show_frames:
                    # cv2.putText(cad, window_title+', fps: '+str(fps)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, fps_color,2)
                    # cv2.putText(cad, str(round(ts)), (0, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, ts_color)
                    cv2.imshow(window_title+'cad', c[::2,::2])

            if cv2.waitKey(1) & 0xFF == ord('q') :
                # looks for a small q to nbe pressed
                # close the time stamp file
                tsfile.close
                pipeline.stop()
                print('device '+str(which_device)+' is now closed!')
                break

    finally:
        print('caught ctrl+C in dev')
        tsfile.close
        pipeline.stop()
        print('device '+str(which_device)+' is now closed!')

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
        Process(target=blink_using_firmata_random_sound).start()

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
