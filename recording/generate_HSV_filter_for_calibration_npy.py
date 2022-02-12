#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:34:39 2018

@author: chrelli

file, which will replay the calibration trace

"""

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
from multiprocessing import Process

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *

import h5py

#%% Parse some inputs

import argparse

parser = argparse.ArgumentParser(description='Replays the calibration recording for HSV filetering and background subtraction to generate HSV values',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = parser.parse_args()


#%% Constants and folders
#frame_width,frame_height  = 640,480
#TODO don't hardcode the frame size
frame_width,frame_height  = 848,480


#%% THIS is just going to select the most recent calibration!
# TODO implement GIU so that you can (1) Use most recent (2) Select a specific folder

from utils.reading_utils import most_recent_calibration_folders

top_folder_0, top_folder_1 = most_recent_calibration_folders()

check_folder_if_present(top_folder_0)
check_folder_if_present(top_folder_1)




#%% define a function to list all the files in the calibration folder

#which_device = 0
#top_folder = top_folder_0

def replay_device(which_device,top_folder):

    # check if there are already values saved?

    hsv_values, gray_values = check_for_hsv_file(which_device,top_folder)

    # set to blue:
    hsv_values[0] = 98
    hsv_values[1] = 105

    # LOTS OF SETTING UP OF TRACKBAR
    #initialize empty argument for cv2
    def nothing(x):
        pass

    # open a cv2 window
    # open a window
    window_title = "dev"+str(which_device)+ ' replay of calib. cad'
    cv2.namedWindow('image')
    cv2.namedWindow('masked')

    #easy assigments
    hh='Hue High'
    hl='Hue Low'
    sh='Saturation High'
    sl='Saturation Low'
    vh='Value High'
    vl='Value Low'

    gc='Low cut of blur'
    ba='blur amount'


    # create the trackbars with ranges,
     # start at the hsv_values
    cv2.createTrackbar(hl, 'image',hsv_values[0],179,nothing)
    cv2.createTrackbar(hh, 'image',hsv_values[1],179,nothing)
    cv2.createTrackbar(sl, 'image',hsv_values[2],255,nothing)
    cv2.createTrackbar(sh, 'image',hsv_values[3],255,nothing)
    cv2.createTrackbar(vl, 'image',hsv_values[4],255,nothing)
    cv2.createTrackbar(vh, 'image',hsv_values[5],255,nothing)

    cv2.createTrackbar(gc, 'image',gray_values[0],255,nothing)
    cv2.createTrackbar(ba, 'image',gray_values[1],33,nothing)


    # create an hsv colorbar
    list_h = np.round(np.linspace(0,179,frame_width))
    list_s = np.ones(frame_width)*255
    list_v = np.ones(frame_width)*255

    full = np.empty(shape=(1,frame_width,3),dtype='uint8')
    full[:,:,0] = list_h
    full[:,:,1] = list_s
    full[:,:,2] = list_v

    imim = np.tile(full,(20,1,1))
    imim = cv2.cvtColor(imim, cv2.COLOR_HSV2BGR)
    cv2.imshow('hsv hue spectrum', imim)

    ###################
    # Block for reading from npy saving
    ###################
    d_list = get_file_shortlist(which_device,top_folder+'/npy_raw','d')
    cad_list = get_file_shortlist(which_device,top_folder+'/npy_raw','cad')


    n_frames = np.min((len(d_list),len(cad_list)))


    print(str(n_frames)+' frames to process from dev'+str(which_device)+'.')
    # set it up to read from a folder
#    reader_string = top_folder+'/dev'+str(which_device)+'_cad_'+'%0'+ str(n_padding_digits)+ 'd.png'
#    cap = cv2.VideoCapture(reader_string, cv2.CAP_IMAGES)

    # start a background subtraction algorithm
#    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    bgSubThreshold = 100
    fgbg = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=bgSubThreshold,detectShadows=False)


    frame_counter = 0
    while True:
#        time.sleep(1./30)
        # this uses the built-in reader
        # rval, c = cap.read()
        # hack
        # try to read the color frame
#        c = cv2.imread(top_folder+'/'+cad_list[frame_counter])
        # check if it worked, i.e. is the frame is none
        # r is True is all is good

        ###########
        # numpy reading
        ###########
        c = np.load(top_folder+'/npy_raw/'+cad_list[frame_counter])

        rval = c is not None

        frame_counter += 1
        # the rval checks if the reading was OK

#        print('frame: '+str(frame_counter)+', rval = '+str(rval))

        if rval:
            # convert to bgr
#            print(frame_counter)
#            print(c.shape)

            c = cv2.cvtColor(c,cv2.COLOR_RGB2BGR)
            c_raw = c
#            cv2.imshow('raw video',c)
            # do the backgrounhd step
            fgmask = fgbg.apply(c)

            res = cv2.bitwise_and(c, c, mask=fgmask)

            cv2.imshow('fgmask',res)


            # optional blur:
#            gaussian_size = 5
#            res = cv2.GaussianBlur(res,(gaussian_size,gaussian_size),0)

            # convert to grayscale
#            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)



            # apply the background

            #convert to HSV from BGR
            hsv=cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            blurValue = 7 # GaussianBlur parameter
            hsv = cv2.GaussianBlur(hsv, (blurValue, blurValue), 0)


            #read trackbar positions for all
            hul=cv2.getTrackbarPos(hl, 'image')
            huh=cv2.getTrackbarPos(hh, 'image')
            sal=cv2.getTrackbarPos(sl, 'image')
            sah=cv2.getTrackbarPos(sh, 'image')
            val=cv2.getTrackbarPos(vl, 'image')
            vah=cv2.getTrackbarPos(vh, 'image')

            gcut = cv2.getTrackbarPos(gc, 'image')
            blur_amount = cv2.getTrackbarPos(ba, 'image')
            # make sure that it is odd by simply forcing it!
            if blur_amount % 2 == 0:
                blur_amount += 1

            #make array for final values
            HSVLOW=np.array([hul,sal,val])
            HSVHIGH=np.array([huh,sah,vah])
            # and a full string for saving later!
            hsv_values = np.array([hul,huh,sal,sah,val,vah])
            gray_values = np.array([gcut,blur_amount])
            #apply the range on a mask
            mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)

            # optional blur step!

            mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
            mask = cv2.inRange(mask,gcut, 255)

            # show the masked image
            cv2.putText(mask, 'masked'+str(which_device) , (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)



            # qdd text for plotting
            cv2.putText(c_raw, 'raw calibration, dev'+str(which_device) , (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            cv2.putText(c_raw, 'frame #'+str(frame_counter)+'/'+str(n_frames) , (0, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            cv2.imshow('masked', c_raw)

            # and the raw image
            cv2.imshow('image', mask)


        # if the reader runs out of frames, reset the cap and run again
        if frame_counter == n_frames:
            frame_counter = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # save values
            with open(top_folder+'/hsv_values_'+str(which_device)+'.csv','w') as intrfile:
                writer = csv.writer(intrfile, delimiter=',')
                writer.writerow(np.hstack((hsv_values,gray_values)))

            #TODO for some reason, when there are trackbars, that window has to be killed by name
            # otherwise the sliders won't reset. cv2 bug?
            cv2.destroyWindow('image')
            cv2.destroyAllWindows


            break

replay_device(0,top_folder_0)
replay_device(1,top_folder_0)
replay_device(2,top_folder_1)
replay_device(3,top_folder_1)
