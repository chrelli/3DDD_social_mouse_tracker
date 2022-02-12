#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:53:19 2018

@author: chrelli
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

#%% arguments

import argparse

parser = argparse.ArgumentParser(description='Aligns the time stamps of the recorded (calibration or real) frames',formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--convert', type=str, default = 'both' , choices=['calib','scene','both'],
                    help='what traces to align')

parser.add_argument("--noshow", help="disables showing the plots, will just dump the csv and png's",
                    action="store_false")

parser.add_argument("--rtag", type=str, default = 'None', help="glob for a date tag in the recording folders")


args = parser.parse_args()


#%% load the calibration time steps

def load_calib_stamps(which_device,top_folder):
    # loads the frame numbers, time stamps, and unix time stamps, all are rescaled to start at zero!
    rawdata = np.genfromtxt(top_folder+'/timestamps_'+str(which_device)+'.csv', delimiter=',')
#    leds = np.genfromtxt(top_folder+'/ledstamps_'+str(which_device)+'.csv', delimiter=',')
#    next_frame_new = np.hstack(( np.diff(rawdata[:,1]) > 0,True))

#    fn = rawdata[:,0]
#    ts = rawdata[:,1]
#    nix = rawdata[:,2]
    #TODO fix in better way!
#    fn = rawdata[next_frame_new,1]
#    ts = rawdata[next_frame_new,2]
#    nix = rawdata[next_frame_new,3]
#    leds = rawdata[next_frame_new,-1] # it's the last one!

    # ADD this to reset the times to start at zero for each cam!

    # this will only happen once!x

    f_counter = rawdata[:,0]
    fn = rawdata[:,1]
    ts = rawdata[:,2]
    nix = rawdata[:,3]
    leds = rawdata[:,-1] # it's the last one!

    # use the median, very robust!
#    ts = ts - np.median(ts)
#    nix = nix - np.median(nix)
    # set the start time of both to be zero!
    print('start time')
    print(ts[0])

    ts = ts - ts[0]
    nix = nix - nix[0]

    #TODO HMMMMMMMMmmmmmmm weirdo issue, some times
    #the wait for frames will fail and
    #the same frame is passed twice



    return f_counter,fn,ts,nix,leds


#%% loop over the devices and set the references

#which_device,top_folder,ref_device,ref_folder,ShowPlots = 1,top_folder_0,0,top_folder_0,True

def align_time_stamps(which_device,top_folder,ref_device,ref_folder,ShowPlots):
    # i is the current, j is the reference
    f_counter_i,fn_i,ts_i,nix_i,leds_i = load_calib_stamps(which_device,top_folder)
    f_counter_j,fn_j,ts_j,nix_j,leds_j = load_calib_stamps(ref_device,ref_folder)

    # binarize the LED stamps, keep this function here
    def binary_led(leds):
        # cuts at 90% between the max and the mean
        cutoff = .7
        maxled = leds.max()
        meanled = np.mean(leds)
        print(meanled)
        cut = cutoff*(maxled-meanled)+meanled

        leds_binary = leds > cut
        return leds_binary

    # this binarization is not really nesc, but gets rid of small drifts
    # maybe profile if it helps or not? Dunno..
    leds_i = binary_led(leds_i)
    leds_j = binary_led(leds_j)

    # generate interpolation functions from sp
    from scipy.interpolate import interp1d # due to some scipy weirdness
    f_i = interp1d(ts_i,leds_i,fill_value="extrapolate")
    f_j = interp1d(ts_j,leds_j,fill_value="extrapolate")

#    from scipy.interpolate import UnivariateSpline # due to some scipy weirdness
#    f_i = UnivariateSpline(ts_i,leds_i,s=0,ext=0)
#    f_j = UnivariateSpline(ts_j,leds_j,s=0,ext=0)

    # time step now:
    dt = np.median(np.diff(ts_j))
    # upsample 10 times!, i.e. 1 10th framerate resolution
    newdt = .1*dt
    # these are the new x_times, a bit inefficient
    xnew_i = np.arange(ts_i[0],ts_i[-1],newdt)
    xnew_j = np.arange(ts_j[0],ts_j[-1],newdt)

#    pad = 5e3 #with 5 sec
#    xnew_i = np.arange(ts_i[0]-pad,ts_i[-1]+pad,newdt)
#    xnew_j = np.arange(ts_j[0]-pad,ts_j[-1]+pad,newdt)

    # now upsample!
    y_i = f_i(xnew_i)
    y_j = f_j(xnew_j)

    #TODO
    # sometimes the d435 sendd repeated frames:
    # due to a super stupid scipy interp1d bug: https://github.com/scipy/scipy/issues/4304
    # we have to clean up any nans



    # do padding of any potentially dropped frames by upsampling to
    #TODO no need for the full! Could just look a few secs left and right!
    # todo idea: use pycorrelate as a point process and save only the flip times
    # i.e. the times where the lead flips from down to up!

    if False:
        #TODO NB this uses numpy, but for some reason, max lags are not implemented yet
        # see https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy/47893831#47893831
        # and https://github.com/numpy/numpy/pull/5978
        corr = np.correlate(y_i,y_j,"full")
        # generate a vector of the shifts, not really nesc. but very readable
        corrshifts = np.arange(-(len(y_j)-1),len(y_i))*newdt
        # now find the shift which gives the highest correlation!
        #TODO make binary again? Fit theoretical convolutional model to get sub-sample resolution??
        shift = corrshifts[np.argmax(corr)]
        print('shift: '+str(shift)+', idx: '+str(np.argmax(corr)))

    elif True:
        from scipy.signal import correlate
        # see above!
        # MAYBE this is a bit excessive, but W/E!
        end_idx = int(5e4) # use only these indices!
        if np.max([len(y_i),len(y_j)]) > end_idx:
            corr = np.correlate(y_i[0:end_idx],y_j[0:end_idx],mode='full')
            corrshifts = np.arange(-(end_idx-1),end_idx)*newdt
            shift = corrshifts[np.argmax(corr)]
            print('shift: '+str(shift)+', idx: '+str(np.argmax(corr)))
        else:
            corr = np.correlate(y_i,y_j,"full")
            corrshifts = np.arange(-(len(y_j)-1),len(y_i))*newdt
            shift = corrshifts[np.argmax(corr)]
            print('shift: '+str(shift)+', idx: '+str(np.argmax(corr)))


    elif False:
        # use pycorrelate
        # it only does positive lags, so:
        import pycorrelate as pyc
        maxlag = 100
        c_after = pyc.ucorrelate(y_i,y_j, maxlag=100)
        c_before = pyc.ucorrelate(np.flip(y_i,0),np.flip(y_j,0), maxlag=100)

        if len(y_i)>len(y_j):
            padding = np.zeros(y_i.shape)

        elif len(y_i)<len(y_j):
            padding = np.zeros(y_j.shape)
            padding[0:len(y_i)] = y_i
            c_zero = sum(padding*y_j)

        else:
            c_zero = sum(y_i*y_j)

        c_zero = pyc.ucorrelate( y_i[0:-1] ,y_j, maxlag=1)

        corr = np.h



    if ShowPlots:
        plt.figure()
        # reload to get the non binary
#        f_counter_i,fn_i,ts_i,nix_i,leds_i = load_calib_stamps(which_device,top_folder)
#        f_counter_j,fn_j,ts_j,nix_j,leds_j = load_calib_stamps(ref_device,ref_folder)
        # and plot
        plt.plot(ts_i,leds_i)
        plt.plot(ts_j,leds_j)
        plt.ylabel('led signat [8 bit]')
        plt.xlabel('t [ms]')
        plt.title('not aligned, device ' + str(which_device)+', ref: dev'+str(ref_device))
        plt.legend(('dev'+str(which_device),'ref'))
        plt.savefig(top_folder+'/ledstamps_not_aligned'+str(which_device)+'.png')
        plt.show()


        plt.figure()
        plt.plot(ts_i-shift,leds_i)
        plt.plot(ts_j,leds_j)
        plt.ylabel('led signat [8 bit]')
        plt.xlabel('t [ms]')
        plt.title('aligned!, shift: '+str(shift)+ ', device ' + str(which_device)+', ref: dev'+str(ref_device))
        plt.legend(('dev'+str(which_device),'ref'))
        plt.savefig(top_folder+'/ledstamps_aligned'+str(which_device)+'.png')
        plt.show()


#        plt.figure()
#        plt.plot(y_i)
#        plt.plot(y_j)
#        plt.ylabel('led signat [8 bit]')
#        plt.xlabel('t [ms]')
#        plt.title('binary and upsampled, device ' + str(which_device)+', ref: dev'+str(ref_device))
#        plt.legend(('dev'+str(which_device),'ref'))
#        plt.savefig(top_folder+'/ledstamps_binary'+str(which_device)+'.png')
#        plt.show()


        plt.figure()
        plt.subplot(211)
        plt.plot(corrshifts,corr)
        plt.ylabel('xcorr')
        plt.xlabel('t [ms]')
        plt.title('xcorr, shift: '+str(shift)+', device ' + str(which_device)+', ref: dev'+str(ref_device))
        ax = plt.gca()
        ax.axhline(y=0,c='k')
        ax.axvline(x=shift,c='r')

        plt.subplot(212)
        plt.plot(corrshifts,corr)
        plt.ylabel('xcorr')
        plt.xlabel('t [ms]')
        plt.title('xcorr, shift: '+str(shift)+', device ' + str(which_device)+', ref: dev'+str(ref_device))
        plt.xlim([-2000,2000])
        ax = plt.gca()
        ax.axhline(y=0,c='k')
        ax.axvline(x=shift,c='r')

        plt.savefig(top_folder+'/ledstamps_xcorr'+str(which_device)+'.png')
        plt.show()


    # collect it all and remove the shift ts_j
    shiftedstamps = np.vstack((f_counter_i,fn_i,ts_i-shift,nix_i-shift,leds_i)).transpose()

    np.savetxt(top_folder+'/shiftedstamps_'+str(which_device)+'.csv',shiftedstamps,delimiter=',')

    return shift


#%% NOW run the calculation
# will become false if the noshow command is given
ShowPlots = args.noshow


if args.convert == 'both':


    from utils.reading_utils import most_recent_calibration_folders

    top_folder_0,top_folder_1 = most_recent_calibration_folders()

    check_folder_if_present(top_folder_0)
    check_folder_if_present(top_folder_1)


    align_time_stamps(0,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(1,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(2,top_folder_1,0,top_folder_0,ShowPlots)
    align_time_stamps(3,top_folder_1,0,top_folder_0,ShowPlots)



    from utils.reading_utils import most_recent_recording_folders,date_tag_to_recording_folders

    top_folder_0,top_folder_1 = most_recent_recording_folders()

    check_folder_if_present(top_folder_0)
    check_folder_if_present(top_folder_1)


    align_time_stamps(0,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(1,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(2,top_folder_1,0,top_folder_0,ShowPlots)
    align_time_stamps(3,top_folder_1,0,top_folder_0,ShowPlots)

elif args.convert == 'calib':

    from utils.reading_utils import most_recent_calibration_folders

    top_folder_0, top_folder_1 = most_recent_calibration_folders()

    check_folder_if_present(top_folder_0)
    check_folder_if_present(top_folder_1)

    align_time_stamps(0,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(1,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(2,top_folder_1,0,top_folder_0,ShowPlots)
    align_time_stamps(3,top_folder_1,0,top_folder_0,ShowPlots)

elif args.convert == 'scene':

    from utils.reading_utils import most_recent_recording_folders,date_tag_to_recording_folders

    if args.rtag == 'None':
        top_folder_0, top_folder_1 = most_recent_recording_folders()
    else:
        top_folder_0, top_folder_1 = date_tag_to_recording_folders(args.rtag)


    check_folder_if_present(top_folder_0)
    check_folder_if_present(top_folder_1)

    align_time_stamps(0,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(1,top_folder_0,0,top_folder_0,ShowPlots)
    align_time_stamps(2,top_folder_1,0,top_folder_0,ShowPlots)
    align_time_stamps(3,top_folder_1,0,top_folder_0,ShowPlots)
