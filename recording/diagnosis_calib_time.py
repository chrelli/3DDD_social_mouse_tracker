#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:34:06 2018

@author: chrelli

#TODO:
 automatically check how many camera files were present

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

# for recording and connecting to the intel realsense library
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

parser = argparse.ArgumentParser(description='Plots the time stamps to diagnose obvious problems',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--noshow", help="disables showing the plots, will just dump the png's",
                    action="store_false")

args = parser.parse_args()


#%%  plot the time stamps of the calibration recording, save as png

# FOR NOW: jyst take the most recent calibration folder!!
#TODO for future, make it possible to ask for a specific calibration

#def most_recent_calibration_folders():
#    # simply look for the most recent LED folder!
#    top_folder = '/media/chrelli/Data0'
#    # get a list of the folders in that directory
#    folder_list = next(os.walk(top_folder))[1]
#    logic_list = [x[0:11] == 'calibration' for x in folder_list]
#    
#    led_list = list(compress(folder_list,logic_list)) 
#    led_list.sort()
#    # get the last one
#    newest_folder = led_list[-1]
#    # and make a folder        
#    constant_folder0 = '/media/chrelli/Data0/'+newest_folder #TODO fix this shit!
#    constant_folder1 = '/media/chrelli/Data1/'+newest_folder
#    return constant_folder0,constant_folder1

from utils.reading_utils import most_recent_calibration_folders

top_folder_0, top_folder_1 = most_recent_calibration_folders()

#top_folder_0 = '/media/chrelli/Data0/calibration'
check_folder_if_present(top_folder_0)
#top_folder_1 = '/media/chrelli/Data1/calibration'
check_folder_if_present(top_folder_1)


def plot_timestamps_0(which_device,top_folder,labelstring):
    print('starting camera 1!')
    # load the
    my_data = np.genfromtxt(top_folder+'/timestamps_'+str(which_device)+'.csv', delimiter=',')
#    plt.plot(my_data[:,1])
    plt.figure()
#    plt.hold(True)
    plt.plot(1000./60*np.ones(len(my_data)))
    plt.plot(1000./30*np.ones(len(my_data)))
    plt.plot(np.diff(my_data[:,3]*1000))
    plt.plot(np.diff(my_data[:,2]))

    plt.xlabel('# frame')
    plt.ylabel('dt [ms]')
    plt.title(labelstring+' stamps, device ' + str(which_device) + 'nframes = ' + str(len(my_data[:,0])))
    plt.legend(('60 fps','30 fps','unix','cam data'))
    plt.savefig(top_folder+'/'+labelstring+'_timestamps_'+str(which_device)+'.png')

    if args.noshow:
        plt.show()


plot_timestamps_0(0,top_folder_0,'calibration')
plot_timestamps_0(1,top_folder_0,'calibration')
plot_timestamps_0(2,top_folder_1,'calibration')
plot_timestamps_0(3,top_folder_1,'calibration')


#%% plot the LED stamps of the recordings in the same way


def plot_led_stamps_0(which_device,top_folder,labelstring):
    print('starting camera 1!')

    my_data = np.genfromtxt(top_folder+'/timestamps_'+str(which_device)+'.csv', delimiter=',')

    plt.figure()

    plt.plot(my_data[:,4])

    plt.xlabel('# frame')
    plt.ylabel('dt [ms]')
    plt.title(labelstring+' time stamps, device ' + str(which_device))
    plt.legend(('auto LED trace',))
    plt.savefig(top_folder+'/'+labelstring +'_ledstamps_'+str(which_device)+'.png')

    if args.noshow:
        plt.show()


plot_led_stamps_0(0,top_folder_0,'calibration')
plot_led_stamps_0(1,top_folder_0,'calibration')
plot_led_stamps_0(2,top_folder_1,'calibration')
plot_led_stamps_0(3,top_folder_1,'calibration')



##%% Also plot the recording time stamps, just by changing the top folder!
#
#
#plot_timestamps_0(0,top_folder_0,'recording')
#plot_timestamps_0(1,top_folder_0,'recording')
#plot_timestamps_0(2,top_folder_1,'recording')
#plot_timestamps_0(3,top_folder_1,'recording')
#
#
#plot_led_stamps_0(0,top_folder_0,'recording')
#plot_led_stamps_0(1,top_folder_0,'recording')
#plot_led_stamps_0(2,top_folder_1,'recording')
#plot_led_stamps_0(3,top_folder_1,'recording')
