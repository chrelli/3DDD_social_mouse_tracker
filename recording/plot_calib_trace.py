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

parser = argparse.ArgumentParser(description='Plots the 3d trace of the calibration',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--noshow", help="disables showing the plots, will just dump the png's",
                    action="store_false")

args = parser.parse_args()


#%% check if the folders exist

from utils.reading_utils import most_recent_calibration_folders

top_folder_0, top_folder_1 = most_recent_calibration_folders()

check_folder_if_present(top_folder_0)
check_folder_if_present(top_folder_1)


# debugging step, just for mac
if sys.platform == 'darwin':
    top_folder_0 = '/Volumes/Seagate Backup Plus Drive/example_data/calibration'
    top_folder_1 = '/Volumes/Seagate Backup Plus Drive/example_data/calibration'

    check_folder_if_present(top_folder_0)
    check_folder_if_present(top_folder_1)


#%% and now show the data!
def plot_clean_calibration(which_device,top_folder,ShowPlots):
    raw_data = np.genfromtxt(top_folder+'/central_point_'+str(which_device)+'.csv', delimiter=',')
    frame,x,y,z,r,r_ball = raw_data[:,0],raw_data[:,1],raw_data[:,2],raw_data[:,3],raw_data[:,4],raw_data[:,5]


    def clean_by_r_ball(frame,x,y,z,r,r_ball):
        index_vector = np.where((r>0.2)*(r < .65)*(r_ball>0.01)*(r_ball < 0.04))
        return frame[index_vector],x[index_vector],y[index_vector],z[index_vector],r[index_vector]


    # frame,x,y,z,r = clean_by_r(frame,x,y,z,r)
    frame,x,y,z,r = clean_by_r_ball(frame,x,y,z,r,r_ball)


    #print(x)
    #print(which_device)

    plt.title('Trajectory, dev'+str(which_device))
    plt.plot(frame,x)
    plt.plot(frame,y)
    plt.plot(frame,z)
#    plt.plot(r)
    plt.xlabel('frame number')
    plt.ylabel('word space [m]')
    plt.legend(['x','y','z'],loc=4)
    plt.savefig(top_folder+'/calibration_trajectory_'+str(which_device)+'.png')
    plt.ylim([-.25, 1.25])
    if ShowPlots:
        plt.show()



    plt.title('Trajectory, dev'+str(which_device))
#    plt.plot(x)
#    plt.plot(y)
    plt.plot(frame,r)
    #plt.plot(r_ball)
    plt.xlabel('frame number')
    plt.ylabel('word space [m]')
    plt.legend(['r','r_ball'],loc=4)
    plt.ylim([0, 1.25])

    plt.show()
    #plt.ylim([-.25, 1.25])


#%% run the plotting:
ShowPlots = args.noshow
if ShowPlots:
    plot_clean_calibration(0,top_folder_0,ShowPlots)
    plot_clean_calibration(1,top_folder_0,ShowPlots)
    plot_clean_calibration(2,top_folder_1,ShowPlots)
    plot_clean_calibration(3,top_folder_1,ShowPlots)
