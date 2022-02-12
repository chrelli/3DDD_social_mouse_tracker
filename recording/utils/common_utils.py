#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:54:04 2018

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
import multiprocessing
from multiprocessing import Process


#%% Small handy functions for folders
def print_c_cores():
    number_of_cpus = multiprocessing.cpu_count()
    print('This machine has '+str(number_of_cpus)+' available cpu cores.')

def check_folder_if_present(this_path):
    if os.path.isdir(this_path):
        print(this_path+' was detected!')
    else:
        print('ERROR: ' +this_path+' was not detected!')
        sys.exit(1)

def reset_folder_if_present(this_path):
    if os.path.isdir(this_path):
        shutil.rmtree(this_path)
        time.sleep(0.1)
        print(this_path+' was deleted!')
    os.mkdir(this_path)

#%% firmata stuff
# get the serial port of the arduino for firmata
def get_serial_port():
    ser_devs = [dev for dev in os.listdir('/dev') if dev.startswith('ttyAC')]
    if len(ser_devs) > 0:
        return '/dev/'+ser_devs[0]
    return None

#%% plotting tools

def bare_plot3(a,b,c,mark="o",col="r"):
    # very simple plot3 version
    from matplotlib import pyplot
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    pylab.ion()
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(a, b, c,marker=mark,color=col)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

#%% Functions for loading cad images and parameters

#TODO resolve is there is any conflict here!!
def read_hsv_file(which_device,top_folder): # kill this one??
    # MAKE A FULLFILE
    # reads the hsv and gray values after filering
    this_name = top_folder+'/hsv_values_'+str(which_device)+'.csv'
    if os.path.exists(this_name):
        raw_values = np.genfromtxt(this_name, delimiter=',',dtype='int' )
        print(raw_values)
        hsv_values = raw_values[0:6]
        gray_values = raw_values[6:8]
    else:
        print('ERROR: '+ this_name+' not found!')
        sys.exit(0)

    return hsv_values,gray_values


def check_for_hsv_file(which_device,top_folder):
    # these are the default values
    hsv_values = np.array([0,179,0,255,0,255])
    gray_values = np.array([1,1])
    # these are the some guess values, good starting point
    hsv_values = np.array([0,1,0,255,117,255])
    gray_values = np.array([200,11])

    this_name = top_folder+'/hsv_values_'+str(which_device)+'.csv'
    if os.path.exists(this_name):
        raw_values = np.genfromtxt(this_name, delimiter=',',dtype='int' )
        print(raw_values)
        if raw_values.size > 0:
            # only get from text if it is not empty by mistake
            hsv_values = raw_values[0:6]
            gray_values = raw_values[6:8]

    return hsv_values, gray_values









def read_cam_params(which_device,top_folder):
    # reads the camera parameters of that camera
    this_name = top_folder+'/parameters_'+str(which_device)+'.csv'
    if os.path.exists(this_name):
        raw_list = np.genfromtxt(this_name, delimiter=',')[1,:]
        cam_params = raw_list
        fps_choice,frame_width,frame_height = raw_list[5],raw_list[6],raw_list[7]
    else:
        print('ERROR: '+ this_name+' not found!')
        sys.exit(0)

    return cam_params

def get_file_shortlist(which_device,top_folder,image_type):
    # TODO check if the padding digits have overflown!!!! if yes, do proper sorting by number!
    # list of files in the folder, specific to images!
    file_list = os.listdir(top_folder)
    # sort the list
    file_list.sort()
    file_logic = np.empty(len(file_list))
    for num,name in enumerate(file_list):
        file_logic[num]=name.startswith('dev'+str(which_device)+'_'+image_type+'_')
    short_list = list(compress(file_list,file_logic))
    return short_list


def load_data(which_device,top_folder):
    raw_data = np.genfromtxt(top_folder+'/central_point_'+str(which_device)+'.csv', delimiter=',')
    frame,x,y,z,r = raw_data[:,0],raw_data[:,1],raw_data[:,2],raw_data[:,3],raw_data[:,4]

    frame,x,y,z,r = clean_by_r(frame,x,y,z,r)
    x,y,z = add_ping_pong_radius(x,y,z,r)
    return x,y,z


#%% Functions to do filtering of image masks

def mask_stepper(c,hsv_values,gray_values,fgmask):
    # takes the a cad in BGR as an input and returns the mask after filtering
    HSVLOW = hsv_values[[0,2,4]]
    HSVHIGH = hsv_values[[1,3,5]]
    gcut, blur_amount = gray_values[0],gray_values[1]
    if blur_amount % 2 == 0: # make sure it's odd
        blur_amount += 1

    res = cv2.bitwise_and(c, c, mask=fgmask)
    #convert to HSV from BGR
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    #apply the range on a mask
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    # and blur
    mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
    # for some reason the gcut has to be a float, bot an int - cv2 bug?
    mask = cv2.inRange(mask,gcut.astype('float64'), 255)

    return mask

# get the largest region in the image, and fill it!
def fill_largest_region(image_input):
    # Find the largest contour and fill it
    im, contours, hierarchy = cv2.findContours(image_input,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )
    maxContour = 0
    maxContourData = 0
    # loop over the contours and get the size, and the max
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    # Create a mask from the largest contour
    mask = np.zeros_like(image_input)
    cv2.fillPoly(mask,[maxContourData],1)
    return mask


#%% small fucntions to do gymnastics with the point clouds

def clean_by_pd(pi,pj,pd):
    # function which cleans the data of instances where the depth is zero
    clean_index = np.where(pd > 0)
    pi,pj,pd = pi[clean_index],pj[clean_index],pd[clean_index]
    return pi,pj,pd

                # conver the pi,pj,pd = pixel_i,pixel_j,pixel_depth to xyz
def pixel_2_world(pi,pj,dij,cam_params):
    # takes the pi pj pd as vectors
    # the cam params are fx,fx,ppx,ppy,d_scale,fps_choice,frame_width,frame_height
    # to calculate in mm, multiply with the depth scale
    # WAIT this is not in mm at all - this is in meters!
    fx,fy,ppx,ppy,depth_scale,frame_width,frame_height = cam_params[0],cam_params[1],cam_params[2],cam_params[3],cam_params[4],cam_params[6],cam_params[7]
    
    z_m = dij*depth_scale

    # and now use pinhole cam function to get the x and y
    # remember the half is positive because of python!
#    x_m = (pj + .5 - ppx) * z_m / fx
#    y_m = (pi + .5 - ppy) * z_m / fy
    
    x_m = (pj - ppx) * z_m / fx
    y_m = (pi - ppy) * z_m / fy
    return x_m,y_m,z_m

def world_2_range(x_m,y_m,z_m):
    # calculates the range from the x,y,z values
    r_m = np.linalg.norm([x_m,y_m,z_m],axis=0)
    return r_m



#%%some helper functions for handling calibration traces

def clean_by_r(frame,x,y,z,r):
    index_vector = np.where((r>0.5)*(r < 1.5))
    return frame[index_vector],x[index_vector],y[index_vector],z[index_vector],r[index_vector]

def add_ping_pong_radius(x,y,z,r):
    radius = 0.02 # m
    points = np.vstack((x,y,z))
    # rescale all the poins where r>0
    points[:,r>0] = points[:,r>0]*( (1+radius/r[r>0]) )

    x,y,z = points[0,:],points[1,:],points[2,:]
    return x,y,z


def load_central_point(which_device,top_folder):
    raw_data = np.genfromtxt(top_folder+'/central_point_'+str(which_device)+'.csv', delimiter=',')
    frame,x,y,z,r = raw_data[:,0],raw_data[:,1],raw_data[:,2],raw_data[:,3],raw_data[:,4]

    frame,x,y,z,r = clean_by_r(frame,x,y,z,r)
    x,y,z = add_ping_pong_radius(x,y,z,r)
    return x,y,z
