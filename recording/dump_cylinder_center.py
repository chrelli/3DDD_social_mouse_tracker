#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pwd
"""
Created on Wed May  2 14:42:30 2018

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

# for cloud handling
from pyntcloud import PyntCloud

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *
from utils.cloud_utils import *
from utils.reading_utils import *
from utils.localization_utils import *


import h5py


#%%

import argparse

parser = argparse.ArgumentParser(description='Finds the location of the cylinder in the depth data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--rtag", type=str, default = 'None', help="glob for a date tag in the recording folders")

args = parser.parse_args()


#%% check for the most recent recording folder

from utils.reading_utils import most_recent_recording_folders,date_tag_to_recording_folders

rtag = args.rtag

if rtag == 'None':
    top_folder_0, top_folder_1 = most_recent_recording_folders()
else:
    top_folder_0, top_folder_1 = date_tag_to_recording_folders(rtag)

scene_folders = [top_folder_0,top_folder_0,top_folder_1,top_folder_1]

check_folder_if_present(top_folder_0)
check_folder_if_present(top_folder_1)



# load the master frame table to display!
master_frame_table, reference_time_cam, reference_stamps = load_master_frame_table(scene_folders)

n_frames = master_frame_table.shape[0]
median_frame = int(np.round(master_frame_table.shape[0]/2))


argsframe = median_frame
print('selected frame is '+str(argsframe)+'!')
#frame_index = args.frame


# check if the roi was used or not!
use_roi = 1 == np.genfromtxt(top_folder_0+'/use_roi.csv',delimiter=',')
use_roi = False  #HACKE

if use_roi:
    roi_list = [read_roi_file(i) for i in range(4)]


#%% NOW SET up for loading everything
# load the camera parameters for all cameras to a list
scene_folders = [top_folder_0,top_folder_0,top_folder_1,top_folder_1]
cam_param_list = [read_cam_params(i,scene_folders[i]) for i in range(4)]

# load the master frame table! so we can look up which frames belong with which
master_frame_table, reference_time_cam, reference_stamps = load_master_frame_table(scene_folders)
#

###################
# Block for reading from npy saving
###################
d_lists = [get_file_shortlist(i,scene_folders[i]+'/npy_raw','d') for i in range(4)]
if False:# args.show == 'cad':
    cad_lists = [get_file_shortlist(i,scene_folders[i]+'/npy_raw','cad') for i in range(4)]

# also load the transformational matrices
from utils.reading_utils import most_recent_calibration_folders
calib_folder_0, calib_folder_1 = most_recent_calibration_folders()
# make a list, a bit silly
calib_folders = [calib_folder_0,calib_folder_0,calib_folder_1,calib_folder_1]
# get a list of the transformations
transformation_list = [load_transformation(i,calib_folders[i]) for i in range(4)]
# tupe will have the content: R_0,t_0,_ =
R_matrices = [transformation_list[i][0] for i in range(4)]
t_vectors = [transformation_list[i][1] for i in range(4)]

# run the function to get the geometry data!
M0,floor_point,floor_normal,refined_corners = load_arena_geometry(scene_folders,load_corners=False)

# print("corners loaded:")
# print(refined_corners)
# print("look good?")

def pixel_2_position_decimated(pi,pj,dij,cam_params):
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

    x_m = (2*pj - ppx) * z_m / fx
    y_m = (2*pi - ppy) * z_m / fy


    positions = np.vstack((x_m,y_m,z_m)).T
    return positions


# top_folder = top_folder_0+'/npy_raw'
# d_list, cam_params = d_lists[0],cam_param_list[0]
# frame = 1000

def depth_to_positions_npy_decimated(which_device,top_folder,d_list,cam_params,frame):
    # UPDATED
    # load a depth frame, using the keys and the frame #
    # the function recieves a ref to the h5py file
    # if it's a pickeld file:
    # d = np.load(top_folder+'/'+d_list[frame])
    # or if it's a png file
    # The -1 flag forces cv2 to read as 16 bit
    d = cv2.imread(top_folder+'/'+d_list[frame], -1)
    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]

    # now convert to positions
    positions = pixel_2_position_decimated(pi,pj,dij,cam_params)
    return positions


#%% FIRST we load afew frames to determine a good cutoff for the floor

def load_aligned_depth_frame(frame_index,cut=True,voxel_size = 0.002):
    """
    Will load a depth frame from the desired
    """
    position_list = [depth_to_positions_npy_decimated(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],master_frame_table[frame_index,i]) for i in range(4)]

    # map over the list?? Or is list comprehension mor pythonic?
    position_list = [clean_positions_by_z(yo) for yo in position_list]

    # transform the positions to a common coordinate system, also by mapping?
    # TODO get this upstream!
    position_list = [ apply_rigid_transformation(position_list[i],R_matrices[i],t_vectors[i]) for i in range(4) ]

    # stack all the positions
    positions = np.concatenate( position_list, axis=0 )
    # ALSO apply the arena transformations!
    positions = positions - floor_point
    # rotate!
    #TODO desperate need to convert everything to 4D transformations!! Here translation is first, then rotate. Above it's the other way around Yikes!!
    positions = np.transpose(np.matmul(M0,positions.T))


    if cut:
        round_points = np.round(positions/voxel_size)
        cut_positions,weights = np.unique(round_points, return_counts=True, axis=0)
        return cut_positions*voxel_size,weights
    else:
        return positions,np.ones((positions.shape[0],1))




#%% NOW fit a gaussian to get a good
# plotting was inspired by  reuse of https://stackoverflow.com/questions/19206332/gaussian-fit-for-python

# first chec, if the floor alerady exists!

from scipy.optimize import curve_fit

# get a middle frame and make a histogram (a low cut is fine here)
frame_index = argsframe
#frame_index = round(master_frame_table.shape[0]/2)
positions,weights = load_aligned_depth_frame(frame_index,cut=False,voxel_size=0.001)



def cut_by_floor_roof(positions,floor_point,floor_normal,floor_cut=0.01,roof_cut=0.01):
    """
    Function to cut away the floor w/o a need to rotate the points fikst, just use the dot product trick
    # cut away floor?
    # use the equation of the plane: http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfPlanes.aspx
    # and evaluate this to check if it's above or below: https://stackoverflow.com/questions/15688232/check-which-side-of-a-plane-points-are-on

    """
    # find the first coefficients of the equation of the plane!
    plane_coeffs = floor_normal

        # find a point above the plane!
    hover_point = floor_point + floor_normal * floor_cut
    roof_point = floor_point + floor_normal * roof_cut
        # calculate d, which is the dot product between a point on the plane and the normal
    floor_d = np.dot(floor_normal,hover_point)
    roof_d = np.dot(floor_normal,roof_point)

    # the idea is to calc ax+by+cz+d where abc is the normal and xyz is the point being tested
    # now do the dot product as the logic to pflip on the sign (don't care about equal to)
    #test_prod = np.dot(positions,plane_coeffs[0:3])
    # einsum is faster!
    test_prod = np.einsum('j,ij->i',plane_coeffs,positions)


    above_logic = (test_prod > floor_d) * (test_prod < roof_d)

    return positions[above_logic,:],above_logic



#%% Define two functions, one which will load just a raw depth frame, and one which will load both depth and colors

def load_raw_depth_frame(frame_index,cut=True,voxel_size = 0.003):
    """
    Will load a depth frame from the desired
    """

    if use_roi:
        position_list = [depth_to_positions_roi_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],roi_list[i],master_frame_table[frame_index,i]) for i in range(4)]

    else:
        position_list = [depth_to_positions_npy_decimated(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],master_frame_table[frame_index,i]) for i in range(4)]
    # map over the list?? Or is list comprehension mor pythonic?
    position_list = [clean_positions_by_z(yo) for yo in position_list]

    # transform the positions to a common coordinate system, also by mapping?
    # TODO get this upstream!
    position_list = [ apply_rigid_transformation(position_list[i],R_matrices[i],t_vectors[i]) for i in range(4) ]

    # stack all the positions
    positions = np.concatenate( position_list, axis=0 )
    if cut:
        round_points = np.round(positions/voxel_size)
        cut_positions,weights = np.unique(round_points, return_counts=True, axis=0)
        return cut_positions*voxel_size,weights
    else:
        return positions,np.ones((positions.shape[0]))


def load_raw_depth_frame_color(frame_index,cut=True,voxel_size = 0.003):
    """
    Will load a depth frame from the desired WITH COLOR
    """
    if use_roi:
        cad_tuple_list = [cad_to_positions_roi_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cad_lists[i],cam_param_list[i],roi_list[i],master_frame_table[frame_index,i]) for i in range(4)]

    else:
        cad_tuple_list = [cad_to_positions_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cad_lists[i],cam_param_list[i],master_frame_table[frame_index,i]) for i in range(4)]
    # clean by z
    cad_tuple_list = [clean_cad_tuple_by_z(that_tuple) for that_tuple in cad_tuple_list]
    # stack all the positions, and all the colors as well
    positions = np.concatenate( [ apply_rigid_transformation(cad_tuple_list[i][0],R_matrices[i],t_vectors[i]) for i in range(4) ], axis =0)
    colors = np.concatenate( [yo[1] for yo in cad_tuple_list], axis=0 )

    if cut:
        round_points = np.round(positions/voxel_size)
        cut_positions,cut_logic,weights = np.unique(round_points,return_index = True, return_counts=True, axis=0)
        colors = colors[cut_logic]
        return cut_positions*voxel_size,weights,colors
    else:
        return positions,np.ones((positions.shape[0],1)),colors


easy3d(positions[::10,:])
## Fit the data to the cylinder!
# first, load the positions
positions,weights = load_raw_depth_frame(frame_index,cut=False)


# automatically determine a good z range to find the cylinder!

plt.figure()
edges = np.arange(0,0.5,.005)
roughly_center_z = positions[:,2][ np.sum(positions[:,:2]**2,axis=-1)**(1./2) < .20 ]
roughly_center_z = positions[:,2]

count, edges = np.histogram(roughly_center_z,edges)
edges = edges[:-1]
from scipy.stats import zscore
count = zscore(count)

thresh_cut = edges[(count > 0) * edges > .10][0]
thresh_lo = thresh_cut - 0.03
thresh_hi = thresh_cut + 0.07

plt.axvline(thresh_cut,c='r',label='thresh')
plt.axvline(thresh_lo,c='g',label='low cut')
plt.axvline(thresh_hi,c='g',label='high cut')
plt.plot(edges,count,'o-k')
plt.title('histogram of z')
plt.xlabel('z [m]')
plt.ylabel('z(count) ')
plt.legend()
plt.show()


# do a fast cut by the floor, 1m roof for now
positions,above_logic = cut_by_floor_roof(positions,floor_point,floor_normal,floor_cut=thresh_lo,roof_cut=thresh_hi)

# now,we need to align in order to use the polygon cut
positions = positions - floor_point
# rotate!
#TODO desperate need to convert everything to 4D transformations!! Here translation is first, then rotate. Above it's the other way around Yikes!!
positions = np.transpose(np.matmul(M0,positions.T))

#%% ask for a guess click
if False:
    plt.close('all')
    plt.figure()
    plt.scatter(positions[:,0],positions[:,1],marker=".")
    plt.gca().set_aspect('equal')
    plt.title("click center")
    plt.show()
    center_click = plt.ginput(1)
    plt.close()
else:
    center_click = [(.022,0.0062)]

# do a loop over many frames and fit each one!

#%%
r_cylinder = 0.245/2 #m
r_cylinder = 0.29/2 #m it's really 30 though

x0 = center_click[0]
def distance_to_cylinder(x0,positions0,positions1):
    dd= np.abs(r_cylinder - np.sqrt( (positions0 - x0[0])**2 + (positions1 - x0[1])**2 ) )
    return np.clip(dd,0,0.05)

from scipy.optimize import least_squares


bounds = ([-np.inf,-np.inf],[np.inf,np.inf])
# made robust by clipping the distance contribution to the loss
res_robust = least_squares(distance_to_cylinder, x0,
                            args=(positions[:,0],positions[:,1]),
                            bounds=bounds )

x_fit = res_robust.x

# plot the fitted data!

plt.figure()
plt.scatter(positions[:,0],positions[:,1],marker=".")

circle1 = plt.Circle((x_fit[0], x_fit[1]), r_cylinder, color='r', fill = False)
ax = plt.gca()
ax.add_artist(circle1)

plt.axes().set_aspect('equal', 'datalim')
plt.title("fitted corners by weighted fit to histogram")

plt.show()

np.save(scene_folders[0]+'/c_cylinder.npy',x_fit)
np.save(scene_folders[0]+'/r_cylinder.npy',r_cylinder)
