#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:22:13 2018

@author: chrelli

SET up the namespace and methods for numpy reading from numpy binary files

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



#%% check for the most recent recording folder

from utils.reading_utils import most_recent_recording_folders

top_folder_0, top_folder_1 = most_recent_recording_folders()

check_folder_if_present(top_folder_0)
check_folder_if_present(top_folder_1)


#%% OK - now we are in business, simply loop over the clouds and show them!
# check if the roi was used or not!
use_roi = 1 == np.genfromtxt(top_folder_0+'/use_roi.csv',delimiter=',')

if use_roi:
    roi_list = [read_roi_file(i) for i in range(4)]

#%% NOW SET up for loading everything
# load the camera parameters for all cameras to a list
scene_folders = [top_folder_0,top_folder_0,top_folder_1,top_folder_1]
cam_param_list = [read_cam_params(i,scene_folders[i]) for i in range(4)]

# load the master frame table! so we can look up which frames belong with which       
master_frame_table, reference_time_cam, reference_stamps = load_master_frame_table(scene_folders)   

###################
# Block for reading from npy saving - open lists of the files to read from
###################
d_lists = [get_file_shortlist(i,scene_folders[i]+'/npy_raw','d') for i in range(4)]
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
   


#%% ALSO collect all the methods for loading here??



def load_raw_depth_frame(frame_index,cut=True,voxel_size = 0.003):
    """
    Will load a depth frame from the desired 
    """

    if use_roi:
        position_list = [depth_to_positions_roi_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],roi_list[i],master_frame_table[frame_index,i]) for i in range(4)]
            
    else:
        position_list = [depth_to_positions_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],master_frame_table[frame_index,i]) for i in range(4)]
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
        return positions,np.ones((positions.shape[0],1))


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


#%% Import tools for localizing the arena
#TODO make this is functional definition??
def load_aligned_depth_frame(frame_index,M0,floor_point,cut=True,voxel_size = 0.002):
    """
    Will load a depth frame from the desired 
    """
    
    if use_roi:
        position_list = [depth_to_positions_roi_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],roi_list[i],master_frame_table[frame_index,i]) for i in range(4)]
            
    else:
        position_list = [depth_to_positions_npy(i,scene_folders[i]+'/npy_raw',d_lists[i],cam_param_list[i],master_frame_table[frame_index,i]) for i in range(4)]

    # map over the list?? Or is list comprehension mor pythonic?
    position_list = [clean_positions_by_z(yo) for yo in position_list]
   
    # transform the positions, also by mapping?
    position_list = [ apply_rigid_transformation(position_list[i],R_matrices[i],t_vectors[i]) for i in range(4) ]
    
    # stack all the positions
    positions = np.concatenate( position_list, axis=0 )
#        points = pd.DataFrame(positions, columns=['x', 'y', 'z'])
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


#%% ALSO need functions to do the vectorized polygon stuff
        
def make_polygon(refined_corners,buff=0.005):
    # generate the polygon and return it!
    from shapely.geometry.polygon import Polygon
    polygon = Polygon([(refined_corners[ii,0],refined_corners[ii,1]) for ii in range(4) ])
    # cut 3mm, the buff has to be put as a negative value here! AND for some weido reason, 
    # returning with the buffer applied works, otherwise not o_O?
    return polygon.buffer(-buff)


import shapely.vectorized    
def cut_positions_by_polygon(positions,arena_polygon):
    inliers = shapely.vectorized.contains(arena_polygon,positions[:,0],positions[:,1])
    return positions[inliers,:],inliers


def cut_by_plane(positions,floor_point,floor_normal,floor_cut=0.01, above = True):  
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

        # calculate d, which is the dot product between a point on the plane and the normal
    plane_d = np.dot(floor_normal,hover_point)

    # the idea is to calc ax+by+cz+d where abc is the normal and xyz is the point being tested
    # now do the dot product as the logic to pflip on the sign (don't care about equal to)

    if above:
        above_logic = np.dot(positions,plane_coeffs[0:3]) - plane_d > 0
    else:
        above_logic = (np.dot(positions,plane_coeffs[0:3]) - plane_d < 0)
        
    return positions[above_logic,:],above_logic



