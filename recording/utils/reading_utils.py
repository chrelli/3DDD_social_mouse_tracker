#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:44:50 2018

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

# for handling the h5py files
import h5py
import glob

#%%  plot the time stamps of the calibration recording, save as png

# FOR NOW: jyst take the most recent calibration folder!!
#TODO for future, make it possible to ask for a specific calibration

def date_tag_to_recording_folders(date_tag):
    top_folder_0 = '/media/chrelli/Data0'
    top_folder_1 = '/media/chrelli/Data1'
    folder_0 = glob.glob(top_folder_0+'/recording*'+date_tag+'*')
    folder_1 = glob.glob(top_folder_1+'/recording*'+date_tag+'*')
    assert len(folder_0) == 1 # make sure that the tag only fits one!
    return folder_0[0], folder_1[0]

def most_recent_calibration_folders():
    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:11] == 'calibration' for x in folder_list]

    led_list = list(compress(folder_list,logic_list))
    led_list.sort()
    # get the last one
    newest_folder = led_list[-1]
    # and make a folder
    constant_folder0 = '/media/chrelli/Data0/'+newest_folder
    constant_folder1 = '/media/chrelli/Data1/'+newest_folder
    return constant_folder0,constant_folder1

def most_recent_recording_folders():
    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:9] == 'recording' for x in folder_list]

    led_list = list(compress(folder_list,logic_list))
    led_list.sort()
    # get the last one
    newest_folder = led_list[-1]
    # and make a folder
    constant_folder0 = '/media/chrelli/Data0/'+newest_folder
    constant_folder1 = '/media/chrelli/Data1/'+newest_folder
    return constant_folder0,constant_folder1

def all_master_recording_folders():
    # RETURNS a list of recording folders sorted to newest first!
    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:9] == 'recording' for x in folder_list]

    led_list = list(compress(folder_list,logic_list))
    led_list.sort(reverse=True)
    # get the last one
    full_folders = ['/media/chrelli/Data0/'+i for i in led_list] 
    return full_folders


def load_arena_geometry(scene_folders,load_corners=True):
    # check if it exists, then load it
    if os.path.exists(scene_folders[0]+'/M0.csv'):
        # we also have to load the transformation matrices for the arena rotation
        M0 = np.genfromtxt(scene_folders[0]+'/M0.csv',delimiter = ',')
        floor_point = np.genfromtxt(scene_folders[0]+'/floor_point.csv',delimiter = ',')
        floor_normal = np.genfromtxt(scene_folders[0]+'/floor_normal.csv',delimiter = ',')
    else:
        #else if it doesn't exist, look back through recording folders to find the most recent
        print('DANGER - M0 not found, trying to load from older folders')
        prev_folders = all_master_recording_folders()
        for folder in prev_folders:
            if os.path.exists(folder+'/M0.csv'):
                M0 = np.genfromtxt(folder+'/M0.csv',delimiter = ',')
                floor_point = np.genfromtxt(folder+'/floor_point.csv',delimiter = ',')
                floor_normal = np.genfromtxt(folder+'/floor_normal.csv',delimiter = ',')
                # break the loop!
                print('found M0 in '+folder+'..')
                break

    if os.path.exists(scene_folders[0]+'/fitted_corners.csv') and load_corners:
        # AND we have to load the corners of the polygon! (in the rotated coordinate system)
#        refined_corners = np.genfromtxt(scene_folders[0]+'/refined_corners.csv',delimiter=',')
        fitted_corners = np.genfromtxt(scene_folders[0]+'/fitted_corners.csv',delimiter=',')

    elif load_corners:
        print('DANGER - M0 corners not found, trying to load from older folders')
        prev_folders = all_master_recording_folders()
        for folder in prev_folders:
            if os.path.exists(folder+'/fitted_corners.csv'):
                fitted_corners = np.genfromtxt(folder+'/fitted_corners.csv',delimiter=',')
                # break the loop!
                print('found fitted_corners in '+folder+'..')
                break
    else:
        fitted_corners = [None]

    return M0,floor_point,floor_normal,fitted_corners

#%% for h5py

def open_h5(file_path):
    hf = h5py.File(file_path, 'r')
    # get the keys in the file, and sort them!
#    what_keys = list(hf.keys())
    # this turns the keys into integers and loads them
    what_keys = list(map(int,hf.keys()))
    what_keys.sort()
    return hf,what_keys


#%% These are helping functions for the first look at a recording

def read_shifted_stamps(which_device,top_folder):
    this_name = top_folder+'/shiftedstamps_'+str(which_device)+'.csv'
    # from when the time stamps are saved
    # shiftedstamps = np.vstack((f_counter_i,fn_i,ts_i-shift,nix_i-shift,leds_i)).transpose()
    # we actually need the frames (indexed from 0) and the time stamps, shifted, i.e. col
    # had a mistake here once, had forgotten to update
    frames = np.genfromtxt(this_name, delimiter=',')[:,0]
    stamps = np.genfromtxt(this_name, delimiter=',')[:,2]
    n_dropped_frames = np.sum(np.diff(stamps) > 1.2*np.median(np.diff(stamps)))
    return frames, stamps, n_dropped_frames



#%%TODO make this parallelized or vectorized or SOMETHING...
#TODO move the number of dropped frames somewhere else
# for each stamp vector, find the closest time point!

def bisection(array,value):
    # these is a super neat bisection searching method, described here: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl

# TODO if there are no dropped frames, this step can be skipped!
def find_closest_time_and_frame(frames,stamps,reference_stamps):
    closest_frame = np.empty(np.shape(reference_stamps))
    closest_time = np.empty(np.shape(reference_stamps))

    n_frames = len(stamps)
    n_ref_frames = len(reference_stamps)

    for i in range(n_ref_frames):
        # get the target index:
        target_index = bisection(stamps,reference_stamps[i])
        # handle the cases where the target is outside the range (see bisection)
        target_index = np.clip(target_index,0,n_frames-1)

        #print(target_index)
        closest_frame[i] = frames[target_index]
        closest_time[i] = stamps[target_index]

    time_diff = closest_time - reference_stamps
    return closest_frame.astype(int),time_diff


#%% Lots of reading and filtering functions


def pixel_2_position(pi,pj,dij,cam_params):
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


    positions = np.vstack((x_m,y_m,z_m)).T
    return positions

def pixel_2_position_roi(pi,pj,dij,cam_params,roi):
    # takes the pi pj pd as vectors
    # the cam params are fx,fx,ppx,ppy,d_scale,fps_choice,frame_width,frame_height
    # to calculate in mm, multiply with the depth scale
    # WAIT this is not in mm at all - this is in meters!
    fx,fy,ppx,ppy,depth_scale,frame_width,frame_height = cam_params[0],cam_params[1],cam_params[2],cam_params[3],cam_params[4],cam_params[6],cam_params[7]
    z_m = dij*depth_scale

    # NOW we correct for the ROI by adding the shift back
    pi = pi + roi[1]
    pj = pj + roi[0]
    # WAIT we also have to add the roi back to the PPX and PPY

    # and now use pinhole cam function to get the x and y
    # remember the half is positive because of python!
#    x_m = (pj + .5 - ppx) * z_m / fx
#    y_m = (pi + .5 - ppy) * z_m / fy

    x_m = (pj - ppx) * z_m / fx
    y_m = (pi - ppy) * z_m / fy

#    plt.figure();plt.scatter(pj,pi,marker = '.');plt.plot(ppx,ppy,'o',c='r');plt.plot(frame_width/2,frame_height/2,'o',c='g')

    # and now use pinhole cam function to get the x and y
    # remember the half is positive because of python!
#    x_m = (pj + .5 - frame_width/2) * z_m / fx
#    y_m = (pi + .5 - frame_height/2) * z_m / fy

    positions = np.vstack((x_m,y_m,z_m)).T
    return positions

def depth_to_positions(which_device,top_folder,hf_d,keys_d,cam_params,frame):
    # UPDATED
    # load a depth frame, using the keys and the frame #
    # the function recieves a ref to the h5py file
    d = np.asanyarray(hf_d[str(keys_d[frame])])

    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]

    # now convert to positions
    positions = pixel_2_position(pi,pj,dij,cam_params)
    return positions

def depth_to_positions_roi(which_device,top_folder,hf_d,keys_d,cam_params,roi,frame):
    # UPDATED
    # load a depth frame, using the keys and the frame #
    # the function recieves a ref to the h5py file
    d = np.asanyarray(hf_d[str(keys_d[frame])])
    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]

    # now convert to positions
    positions = pixel_2_position_roi(pi,pj,dij,cam_params,roi)
    return positions

def cad_to_positions(which_device,top_folder,hf_d,keys_d,hf_cad,keys_cad,cam_params,frame):
    # load a depth frame
    d = np.asanyarray(hf_d[str(keys_d[frame])])
    cad = np.asanyarray(hf_cad[str(keys_cad[frame])])

    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]
    # and the color of each point as well
    cadij = cad[pi,pj,:]

    # now convert to positions
    positions = pixel_2_position(pi,pj,dij,cam_params)
    return positions, cadij

def cad_to_positions_roi(which_device,top_folder,hf_d,keys_d,hf_cad,keys_cad,cam_params,roi,frame):
    # load a depth frame
    d = np.asanyarray(hf_d[str(keys_d[frame])])
    cad = np.asanyarray(hf_cad[str(keys_cad[frame])])

    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]
    # and the color of each point as well
    cadij = cad[pi,pj,:]

    # now convert to positions
    positions = pixel_2_position_roi(pi,pj,dij,cam_params,roi)
    return positions, cadij

##################
# npy binaly file versions:
##################


def depth_to_positions_npy(which_device,top_folder,d_list,cam_params,frame):
    # UPDATED
    # load a depth frame, using the keys and the frame #
    # the function recieves a ref to the h5py file
    d = np.load(top_folder+'/'+d_list[frame])

    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]

    # now convert to positions
    positions = pixel_2_position(pi,pj,dij,cam_params)
    return positions

def depth_to_positions_roi_npy(which_device,top_folder,d_list,cam_params,roi,frame):
    # UPDATED
    # load a depth frame, using the keys and the frame #
    # the function recieves a ref to the h5py file
    d = np.load(top_folder+'/'+d_list[frame])
    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]

    # now convert to positions
    positions = pixel_2_position_roi(pi,pj,dij,cam_params,roi)
    return positions


def cad_to_positions_npy(which_device,top_folder,d_list,cad_list,cam_params,frame):
    # load a depth frame
    d = np.load(top_folder+'/'+d_list[frame])
    cad = np.load(top_folder+'/'+cad_list[frame])

    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]
    # and the color of each point as well
    cadij = cad[pi,pj,:]

    # now convert to positions
    positions = pixel_2_position(pi,pj,dij,cam_params)
    return positions, cadij

def cad_to_positions_roi_npy(which_device,top_folder,d_list,cad_list,cam_params,roi,frame):
    # load a depth frame
    d = np.load(top_folder+'/'+d_list[frame])
    cad = np.load(top_folder+'/'+cad_list[frame])

    # get the indices where the depth is not zero
    pi,pj = np.where(d>0)
    # get the depth of the masked pixels as a raveled list
    dij = d[pi,pj]
    # and the color of each point as well
    cadij = cad[pi,pj,:]

    # now convert to positions
    positions = pixel_2_position_roi(pi,pj,dij,cam_params,roi)
    return positions, cadij









def filter_and_downsample_cloud(cloud):
    # NOW comes post-processing steps where we use the voxelgrid to downsample
    # and to select the mean colors!
    # some stuff id commented out, if I want to change it later
    #TODO clean the code such that SOR filtering etc is not hard coded

    # calculate the kdtree structure
    kdtree = cloud.add_structure("kdtree")
    # denoising filter step
    filt = cloud.get_filter('SOR',and_apply=True,kdtree=kdtree,k=16,z_max=1)


        # NOW Do decimation by voxel grid!
        # set voxel size, uniform for now
    #voxel_size = 0.003
        # get the grid id
    grid_id = cloud.add_structure("voxelgrid",
                               sizes=[voxel_size, voxel_size,voxel_size],
                               bb_cuboid=False)

    # these are the points at the centroid of the voxelgrid
    centroid_points = cloud.get_sample("voxelgrid_centroids", voxelgrid=grid_id)
    near_points = cloud.get_sample("voxelgrid_nearest", voxelgrid=grid_id)


    # this is the voxelgrid object!
    voxelgrid=cloud.structures.get(grid_id)

    # now, we can query the voxel id of all the points in the cloud!
    voxel_ids_all = voxelgrid.query(cloud.xyz)
    voxel_ids_centr = voxelgrid.query(centroid_points.values)
    voxel_ids_near = voxelgrid.query(near_points.values)

    # AND now we can get the me
    fullcolors = cloud.points[['red','green','blue']]
    # add the voxelid to the colors
    fullcolors['voxid'] = voxelgrid.query(cloud.xyz)
    # calculate the mean color in each voxel
    meancolors = fullcolors.groupby("voxid").mean()

    # now generate new points with rgb values for the decimated cloud
    dpoints = pd.concat((centroid_points,meancolors),axis=1)
    # overwrite
    cloud = PyntCloud(dpoints)

    # denoise once more
    kdtree = cloud.add_structure("kdtree")
    filt = cloud.get_filter('SOR',and_apply=True,kdtree=kdtree,k=24,z_max=.7)
    return cloud


def clean_positions_and_cad_by_z(positions,cad):
    good_range = (positions[:,2] > .2 )*(positions[:,2] < .65)
    positions = positions[good_range,:]
    cad = cad[good_range,:]
    return positions, cad

# SIMPLE GRAYSCALE FILTERING!!
def add_gray_to_points(points):
    points[['gray']] = pd.DataFrame(.299*points.red + .587*points.green + .114*points.blue , index=points.index)
    return points

# SIMPLE GRAYSCALE CUTTING!!
def remove_white_points_gray(points,cutoff = 200):
    return points[ (.299*points.red + .587*points.green + .114*points.blue) < cutoff]

def remove_white_points_red(points,cutoff = 210):
    return points[points.red < cutoff]

def remove_white_points_rgb(points,cutoff = 230):
    return points[(points.red < cutoff)*(points.green < cutoff)*(points.blue < cutoff)]

#%% more helpers
# helper function to clean the positions by x coordinate
def clean_positions_by_z(positions):
    good_range = (positions[:,2] > .2 )*(positions[:,2] < .65)
    positions = positions[good_range,:]
    return positions

# helper function to clean the positions by x coordinate
def clean_cad_tuple_by_z(cad_tuple):
    positions = cad_tuple[0]
    good_range = (positions[:,2] > .2 )*(positions[:,2] < .65)
    positions = positions[good_range,:]
    return positions,cad_tuple[1][good_range,:]


#%%
def load_master_frame_table(scene_folders, allow_reload = True):
    """
    Function to generate the master frame table. It looks for number of dropped frames
    and uses the cam with the least N of dropped frames as a reference
    #TODO get rid of the dropped frames stuff, and do it dynamically, so that in case
    even the ref cam has dropped frames, some other cams might be fine at that time
    Also: Maaaaayyyyybeee it would make sense for interpolate between depth frames in the
    case of dropped frames?? In case it happens at an ioppertune time? hmmmmmm
    """
    # first see if the files already exist?
    ref_time_name = scene_folders[0]+'/reference_time_cam.csv'
    ref_stamps_name = scene_folders[0]+'/reference_stamps.csv'
    master_name = scene_folders[0]+'/master_frame_table.csv'

    if os.path.isfile(master_name) and allow_reload:
        print("reloading existing master table!")
        master_frame_table = np.genfromtxt(master_name,delimiter=',').astype(int)
        reference_time_cam = np.genfromtxt(ref_time_name,delimiter=',').astype(int)
        reference_stamps = np.genfromtxt(ref_stamps_name,delimiter=',').astype(int)

    else:

        # load all four devices and save the reference times
        #TODO get rid of these repeated lines!
        frames_0, stamps_0, n_dropped_0 = read_shifted_stamps(0,scene_folders[0])
        frames_1, stamps_1, n_dropped_1 = read_shifted_stamps(1,scene_folders[1])
        frames_2, stamps_2, n_dropped_2 = read_shifted_stamps(2,scene_folders[2])
        frames_3, stamps_3, n_dropped_3 = read_shifted_stamps(3,scene_folders[3])

        # look for the camera with the minimum number of dropped frames!
        reference_time_cam = np.argmin([n_dropped_0,n_dropped_1,n_dropped_2,n_dropped_3])

        #TODO fix this folder bullshit
        #if reference_time_cam in [0,1]:
        #    ref_folder = top_folder_0
        #elif reference_time_cam in [2,3]:
        #    ref_folder = top_folder_1
        # this is better
        ref_folder = scene_folders[reference_time_cam]

        _, reference_stamps, _ = read_shifted_stamps(reference_time_cam,ref_folder)

        np.savetxt(ref_time_name, [reference_time_cam], delimiter=',')
        np.savetxt(ref_stamps_name, reference_stamps, delimiter=',')

        # now look ober the frames and find the closest neigbor
        # KIND of inefficient, but only has to be done once, so....
        closest_frame_0,time_diff_0 = find_closest_time_and_frame(frames_0,stamps_0,reference_stamps)
        closest_frame_1,time_diff_1 = find_closest_time_and_frame(frames_1,stamps_1,reference_stamps)
        closest_frame_2,time_diff_2 = find_closest_time_and_frame(frames_2,stamps_2,reference_stamps)
        closest_frame_3,time_diff_3 = find_closest_time_and_frame(frames_3,stamps_3,reference_stamps)
        #
        master_frame_table = np.vstack((closest_frame_0,closest_frame_1,closest_frame_2,closest_frame_3)).T
        np.savetxt(master_name, master_frame_table, delimiter=',')

    return master_frame_table,reference_time_cam,reference_stamps
