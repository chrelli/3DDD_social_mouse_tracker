#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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



#%% arguments

import argparse

parser = argparse.ArgumentParser(description='UPDATE this:will dump the matrices to disk Filteres and merges the recorded data and saves to an hdf5 file.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument("--onlydepth", help="disables the aligned color and just saves the",action="store_true")


parser.add_argument('--voxel', type=float, default = 0.002 , # choices=[1,2,3,4],
                    help='voxel size of the dirty numpy downsampling')

#TODO add the cutting as an input variable!

parser.add_argument('--nframes', type=int, default = 60,
                    help='number frames to use for normal estimation, default 60')

parser.add_argument("--noplots", help="disables the plotting",
                    action="store_false")




args = parser.parse_args()

#TODO fix this
show_plots = args.noplots



#%% check for the most recent recording folder

# import everything from the namespace, and functions as well
from utils.open_npy_namespace_calib import *



#%% OK, now it's ready to start looking for the arena!

# default is 100 steps, might be a bit excessive
#TODO idea: keep track of convergence and only fit as many frames as nesc for a stable estimate of the normal/ppoint
n_steps = args.nframes#len(what_keys)
# take the steps equally spaced
# what's the highest frame that everyone has?
# kill the first and last 10 pct
min_frame = np.round(.1*master_frame_table.shape[0])
max_frame = np.round(.9*master_frame_table.shape[0])
test_frames = np.linspace(min_frame,max_frame,num=n_steps,dtype=int)


# make numpy arrays for holding the fitted normals
normal_holder = np.empty((n_steps,3))
point_holder = np.empty((n_steps,3))
theta_holder = np.empty((n_steps,1))

if show_plots:
    # setu up the plot to plot to
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


for this_step,this_frame in enumerate(test_frames):
    if this_step % 3==0:
        print("RANSACing frame " +str(this_step)+" of "+str(n_steps)+"..." )
    # load the positions
    positions,weights = load_raw_depth_frame(this_frame,cut=True,voxel_size = args.voxel)

    # fit the model
#    this_normal, this_point = fit_plane_to_positions_pyntcloud(positions)
#    this_normal, this_point = fit_plane_to_positions_sklearn(positions)
    # fit the weighted model:
    this_normal, this_point = fit_plane_to_positions_sklearn_weighted(positions,weights)
    # flip the normal, if the angle with z is too large
    theta = rotation_angle(this_normal,np.array([0,0,1]))
    if np.abs(theta) <= np.pi/2:
        this_normal = -this_normal

    # save to holders
    normal_holder[this_step,:] = this_normal
    point_holder[this_step,:] = this_point
    theta_holder[this_step,:] = theta

    # and plot
    if show_plots:
        #print('angle: '+str(theta))
        vec_points = np.vstack((this_point, this_point+this_normal)).T
        ax.plot(vec_points[0,:],vec_points[1,:],vec_points[2,:],c='red',marker='o')

if show_plots:

    plt.show()



#%% caluclate the master normal of the floor

floor_normal = np.median(normal_holder,axis=0)
floor_point = np.median(point_holder,axis=0)

if show_plots:
    plt.figure()
    plt.subplot(121)
    plt.plot(point_holder)
    plt.title('estimated plane point')
    for value in floor_point:
        plt.axhline(y=value,c='peru')
    plt.xlabel('frame #')
    plt.ylabel('space [m]')
    plt.subplot(122)
    plt.plot(normal_holder)
    for value in floor_normal:
        plt.axhline(y=value,c='peru')

    plt.title('estimated plane normal')
    plt.xlabel('frame #')

    plt.show()



#%% now, calculate the required things for the rotation

# we want to have the floor pointing up at the end, so desired normal is z
desired_normal = np.array([0, 0, 1],dtype=float)

# cross them to get the rotation axis
rotation_vector = np.cross(desired_normal,floor_normal)

# so now we know the axis to rotate around, we calculate the the angle and rotation matrix
# the rotation angle and rotation matrix functions are super handy
theta = rotation_angle(this_normal,desired_normal)
M0 = rotation_matrix(rotation_vector,-theta)

# Now dump the matrices to disk
np.savetxt(top_folder_0+'/M0.csv',M0,delimiter=',')
np.savetxt(top_folder_0+'/floor_point.csv',floor_point,delimiter=',')
np.savetxt(top_folder_0+'/floor_normal.csv',floor_normal,delimiter=',')

print("dumping..")
print(top_folder_0)
print("floor point: " + str(floor_point))
print("floor normal: " + str(floor_normal))
print("all done :)")




#%% Plot the first frame with the normals

if show_plots:
    # make a cloud from the first frame

    points = pd.DataFrame(positions, columns=['x', 'y', 'z'])
    cloud = PyntCloud(points[::10])

    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # get points
    X = cloud.points.x.values
    Y = cloud.points.y.values
    Z = cloud.points.z.values

    # clean up!
    selecta = (Z>0.2)*(Z<65)
    selecta = ~np.isnan(Z)

    if 'red' in cloud.points:
        # get the colors
        C = cloud.points[['red', 'blue','green']].values
        ax.scatter(X[selecta], Y[selecta], Z[selecta], zdir='z', s=1, c=C[selecta,:]/255.0,rasterized=True)
    else:
        ax.scatter(X[selecta], Y[selecta], Z[selecta], zdir='z', s=1, c='b',rasterized=True)

    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)


    vec_points = np.vstack((floor_point, floor_point+normal_holder[-1,:]/4)).T
    ax.plot(vec_points[0,:],vec_points[1,:],vec_points[2,:],c='r',marker='o')

#
    vec_points = np.vstack((floor_point, floor_point+desired_normal/4)).T
    ax.plot(vec_points[0,:],vec_points[1,:],vec_points[2,:],c='lightgreen',marker='o')

    vec_points = np.vstack((floor_point, floor_point+rotation_vector/4)).T
    ax.plot(vec_points[0,:],vec_points[1,:],vec_points[2,:],c='orange',marker='o')
#

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    plt.legend(('floor_normal','desired_normal','rotation_vector'))

    ax.view_init(elev=0, azim=100)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.show()


#%% plot a cloud and a rotated version!
#%% function which applies the rotation matrix M0 and the
def rotate_cloud(cloud):
    # center
    positions = cloud.xyz - floor_point
    # rotate!
    positions = np.transpose(np.matmul(M0,positions.T))
    # replace back into cloud
    cloud.points.x = positions[:,0]
    cloud.points.y = positions[:,1]
    cloud.points.z = positions[:,2]
    return cloud

if show_plots:
    # show the rotated cloud!
    plot_cloud_eq(cloud)
    plot_cloud_eq(rotate_cloud(cloud))


#%% look for the size of the thing
