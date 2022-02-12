#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:30:16 2018

@author: chrelli
"""


import time, os, sys, shutil

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


import math

# small utilities
import csv
from colour import Color
from itertools import compress # for list selection with logical
from tqdm import tqdm

# for image manipulation
import cv2

# for recording and connecting to the intel realsense librar
import pyrealsense as pyrs

#import multiprocessing
from multiprocessing import Process

# for cloud handling
#from pyntcloud import PyntCloud

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *
from utils.cloud_utils import *
#from fitting_utils import *

#from merge_and_filter_clouds import filter_and_downsample_cloud

# h5py for acessing data
import h5py

# ALLSO JIT STUFF

from numba import jit, njit


#%% read from processed frames

from utils.reading_utils import most_recent_recording_folders,read_shifted_stamps

# processed frames are kept here:
top_folder, _ = most_recent_recording_folders()

# the reference frames 
frames,ts,n_dropped = read_shifted_stamps(0,top_folder)

def read_processed_frame(top_folder,frame,voxel = 0.003, n_padding_digits = 8):
    raw = np.load(top_folder+'/npy/frame_'+str(frame).rjust(n_padding_digits,'0')+'.npy')
    #todo make column order and split
    positions = raw[:,0:3]*voxel
    weights = raw[:,3]
    return positions,weights
    
#%% 
positions,weights = read_processed_frame(top_folder,15000)
color3d(positions,weights)

#%% Ask for clicks!

from utils.fitting_utils import click_one_mouse

#todo is it called head or nose, make up my mind!
hip_click,mid_click,nose_click = click_one_mouse(positions)


#%% Now, we make a function, which spits out the constants

@njit
def mouse_body_size_constants(body_scale = 1,use_old=False):

    ## HIP is a prolate ellipsoid, centered along the x axis
    a_hip_min = 0.04/2 #m
    a_hip_max = 0.065/2 #m
    b_hip_min = 0.03/2 #m
    b_hip_max = 0.046/2 #m
    d_hip = 0.019 #m

    # converting it to the new terminology
    a_hip_0     = a_hip_min #m
    a_hip_delta = a_hip_max - a_hip_min #m
    b_hip_0     = b_hip_min #m
    b_hip_delta = b_hip_max - b_hip_min #m
    
    ## NOSE is prolate ellipsoid, also along the head direction vector
    # here, there is no re-scaling
    a_nose = 0.040/2 #m
    b_nose = 0.025/2 #m
    d_nose = 0.016 #m
    
    if use_old:
        ## HIP is a prolate ellipsoid, centered along the x axis
        a_hip_min = 0.018 #m
        a_hip_max = 0.025 #m
        b_hip_min = 0.011 #m
        b_hip_max = 0.02 #m
        d_hip = 0.015 #m
    
        # converting it to the new terminology
        a_hip_0     = a_hip_min #m
        a_hip_delta = a_hip_max - a_hip_min #m
        b_hip_0     = b_hip_min #m
        b_hip_delta = b_hip_max - b_hip_min #m
        
        ## NOSE is prolate ellipsoid, also along the head direction vector
        # here, there is no re-scaling
        a_nose = 0.020 #m
        b_nose = 0.010 #m
        d_nose = 0.014 #m
        
    return a_hip_0,a_hip_delta,b_hip_0,b_hip_delta,d_hip,a_nose,b_nose,d_nose

# these are the constants!
a_hip_0,a_hip_delta,b_hip_0,b_hip_delta,d_hip,a_nose,b_nose,d_nose = mouse_body_size_constants()


#%% Make a function to declare the shape of the mouse 




from utils.fitting_utils import rotate_body_model

@njit
def mouse_body_geometry(beta,gamma,s,theta,phi):
    """
    This function calculates the configuration of the mouse body
    In this configureation, it has four free parameters: azimuth and elevation of the nose/hip
    Returns the points, which define the model: center-points and radii
    theta el is elevation of the head (in xz plane)
    phi lr is head rotation in xy plane
    
    beta,gamma,s is hip pitch,yaw and spine scaling
    theta,phi is nose pitch,yaw (i.e. around y and z, respectively since major body is along x axis)
    
    """

    # get the constants for the body model
    a_hip_0,a_hip_delta,b_hip_0,b_hip_delta,d_hip,a_nose,b_nose,d_nose = mouse_body_size_constants()

    # calculate the spine
    a_hip = a_hip_0 + s * a_hip_delta
    b_hip = b_hip_0 + (1-s)**2 * b_hip_delta

    # CAlculate the nescessary rotation matrices
    R_body = rotate_body_model(0,beta,gamma)
    R_head = rotate_body_model(0,theta,phi)
    R_nose = R_body @ R_head

    # And now we get the 
    c_hip = np.array([0,0,0])
    c_mid = np.array([d_hip,0,0])
    c_nose = c_mid + R_head @ np.array([d_nose,0,0])

    # and the Q matrices
    Q_hip = R_body @ np.diag(np.array([1/a_hip**2,1/b_hip**2,1/b_hip**2])) @ R_body.T
    Q_nose = R_nose @ np.diag(np.array([1/a_nose**2,1/b_nose**2,1/b_nose**2])) @ R_nose.T
        
    # now, just return the coordinates and the radii    
    return R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose


#R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(beta,gamma,s,theta,phi)


#%% make a function to plot the mouse as a wireframe! (with points, if desired)
# add 2 cm as the center of the hip (to try is out)

#%% Now, generate a pretty good guess
    
#beta,gamma,s,theta,phi = -.1,.3,.9,.1,.4
    
from utils.fitting_utils import good_guess
    
beta,gamma,s,theta,phi,t_body = good_guess(hip_click,mid_click,nose_click)
s=.7
plot_mouse_body(beta,gamma,s,theta,phi,t_body,positions = positions,weights = None)

#%% for generating a good body model!

positions,weights = read_processed_frame(top_folder,8000)
#8000 has legs
color3d(positions)
hip_click,mid_click,nose_click = click_one_mouse(positions)
measure_dist(positions,weights,nose_click-hip_click,side = True)


beta,gamma,s,theta,phi,t_body = good_guess(hip_click,mid_click,nose_click)
s=.9
plot_mouse_body(beta,gamma,s,theta,phi,t_body,positions = positions,weights = None)


x0_guess = np.hstack((beta,gamma,s,theta,phi,t_body))

#%% Loss function for one mouse

@njit
def min_along_axis(raw):
    N = raw.shape[0]
    distances = np.empty(N)
    for i in range(N):
        distances[i] = np.min( np.abs(raw[i,:]) )
    return distances

@jit
def min_along_axis2(raw):
    return np.min(raw,1)

@njit
def jit_norm(positions):
    return np.sqrt(np.sum(np.power(positions,2),0))

@njit
def distance_to_1_mouse(x0,posx,posy,posz):   
    """
    this calculates the shortest distance from any point to the hull of the mouse, given
    the free parameters in x0. The free parameters 
    The variables, packed into x0 are:
        x0  = [beta,gamma,s,theta,phi,t_body]
            = [beta,gamma,s,theta,phi,t_body[0],t_body[1],t_body[2]] 
            
        positions is a Nx3 numpy array of the cloud points!
    
    """
    
    # and the positions have to be separate vectors for some tuple/scipy bullshit reason
    #TODO cehck if this is really true, could be more momory efficient
    positions = np.column_stack((posx,posy,posz))
    
    # first do the first mouse:
    # x0 has the parameters of the function, need to unpack the angles, the translation and the angles
    beta = x0[0] 
    gamma = x0[1]    
    s = x0[2]
    theta = x0[3]
    phi = x0[4]
    t_body = x0[5:8]
                      
    # get the coordinates c of the mouse body in it's own reference frame 
    R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(beta,gamma,s,theta,phi)

    # Now, calculate the distance vectors from the origin of the hip, mid and head 
    p_hip = (positions - ( c_hip + t_body) ).T
    p_nose = (positions - ( R_body @ c_nose + t_body) ).T
    
    # and the distance to the body 
    delta_hip_0 = np.abs( 1 - 1 / np.sqrt(np.sum(p_hip*(Q_hip @ p_hip),0))  ) *  jit_norm(p_hip) 
    delta_nose_0 = np.abs( 1 - 1 / np.sqrt(np.sum(p_nose*(Q_nose @ p_nose),0))  ) *  jit_norm(p_nose) 

    #distances = np.vstack((delta_hip_0,delta_nose_0))

    # njit trick to get the minimum distance along axis
    # not sure if actually faster, though
    # 1) First calculate the distance
    difference = delta_hip_0 - delta_nose_0
    # 2) ask if the numbers are negative
    logic = difference > 0
    # if the numbers are negative, then the the hip dist is already the smallest
    # 3) since it's pairs, this gives the minimum
    minimum_dist = delta_hip_0 - difference*logic
    
    # return the minimum distance
    return minimum_dist

@njit
def wrapped_loss(x0,posx,posy,posz):
    return np.sum(distance_to_1_mouse(x0,posx,posy,posz))

#%% Try fitting test case of one mouse body!
# select frame
positions,weights = read_processed_frame(top_folder,15000)
#8000 has legs
#plot the frame
color3d(positions)
# click the mouse to generate starting positions
hip_click,mid_click,nose_click = click_one_mouse(positions)
#convert the clicks to a guess
beta,gamma,s,theta,phi,t_body = good_guess(hip_click,mid_click,nose_click)
s=.6
# plot the mouse body
plot_mouse_body(beta,gamma,s,theta,phi,t_body,positions = positions,weights = None)

#%%

x0_guess = np.hstack((beta,gamma,s,theta,phi,t_body))

from scipy.optimize import minimize,Bounds

opt = ({'maxiter': 1000})


res = minimize(wrapped_loss, x0_guess, 
               args=(positions[:,0],positions[:,1],positions[:,2]), 
               bounds = Bounds(hard_lo,hard_hi,keep_feasible=False),
               method = 'SLSQP',options = opt)   

x_fit = res.x
optimality = res.fun


plot_mouse_body(x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],positions = positions,weights = None)



#%% Define the overall boundaries for the variables of the mouse body
    

@njit
def hard_limits():
    """
    defines the absolute hard limits on the values
    The sequence of variables is
    alpha, beta, gamma, t_body, theta, phi
    
    +-+-+-+-+ +-+-+-+-+-+-+
    |H|a|r|d| |l|i|m|i|t|s|
    +-+-+-+-+ +-+-+-+-+-+-+
    """
    # Let's set the limits of the bounding box like this:
        # we're dropping alpha, just beta, gamma, t and theta,phi
    x_range = 0.3*np.array([-1,1]) #[m]
    y_range = x_range
    z_range = 0.1*np.array([0,1]) #[m]
    
    # beta is the body pitch, from - pi/2 (vertical) to slightly more than 0 (horizontal)
    beta_range = np.pi * np.array([-.5,.1])
    # gamma range is the body yaw, i.e. body rotation
    # there is no limit on this (can be all orientations from -pi to pi)
    # but we should keep setting it between -pi and pi to not have it run off
    # computationally
    gamma_range = np.array([-np.inf,np.inf])
#    gamma_range = np.array([None,None])

    
    # now set the range for the spine scaling
    s_range = np.array([0,1]) #[a.u.]
    
    # theta is the head pitch (head up/down, around y' axis)
    # so theta = 0 is horizonal, theta<0 is looking up, theta>0 is looking down
    theta_range = np.pi * np.array([-1/3,3/4])
    
    # phi is the head yaw, i.e. from left to right
    # allow a bit more than 90 degress left/right
    phi_range = np.pi * np.array([-.6,.6])
       
    # so the boundaries are:
    bounds = np.vstack((beta_range,gamma_range,
                        s_range,
                        theta_range,phi_range,
                        x_range,y_range,z_range))
    hard_lo = bounds[:,0]
    hard_hi = bounds[:,1]
    
    return hard_lo,hard_hi



hard_lo,hard_hi = hard_limits()

#%%
start_frame = 15000
n_frames = 10000

res_holder = [None]*n_frames

x0 = x0_guess

tracking_holder = np.empty((len(x0),n_frames))

for i in range(n_frames):
    
    positions,weights = read_processed_frame(top_folder,start_frame+i)
    
    # clean positions by previous step
    prev_dist = distance_to_1_mouse(x0,positions[:,0],positions[:,1],positions[:,2])
    
    positions = positions[prev_dist < 0.02, :] 
    
    res = minimize(wrapped_loss, x0, 
               args=(positions[:,0],positions[:,1],positions[:,2]),
               bounds = Bounds(hard_lo,hard_hi,keep_feasible=False),
               method = 'SLSQP',options = opt)   
    
    # save result
#    res_holder[i] = res
    # update
    tracking_holder[:,i] = res.x
    x0 = res.x
    
    
x_fit = res.x
#x_fit[3] = 1

plt.close('all')
plot_mouse_body(x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],positions = positions,weights = None)


plot_tracking(tracking_holder,hard_lo,hard_hi)



#%% plot tracking and mouse together!
hist_tracking(np.diff(tracking_holder),hard_lo,hard_hi)

#%% do a histogram















#%%
def fit_by_op(positions,weights,x0,lo,hi):
    # x0 is the previous one, get the previous hip positions
    x0_in = x0.copy()
    bounds_low = lo.copy()
    bounds_high = hi.copy()
    
    t_0 = x0_in[2:5].copy()
    t_1 = x0_in[2+7:5+7].copy()
    
    # now, change x0 from beta,gamma,t,theta,phi to beta,gamma,gxyz,theta,phi
    # inital guess for movement is 0, perhaps update
    x0_in[2:5] = 0.
    x0_in[2+7:5+7] = 0.
    
    # set the limits of the g! all have -1
    bounds_low[2:5] = -1.
    bounds_low[2+7:5+7] = -1.
    bounds_low[2+7:4+7] = 0
    bounds_low[4+7] = 0
    
    # and we allow fast forward motion
    bounds_high[2:4] = 1.
    bounds_high[4] = 1.
    bounds_high[2+7:4+7] = 1.
    bounds_high[4+7] = 1.
    
    
    from scipy.optimize import minimize
   
    opt = ({'maxiter': 1000})
    
#    print('passing x0: ' + str(x0_in))
    # ALSO pass the previous positions
    res = minimize(wrap_for_optimizer, x0_in.copy(), args=(positions[:,0],positions[:,1],positions[:,2],weights,t_0.copy(),t_1.copy()), 
                   bounds = np.c_[bounds_low,bounds_high], method = 'SLSQP',options = opt)   
    
    
    # after the optimal fit has been found, we need to convert back from g to real t
    delta_xyz = np.array([0.01,0.00001,0.01])    
    
    # gamma are from last position
    gamma0 = x0_in[1].copy()
    gamma1 = x0_in[1+7].copy()
    
    # rotation
    Ry0 = rotate_body_model(0,0,gamma0)    
    Ry1 = rotate_body_model(0,0,gamma1)   
    
    # these are the fitted g vectors
    x_opt = res.x
    
#    print('got x_fit: ' + str(x_opt))
    
    gxyz0 = x_opt[2:5]
    gxyz1 = x_opt[2+7:5+7]
    
    # so the new positions are
    t_0_new = t_0 + Ry0 @ (gxyz0 * delta_xyz)
    t_1_new = t_1 + Ry1 @ (gxyz1 * delta_xyz)
    
    # which we also pack back
    x_opt[2:5] = t_0_new
    x_opt[2+7:5+7] = t_1_new
    
#    print('returning x_fit: ' + str(x_opt))

    return x_opt,res,res.fun






