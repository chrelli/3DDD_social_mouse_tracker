#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:22:37 2018

@author: chrelli
"""
#%% Do all the imports

import time, os, sys, shutil

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


import math

# small utilities
import csv
#from colour import Color
from itertools import compress # for list selection with logical
from tqdm import tqdm

# for image manipulation
#import cv2

# for recording and connecting to the intel realsense librar
#import pyrealsense as pyrs

#import multiprocessing
from multiprocessing import Process

# for cloud handling
#from pyntcloud import PyntCloud

# import handy Functions
#from common_utils import *
#from recording_utils import *
#from cloud_utils import *
#from fitting_utils import *

#from merge_and_filter_clouds import filter_and_downsample_cloud

# h5py for acessing data
#import h5py

# ALLSO JIT STUFF

from numba import jit, njit

#%%
"""
  ooooo o88    o8     o8                    oooo        o888o o88    o8     o8   o88
   888  oooo o888oo o888oo ooooooooo8  ooooo888       o888oo  oooo o888oo o888oo oooo  oo oooooo     oooooooo8
   888   888  888    888  888oooooo8 888    888        888     888  888    888    888   888   888  888    88o
   888   888  888    888  888        888    888        888     888  888    888    888   888   888   888oo888o
   888  o888o  888o   888o  88oooo888  88ooo888o      o888o   o888o  888o   888o o888o o888o o888o 888     888
8o888                                                                                               888ooo888
"""

#%% A few geometry functions
@njit
def unit_vector(v):
    if np.sum(v) != 0:
        v = v/np.sqrt(v[0]**2+v[1]**2+v[2]**2 )
    return v

a=unit_vector(np.array([.2,0.,1.]))

#@njit
def angle_between(v1, v2):
    """ Returns the SIGNED!! angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    angle = np.arccos(np.dot(v1_u, v2_u))
    cross_product = np.cross(v1_u,v2_u)
    ez = np.array([0,0,1.])

    # check for the sign!
    if np.dot(cross_product,ez)< 0:
        return -angle
    else:
        return angle




#%%
"""
 __  __  ___  _   _ ____  _____   ____   ___  ______   __
|  \/  |/ _ \| | | / ___|| ____| | __ ) / _ \|  _ \ \ / /
| |\/| | | | | | | \___ \|  _|   |  _ \| | | | | | \ V /
| |  | | |_| | |_| |___) | |___  | |_) | |_| | |_| || |
|_|  |_|\___/ \___/|____/|_____| |____/ \___/|____/ |_|

"""


@njit
def mouse_body_size_constants(body_scale = 1,use_old=False,use_experimental=False):
    """
    Now, we make a function, which spits out the constants
    """
    ## HIP is a prolate ellipsoid, centered along the x axis
    a_hip_min = 0.01/2 #m
    a_hip_max = 0.055/2 #m
    b_hip_min = 0.03/2 #m
    b_hip_max = 0.035/2 #m, was 0.046, which was too much
    d_hip = 0.019 #m

    # converting it to the new terminology
    a_hip_0     = body_scale*a_hip_min #m
    a_hip_delta = body_scale*(a_hip_max - a_hip_min) #m
    b_hip_0     = body_scale*b_hip_min #m
    b_hip_delta = body_scale*(b_hip_max - b_hip_min) #m

    ## NOSE is prolate ellipsoid, also along the head direction vector
    # here, there is no re-scaling
    a_nose = body_scale*0.045/2 #m was .04
    b_nose = body_scale*0.025/2 #m
    d_nose = body_scale*0.016 #m

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

    if use_experimental:
        ## HIP is a prolate ellipsoid, centered along the x axis
        a_hip_min = 0.01/2 #m
        a_hip_max = 0.1/2 #m
        b_hip_min = 0.03/2 #m
        b_hip_max = 0.035/2 #m, was 0.046, which was too much
        d_hip = 0.019 #m

        # converting it to the new terminology
        a_hip_0     = body_scale*a_hip_min #m
        a_hip_delta = body_scale*(a_hip_max - a_hip_min) #m
        b_hip_0     = body_scale*b_hip_min #m
        b_hip_delta = body_scale*(b_hip_max - b_hip_min) #m

        ## NOSE is prolate ellipsoid, also along the head direction vector
        # here, there is no re-scaling
        a_nose = body_scale*0.045/2 #m was .04
        b_nose = body_scale*0.025/2 #m
        d_nose = body_scale*0.016 #m

        a_nose = 1e-30
        b_nose = 1e-30

    return a_hip_0,a_hip_delta,b_hip_0,b_hip_delta,d_hip,a_nose,b_nose,d_nose

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
    b_hip = b_hip_0 + (1-s)**1 * b_hip_delta

    # scale the hip position
    d_hip = .75*a_hip # tried this, no good
    # d_hip = a_hip - a_nose

    # CAlculate the nescessary rotation matrices
    R_body = rotate_body_model(0,beta,gamma)
    R_head = rotate_body_model(0,theta,phi)
    R_nose = R_body @ R_head

    # And now we get the spine coordinates
    c_hip = np.array([0,0,0])
    c_mid = np.array([d_hip,0,0])
    c_nose = c_mid + R_head @ np.array([d_nose,0,0])

    # and the Q matrices
    Q_hip = R_body @ np.diag(np.array([1/a_hip**2,1/b_hip**2,1/b_hip**2])) @ R_body.T
    Q_nose = R_nose @ np.diag(np.array([1/a_nose**2,1/b_nose**2,1/b_nose**2])) @ R_nose.T

    # now, just return the coordinates and the radii
    return R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose



@njit
def only_hip_geometry(beta,gamma,s,theta,phi):
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
    ## HIP is a prolate ellipsoid, centered along the x axis
    a_hip_min = 0.01/2 #m
    a_hip_max = 0.1/2 #m
    b_hip_min = 0.03/2 #m
    b_hip_max = 0.035/2 #m, was 0.046, which was too much
    d_hip = 0.019 #m

    body_scale = 1 #

    # converting it to the new terminology
    a_hip_0     = body_scale*a_hip_min #m
    a_hip_delta = body_scale*(a_hip_max - a_hip_min) #m
    b_hip_0     = body_scale*b_hip_min #m
    b_hip_delta = body_scale*(b_hip_max - b_hip_min) #m

    # calculate the spine
    a_hip = a_hip_0 + s * a_hip_delta
    b_hip = b_hip_0 + (1-s)**1 * b_hip_delta

    R_body = rotate_body_model(0,beta,gamma)

    # And now we get the spine coordinates
    c_hip = np.array([0,0,0])

    # and the Q matrices
    Q_hip = R_body @ np.diag(np.array([1/a_hip**2,1/b_hip**2,1/b_hip**2])) @ R_body.T

    # now, just return the coordinates and the radii
    return R_body,c_hip,a_hip,b_hip,Q_hip


#%% Loss function for one mouse
"""
 _      ____   _____ _____   ______ _    _ _   _  _____
| |    / __ \ / ____/ ____| |  ____| |  | | \ | |/ ____|
| |   | |  | | (___| (___   | |__  | |  | |  \| | |
| |   | |  | |\___ \\___ \  |  __| | |  | | . ` | |
| |___| |__| |____) |___) | | |    | |__| | |\  | |____
|______\____/|_____/_____/  |_|     \____/|_| \_|\_____|

"""

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
def distance_to_one_mouse(x0,posx,posy,posz):
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

    # extremely hard oulier removal - TODO might be computationally unstable
    # but since it's gradient decent inside with SLSQR, it might be fine :-?
    # cut = 0.02
    # minimum_dist[minimum_dist>cut] = cut
    #    logic_filter = minimum_dist>cut
    #    minimum_dist[logic_filter] = np.sqrt(minimum_dist[logic_filter])

    # return the minimum distance
    return minimum_dist


@njit
def distance_to_hip_only(x0,posx,posy,posz):
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
    R_body,c_hip,a_hip,b_hip,Q_hip = only_hip_geometry(beta,gamma,s,theta,phi)

    # Now, calculate the distance vectors from the origin of the hip, mid and head
    p_hip = (positions - ( c_hip + t_body) ).T

    # and the distance to the body
    delta_hip_0 = np.abs( 1 - 1 / np.sqrt(np.sum(p_hip*(Q_hip @ p_hip),0))  ) *  jit_norm(p_hip)

    return delta_hip_0








@njit
def wrapped_loss(x0,posx,posy,posz):
    return np.sum(distance_to_1_mouse(x0,posx,posy,posz))

@njit
def wrapped_loss_one_mouse(x0,posx,posy,posz,weights = None):
    # this is simply the sum of squares, with a cutoff
    if weights is None:
        return np.sum(distance_to_one_mouse(x0,posx,posy,posz)**2)
    else:
        return np.sum(weights*distance_to_one_mouse(x0,posx,posy,posz)**2)

@njit
def wrapped_loss3(x0,pos_list):
    # this was for a kind of moving average thing, not that useful
    dists = [np.sum(distance_to_one_mouse(x0,positions[:,0],positions[:,1],positions[:,2])**2) for positions in pos_list]

    return sum(dists)

def load_timed_list(start_frame=15000,NN=3):
    # this loads a list of positions before and after each frame!
    pos_list =[read_processed_frame(top_folder,start_frame+int(i))[0] for i in np.arange(-np.floor(NN/2),np.ceil(NN/2))]
    return pos_list

#%%
"""
+-+-+-+-+ +-+-+-+-+ +-+-+-+ +-+-+-+-+
|l|o|s|s| |f|u|n|c| |t|w|o| |m|i|c|e|
+-+-+-+-+ +-+-+-+-+ +-+-+-+ +-+-+-+-+
"""
# There is room to optimize this!
@njit
def distance_to_two_mice(x0,posx,posy,posz):
    # here, we get passed x0, which is just a concatenation of x0 for both mice!
    x0_0 =x0[0:8]
    x0_1 = x0[8:]
    #todo make this parallel?
    min_dist_0 = distance_to_one_mouse(x0_0,posx,posy,posz)
    min_dist_1 = distance_to_one_mouse(x0_1,posx,posy,posz)

    # the cutoff for soft L1 loss is aready applied
    # now, we do the trick to get the smallest distance of both, in the jitted way!
    # todo, also make parallel?
    # 1) First calculate the distance
    difference = min_dist_0 - min_dist_1
    # 2) ask if the numbers are negative
    logic = difference > 0
    # if the numbers are negative, then the the hip dist is already the smallest
    # 3) since it's pairs, this gives the minimum
    minimum_dist = min_dist_0 - difference*logic

    return minimum_dist

@njit
def wrapped_loss_two_mice(x0,posx,posy,posz,weights):
    # returns the squared distance (with the soft cut from above funct)
    minimum_dist = distance_to_two_mice(x0,posx,posy,posz)
    if weights is None:
        return np.sum(minimum_dist**2)
    else:
        return np.sum(weights*minimum_dist**2)

@njit
def wrapped_loss_one_mouse_first_constant(x0_tune,x0_fixed,posx,posy,posz):
    # this is simply the sum of squares, with a cutoff
    x0 = np.concatenate((x0_fixed,x0_tune))
    minimum_dist = distance_to_two_mice(x0,posx,posy,posz)
    return np.sum(minimum_dist**2)


@njit
def wrapped_loss_one_mouse_second_constant(x0_tune,x0_fixed,posx,posy,posz):
    x0 = np.concatenate((x0_tune,x0_fixed))
    # this is simply the sum of squares, with a cutoff
    minimum_dist = distance_to_two_mice(x0,posx,posy,posz)
    return np.sum(minimum_dist**2)

#%%
# There is room to optimize this!
njit
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def wrapped_for_pool(tup_of_stuff):
    return distance_to_1_mouse(tup_of_stuff[0],tup_of_stuff[1],tup_of_stuff[2],tup_of_stuff[3])

def distance_to_two_mice_par(x0,posx,posy,posz):
    # here, we get passed x0, which is just a concatenation of x0 for both mice!
    x0_0 =x0[0:8]
    x0_1 = x0[8:]
    #todo make this parallel?
    pool = ThreadPool(2)
    min_dist_0 = distance_to_1_mouse(x0_0,posx,posy,posz)
    min_dist_1 = distance_to_1_mouse(x0_1,posx,posy,posz)

    results = pool.map(wrapped_for_pool,[(x0_0,posx,posy,posz),(x0_1,posx,posy,posz)])
    pool.close()
    pool.join()

    min_dist_0 = results[0]
    min_dist_1 = results[1]

    # the cutoff for soft L1 loss is aready applied
    # now, we do the trick to get the smallest distance of both, in the jitted way!
    # todo, also make parallel?
    # 1) First calculate the distance
    difference = min_dist_0 - min_dist_1
    # 2) ask if the numbers are negative
    logic = difference > 0
    # if the numbers are negative, then the the hip dist is already the smallest
    # 3) since it's pairs, this gives the minimum
    minimum_dist = min_dist_0 - difference*logic
    return minimum_dist

def wrapped_loss_two_mice_par(x0,posx,posy,posz):
    # returns the squared distance (with the soft cut from above funct)
    minimum_dist = distance_to_two_mice_par(x0,posx,posy,posz)
    return np.sum(minimum_dist**2)
#%% FIRST we define the limits
"""
 _      _           _ _
| |    (_)         (_) |
| |     _ _ __ ___  _| |_ ___
| |    | | '_ ` _ \| | __/ __|
| |____| | | | | | | | |_\__ \
|______|_|_| |_| |_|_|\__|___/

"""

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
    z_range = np.array([0.02,.1]) #[m]

    # beta is the body pitch, from - pi/2 (vertical) to slightly more than 0 (horizontal)
    beta_range = np.pi * np.array([-.6,.1])
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
    theta_range = np.pi * np.array([-1/2,1])

    # phi is the head yaw, i.e. from left to right
    # allow a bit more than 90 degress left/right
    phi_range = np.pi *0.7* np.array([-1,1])

    # so the boundaries are:
    bounds = np.vstack((beta_range,gamma_range,
                        s_range,
                        theta_range,phi_range,
                        x_range,y_range,z_range))
    hard_lo = bounds[:,0]
    hard_hi = bounds[:,1]

    return hard_lo,hard_hi


@njit
def box_bounds():
    """
    defines the absolute hard limits on the values
    The sequence of variables is
    alpha, beta, gamma, t_body, theta, phi

    +-+-+-+-+ +-+-+-+-+-+-+
    |H|a|r|d| |l|i|m|i|t|s|
    +-+-+-+-+ +-+-+-+-+-+-+
    """
    # Let's set the limits of the bounding box like this:
    x_range = 0.005*np.array([-1,1]) #[m]
    y_range = x_range
    z_range = x_range #[m]

    # beta is the body pitch, from - pi/2 (vertical) to slightly more than 0 (horizontal)
    beta_range = 0.2*np.array([-1,1])
    # gamma range is the body yaw, i.e. body rotation
    # there is no limit on this (can be all orientations from -pi to pi)
    # but we should keep setting it between -pi and pi to not have it run off
    # computationally
    gamma_range = 0.2*np.array([-1,1])
#    gamma_range = np.array([None,None])


    # now set the range for the spine scaling
    s_range = .2*np.array([-1,1]) #[a.u.]

    # theta is the head pitch (head up/down, around y' axis)
    # so theta = 0 is horizonal, theta<0 is looking up, theta>0 is looking down
    theta_range = 0.2*np.array([-1,1])

    # phi is the head yaw, i.e. from left to right
    # allow a bit more than 90 degress left/right
    phi_range = 0.2*np.array([-1,1])

    # so the boundaries are:
    bounds = np.vstack((beta_range,gamma_range,
                        s_range,
                        theta_range,phi_range,
                        x_range,y_range,z_range))

    box_lo = bounds[:,0]
    box_hi = bounds[:,1]

    return box_lo,box_hi


#%%
"""
 _____                  _       _   _   _
|  ___|__  _ __   _ __ | | ___ | |_| |_(_)_ __   __ _
| |_ / _ \| '__| | '_ \| |/ _ \| __| __| | '_ \ / _` |
|  _| (_) | |    | |_) | | (_) | |_| |_| | | | | (_| |
|_|  \___/|_|    | .__/|_|\___/ \__|\__|_|_| |_|\__, |
                 |_|                            |___/
"""

def click_one_mouse(positions):

    ###############
    # Show a 2D plot and ask for two clicks
    ###############
    plt.figure()
    plt.scatter(positions[:,0],positions[:,1],c=positions[:,2]/np.max(positions[:,2]),s=5)
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('click center of hip, then mid, then head of mouse!')
    w,h = 570,800
    # plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)

    click_points = np.asanyarray(plt.ginput(3))
    hip_click = click_points[0]
    mid_click = click_points[1]
    nose_click = click_points[2]

    # plt.show()

    ###############
    # Now calculate a reference direction
    ###############
    v_click = nose_click-hip_click
    # and add to the plot
    def add_vec_from_point(c_mid_est,v_ref_est):
        data = np.vstack((c_mid_est,c_mid_est+v_ref_est))
        plt.plot(data[:,0],data[:,1],c='red')
        plt.plot(data[0,0],data[0,1],c='red',marker='o')


    plt.figure()
    plt.scatter(positions[:,0],positions[:,1],c=positions[:,2]/np.max(positions[:,2]),s=5)
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    add_vec_from_point(hip_click,v_click)
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('estimated hip and heading direction')
    w,h = 570,800
    # plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)

    plt.show()

    return hip_click,mid_click,nose_click


def open_3d():
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #   3D plot of Sphere
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def close_3d(ax,positions):
    """
    The positions keyword is a bit silly, but this is just used to estimate
    the min and max of the axes, so that all are visible
    """
    # ax.set_aspect('equal')

    X,Y,Z = positions[:,0], positions[:,1], positions[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    ax.set_xlabel('x (mm)',fontsize=16)
    ax.set_ylabel('y (mm)',fontsize=16)
    zlabel = ax.set_zlabel('z (mm)',fontsize=16)

def add_mouse_to_axes(ax,beta,gamma,s,theta,phi,t_body,color = 'r',alpha = .7):
    # get the geometry of the mouse body # not really the preferred way
    R_body,R_nose,c_mid,c_hip,c_nose, \
     a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose \
     = mouse_body_geometry(beta,gamma,s,theta,phi)

    # We have to plot two ellipses

    # FIRST PLOT THE ELLIPSE, which is the hip
    # generate points on a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    # get the mesh, by using the equation of an ellipsoid
    x=np.cos(u)*a_hip
    y=np.sin(u)*np.sin(v)*b_hip
    z=np.sin(u)*np.cos(v)*b_hip

    # pack to matrix of positions
    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # apply the rotatation and unpack
    posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    # reshape for wireframe
    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

    ax.plot_wireframe(x, y, z, color=color,alpha = alpha)

    # THEN PLOT THE ELLIPSE, which is the nose
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    x=np.cos(u)*a_nose
    y=np.sin(u)*np.sin(v)*b_nose
    z=np.sin(u)*np.cos(v)*b_nose

    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # kind of old, but w/e
    R_head = rotate_body_model(0,theta,phi)

    posi_rotated = ((R_body @ ( (R_head @ posi).T + c_nose).T ).T + t_body).T
#    posi_rotated = ((R_nose @ (posi.T + c_nose).T ).T + t_body).T

    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

    # make the head green!
#    ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)
    ax.plot_wireframe(x, y, z, color='g',alpha = alpha)

    c_hip = R_body @ c_hip + t_body
    c_mid = R_body @ c_mid + t_body
    c_nose = R_body @ c_nose + t_body

    # Add the points for the skeleton
    ax.scatter(c_mid[0],c_mid[1],c_mid[2],c='orange',s=150,alpha = alpha)
    ax.scatter(c_hip[0],c_hip[1],c_hip[2],c='k',s=150,alpha = alpha)
    ax.scatter(c_nose[0],c_nose[1],c_nose[2],c='orange',s=150,alpha = alpha)

    # and the spine
    cc = np.vstack((c_hip,c_mid,c_nose))
    ax.plot(cc[:,0],cc[:,1],cc[:,2],color='orange',linewidth=4,alpha = alpha)




def plot_mouse_body(beta,gamma,s,theta,phi,t_body,positions = None,weights = None):
    # open thie 3d axes
#    if ax is None:
    fig, ax = open_3d()

    if positions is not None:
        if weights is not None:
            # plot the positions, color by weights
            ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=weights/np.max(weights),alpha=0.8,marker='o',s=10)
        else:
            ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='b',alpha=0.09,marker='o',s=10)


    # FUNCTION to add mouse body to plot
    add_mouse_to_axes(ax,beta,gamma,s,theta,phi,t_body,color = 'r')
    w,h = 570,800
    plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)
    close_3d(ax,np.vstack((np.array([-.2,-.2,0]),np.array([.05,.1,.1]))))


def plot_tracking(tracking_holder,hard_lo,hard_hi):
    beta_fit = tracking_holder[0,:]
    gamma_fit = tracking_holder[1,:]
    s_fit = tracking_holder[2,:]
    theta_fit = tracking_holder[3,:]
    phi_fit = tracking_holder[4,:]
    t_fit = tracking_holder[5:8,:]

    what_to_plot = [beta_fit,gamma_fit,s_fit,theta_fit,phi_fit,t_fit]
    legends = ['$ \\beta $ (hip pitch)','$ \gamma $ (hip yaw)','s (spine scaling)',
               '$ \\theta $ (head pitch)','$ \phi $ (head yaw)']

    plt.figure()

#    import matplotlib
#    pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
#    matplotlib.rcParams.update(pgf_with_rc_fonts)
#    from matplotlib import rcParams
#    rcParams['font.family'] = 'serif'
#    rcParams['text.usetex'] = True
    for i in range(5):
        plt.subplot(6,1,i+1)
        plt.plot(what_to_plot[i])
        plt.legend([legends[i]],loc='right')
        plt.axhline(y=hard_lo[i],c='r')
        plt.axhline(y=0,c='k')

        plt.axhline(y=hard_hi[i],c='r')

    plt.subplot(6,1,6)
    plt.plot(what_to_plot[5].T)
    plt.axhline(y=0,c='k')

    plt.legend(['$t_x$ (body)','$t_y$ (body)','$t_z$ (body)'],loc = 'right')
    plt.xlabel('frame no.')



def hist_tracking(tracking_holder,hard_lo,hard_hi):
    beta_fit = tracking_holder[0,:]
    gamma_fit = tracking_holder[1,:]
    s_fit = tracking_holder[2,:]
    theta_fit = tracking_holder[3,:]
    phi_fit = tracking_holder[4,:]
    t_fit = tracking_holder[5:8,:]

    what_to_plot = [beta_fit,gamma_fit,s_fit,theta_fit,phi_fit,t_fit]
    legends = ['$ \\beta $ (hip pitch)','$ \gamma $ (hip yaw)','s (spine scaling)',
               '$ \\theta $ (head pitch)','$ \phi $ (head yaw)']

#    plt.figure()
#    from matplotlib import rcParams
#    rcParams['font.family'] = 'serif'
#    rcParams['text.usetex'] = True
    for i in range(5):
        plt.subplot(6,1,i+1)
        plt.hist(what_to_plot[i],100)
        plt.legend([legends[i]])
#        plt.axhline(y=hard_lo[i],c='r')
#        plt.axhline(y=0,c='k')

#        plt.axhline(y=hard_hi[i],c='r')

    plt.subplot(6,1,6)
    plt.hist(what_to_plot[5].T,100)
    plt.legend(['$t_x$ (body)','$t_y$ (body)','$t_z$ (body)'])
    plt.xlabel('frame no.')



def add_mouse_for_video(ax,beta,gamma,s,theta,phi,t_body,color = 'r',alpha = .7):
    # get the geometry of the mouse body # not really the preferred way
    R_body,R_nose,c_mid,c_hip,c_nose, a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(beta,gamma,s,theta,phi)

    # We have to plot two ellipses

    # FIRST PLOT THE ELLIPSE, which is the hip
    # generate points on a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    # get the mesh, by using the equation of an ellipsoid
    x=np.cos(u)*a_hip
    y=np.sin(u)*np.sin(v)*b_hip
    z=np.sin(u)*np.cos(v)*b_hip
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    # pack to matrix of positions
    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # apply the rotatation and unpack
    posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    # reshape for wireframe
    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

    h_hip = ax.plot_wireframe(x, y, z, color=color,alpha = alpha)

    # THEN PLOT THE ELLIPSE, which is the nose
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    x=np.cos(u)*a_nose
    y=np.sin(u)*np.sin(v)*b_nose
    z=np.sin(u)*np.cos(v)*b_nose

    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # kind of old, but w/e
    R_head = rotate_body_model(0,theta,phi)

    posi_rotated = ((R_body @ ( (R_head @ posi).T + c_nose).T ).T + t_body).T
#    posi_rotated = ((R_nose @ (posi.T + c_nose).T ).T + t_body).T

    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

#    h_nose = ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)
    h_nose = ax.plot_wireframe(x, y, z, color='green',alpha = alpha)

#
#    c_hip = R_body @ c_hip + t_body
#    c_mid = R_body @ c_mid + t_body
#    c_nose = R_body @ c_nose + t_body

#    # Add the points for the skeleton
#    ax.scatter(c_mid[0],c_mid[1],c_mid[2],c='blue',s=150)
#    ax.scatter(c_hip[0],c_hip[1],c_hip[2],c='k',s=150)
#    ax.scatter(c_nose[0],c_nose[1],c_nose[2],c='blue',s=150)

    # and the spine
#    cc = np.vstack((c_hip,c_mid,c_nose))
#    h_spine = ax.plot(cc[:,0],cc[:,1],cc[:,2],color='k',linewidth=4)

    return ax,h_hip,h_nose
'''
  _   _   _   _   _   _   _   _     _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \   / \ / \ / \ / \
( v | e | r | s | i | o | n | s ) ( w | i | t | h )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/   \_/ \_/ \_/ \_/
  _   _   _   _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( b | o | d | y | _ | c | o | n | s | t | a | n | t | s )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
'''
#%% New versions with variable body geometry!


@njit
def geo_mouse_body_geometry(alpha,beta,gamma,s,psi,theta,phi,body_constants):
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
    # a_hip_0,a_hip_delta,b_hip_0,b_hip_delta,d_hip,a_nose,b_nose,d_nose = mouse_body_size_constants()

    body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl = body_constants

    a_hip_0     = body_scale*a_hip_min
    a_hip_delta = body_scale*(a_hip_max - a_hip_min)
    b_hip_0     = body_scale*b_hip_min
    b_hip_delta = body_scale*(b_hip_max - b_hip_min)




    # calculate the spine
    a_hip = a_hip_0 + s * a_hip_delta
    b_hip = b_hip_0 + (1-s)**1 * b_hip_delta

    # scale the hip position
    d_hip = .75*a_hip # tried this, no good
    # d_hip = a_hip - a_nose

    # CAlculate the nescessary rotation matrices
    R_body = rotate_body_model(alpha,beta,gamma)
    R_head = rotate_body_model(psi,theta,phi)
    R_nose = R_body @ R_head

    # And now we get the spine coordinates
    c_hip = np.array([0,0,0])
    c_mid = np.array([d_hip,0,0])
    c_nose = c_mid + R_head @ np.array([d_nose,0,0])

    # and the Q matrices
    Q_hip = R_body @ np.diag(np.array([1/a_hip**2,1/b_hip**2,1/b_hip**2])) @ R_body.T
    Q_nose = R_nose @ np.diag(np.array([1/a_nose**2,1/b_nose**2,1/b_nose**2])) @ R_nose.T

    # now, just return the coordinates and the radii
    return R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose


def add_geo_mouse_for_video(ax,aleph,beta,gamma,s,theta,psi,phi,t_body,body_constants,color = 'r',alpha = .7, implant=False):
    # this also need a vector
    # get the geometry of the mouse body # not really the preferred way
    R_body,R_nose,c_mid,c_hip,c_nose, a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose = geo_mouse_body_geometry(aleph,beta,gamma,s,psi,theta,phi,body_constants)

    print("plotting thinks that c_nose is {}".format(c_nose))
    print("plotting thinks that c_mid is {}".format(c_mid))

    h_hip,h_nose,h_impl = None,None,None
    # We have to plot two ellipses

    # FIRST PLOT THE ELLIPSE, which is the hip
    # generate points on a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    # get the mesh, by using the equation of an ellipsoid
    x=np.cos(u)*a_hip
    y=np.sin(u)*np.sin(v)*b_hip
    z=np.sin(u)*np.cos(v)*b_hip

    # pack to matrix of positions
    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # apply the rotatation and unpack
    posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    # reshape for wireframe
    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

    h_hip = ax.plot_wireframe(x, y, z, color=color,alpha = alpha)

    # THEN PLOT THE ELLIPSE, which is the nose
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    x=np.cos(u)*a_nose
    y=np.sin(u)*np.sin(v)*b_nose
    z=np.sin(u)*np.cos(v)*b_nose

    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # kind of old, but w/e
    R_head = rotate_body_model(psi,theta,phi)

    # posi_rotated = ((R_body @ ( (R_head @ posi).T + c_nose).T ).T + t_body).T
    posi_rotated = R_body @ ((R_head @ (posi.T + c_nose).T ).T + t_body).T

    vvv = c_nose+t_body
    ax.scatter(vvv[0],vvv[1],vvv[2],c='r',s = 900)

    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

#    h_nose = ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)
    h_nose = ax.plot_wireframe(x, y, z, color='green',alpha = alpha)
    if implant:

        body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl = body_constants
        d_hip = .75*a_hip

        c_mid = np.array([d_hip,0,0])
        c_impl = c_mid + R_head @ np.array([x_impl,0,z_impl])

        # THEN PLOT THE ELLIPSE, which is the nose
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        x=np.cos(u)*r_impl
        y=np.sin(u)*np.sin(v) * r_impl
        z=np.sin(u)*np.cos(v) *r_impl

        posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

        # kind of old, but w/e
        R_head = rotate_body_model(psi,theta,phi)

        posi_rotated = ((R_body @ ( (R_head @ posi).T + c_impl).T ).T + t_body).T
    #    posi_rotated = ((R_nose @ (posi.T + c_nose).T ).T + t_body).T

        x = posi_rotated[0,:]
        y = posi_rotated[1,:]
        z = posi_rotated[2,:]

        x = np.reshape(x, (u.shape) )
        y = np.reshape(y, (u.shape) )
        z = np.reshape(z, (u.shape) )

    #    h_nose = ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)
        h_impl = ax.plot_wireframe(x, y, z, color='blue',alpha = alpha)





#
#    c_hip = R_body @ c_hip + t_body
#    c_mid = R_body @ c_mid + t_body
#    c_nose = R_body @ c_nose + t_body

#    # Add the points for the skeleton
#    ax.scatter(c_mid[0],c_mid[1],c_mid[2],c='blue',s=150)
#    ax.scatter(c_hip[0],c_hip[1],c_hip[2],c='k',s=150)
#    ax.scatter(c_nose[0],c_nose[1],c_nose[2],c='blue',s=150)

    # and the spine
#    cc = np.vstack((c_hip,c_mid,c_nose))
#    h_spine = ax.plot(cc[:,0],cc[:,1],cc[:,2],color='k',linewidth=4)

    return ax,h_hip,h_nose,h_impl






#%%
"""
       _     _              _          _
__   _(_) __| | ___  ___   | |__   ___| |_ __   ___ _ __
\ \ / / |/ _` |/ _ \/ _ \  | '_ \ / _ \ | '_ \ / _ \ '__|
 \ V /| | (_| |  __/ (_) | | | | |  __/ | |_) |  __/ |
  \_/ |_|\__,_|\___|\___/  |_| |_|\___|_| .__/ \___|_|
                                        |_|
"""

def run_video(tracking_holder,start_frame,top_folder,read_processed_frame,n_frames=None,
              decimate = 10,export = False,text_line="_",outlier_cut=0.08,min_spine=None):

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    import time

#    from utils.fitting_utils import add_mouse_for_video

    # close all. to increase speed
    plt.close('all')

    if export:
        # import animation and declare writer
        import matplotlib.animation as manimation

#        FFMpegWriter = manimation.writers['ffmpeg']
        FFMpegWriter = manimation.writers['ffmpeg']

        metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')

        writer = FFMpegWriter(fps=30, metadata=metadata)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = 'video_'+text_line+timestr+'.mp4'
        print(file_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make the X, Y meshgrid.
    xs = np.linspace(-1, 1, 50)
    ys = np.linspace(-1, 1, 50)

    ax.set_xlim(-.2,.2)
    ax.set_ylim(-.2,.2)
    ax.set_zlim(0,.4)

    ax.view_init(62, 54)

    ax.view_init(84, 19)

    ax.view_init(19, 18)


    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    w,h = 600,600
    plt.get_current_fig_manager().window.setGeometry(2000,60,w,h)

    plt.pause(0.001)

    scat = None
    h_spine = None
    h_hip_0 = None
    h_nose_0 = None
    h_hip_1 = None
    h_nose_1 = None
    # If a line collection is already remove it before drawing.

    tstart = time.time()

    alpha = .5
    size = .1

    if not n_frames:
        n_frames = tracking_holder.shape[1]

    if export:
        with writer.saving(fig, file_name, dpi=100):
            for i in tqdm(range(n_frames)):

                if not i%decimate==0:
                    continue

                if scat:
                    ax.collections.remove(scat)
                    ax.collections.remove(scat2)

                if h_hip_0:
                    ax.collections.remove(h_hip_0)
                if h_nose_0:
                    ax.collections.remove(h_nose_0)
                if h_hip_1:
                    ax.collections.remove(h_hip_1)
                if h_nose_1:
                    ax.collections.remove(h_nose_1)

                # load the positions and scatter them
                positions,weights = read_processed_frame(top_folder,start_frame+i)
#                scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='b',alpha=alpha,marker='o',s=size)
                scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=weights/np.max(weights),alpha=alpha,marker='o',s=size)


                # also add the wireframe body
                x_fit = tracking_holder[:,i]
                ax,h_hip_0,h_nose_0 = add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],color = 'r')
                if len(x_fit) > 8:
                    x_fit = tracking_holder[8:,i]
                    ax,h_hip_1,h_nose_1 = add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],color = 'orange')

                # and the outliers
                prev_dist = distance_to_one_mouse(tracking_holder[:,i-1],positions[:,0],positions[:,1],positions[:,2])
                outliers = positions[prev_dist > outlier_cut, :]
                scat2 = ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2],c='orange',alpha=1,marker='o',s=size)

                ax.set_xlabel('x [m], frame #'+str(i))
                if min_spine is not None:
                    ax.set_ylabel('y [m], min_spine: '+str(min_spine[i]))


                writer.grab_frame()


    else:

        for i in range(n_frames):

            if not i%decimate==0:
                continue

            if scat:
                ax.collections.remove(scat)
                ax.collections.remove(scat2)

            if h_hip_0:
                ax.collections.remove(h_hip_0)
            if h_nose_0:
                ax.collections.remove(h_nose_0)
            if h_hip_1:
                ax.collections.remove(h_hip_1)
            if h_nose_1:
                ax.collections.remove(h_nose_1)

            # load the positions and scatter them
            positions,weights = read_processed_frame(top_folder,start_frame+i)
#            scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='b',alpha=alpha,marker='o',s=size)
            scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=weights/np.max(weights),alpha=alpha,marker='o',s=size)

            # also add the wireframe body
            x_fit = tracking_holder[:,i]
            ax,h_hip_0,h_nose_0 = add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],color = 'r')
            if len(x_fit) > 8:
                x_fit = tracking_holder[8:,i]
                ax,h_hip_1,h_nose_1 = add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],color = 'orange')

            # and the outliers
            prev_dist = distance_to_one_mouse(tracking_holder[:,i-1],positions[:,0],positions[:,1],positions[:,2])
            outliers = positions[prev_dist > outlier_cut, :]
            scat2 = ax.scatter(outliers[:,0],outliers[:,1],outliers[:,2],c='orange',alpha=1,marker='o',s=size)

            ax.set_xlabel('x [m], frame #'+str(i))
            if min_spine is not None:
                ax.set_ylabel('y [m], min_spine: '+str(min_spine[i]))

            plt.pause(.001)


    tstop = time.time()
    print('Ran '+  str(n_frames)+' frames')

    print('Average FPS: %f' % (100 / (tstop - tstart)))




#%%

"""
 _   _      _       _
| | | | ___| |_ __ (_)_ __   __ _
| |_| |/ _ \ | '_ \| | '_ \ / _` |
|  _  |  __/ | |_) | | | | | (_| |
|_| |_|\___|_| .__/|_|_| |_|\__, |
             |_|            |___/
"""
#
# def good_guess(hip_click,mid_click,nose_click):
#     # translation vector, which moves the center of the mouse body
#     z_guess = 0.022 # guess the z as around 2 cm
#     t_body = np.append(hip_click,z_guess)
#
#     # set the scaling
#     s = 0.5
#
#     # guess the rotation as the body model:
#     # - alpha is around x axis, none, I guess
#     # - beta is around y axis, so elevation
#     beta = 0
#     # - gamma is around z, so in the xy plane, the left-right angle of the mouse wrt. e^hat_x
#         # get the vector
#     v_click = mid_click-hip_click
#     # make it a 3d vector to use with the handy function
#     target = np.append(v_click,0)
#     angle_with_x = angle_between(np.array([1.,0,0]),target)
#     gamma = angle_with_x
#
#     # same with the head
#     theta = 0
#     v_click = nose_click-mid_click
#     # make it a 3d vector to use with the handy function
#     target = np.append(v_click,0)
#     angle_with_x = angle_between(np.array([1.,0,0]),target)
#     phi = angle_with_x - gamma # NB since phi is with respect to x', not x
#
#     return beta,gamma,s,theta,phi,t_body


def clean_positions_by_weights(positions,weights,cutoff = 2):
    logic_filter = weights > cutoff
    return positions[logic_filter,:],weights[logic_filter]

#%% for generating a good body model!
def measure_dist(positions,weights,v_ref,side = False):
    """
    Will plot the mouse and allow me to click and measure with two clicks
    side is false (so top view)
    but can be True, then it's cut though the major axis of hte mouse (determined by v_reference)
    """
    # simplest trick is to just rotate all points so the reference
    # direction is perpendicular to x
    v_ref = np.append(v_ref,0)
    angle_with_x = angle_between(np.array([1.,0,0]),v_ref)
    RR = rotate_body_model(0,0,-angle_with_x)

    positions = (RR @ positions.T).T - v_ref
    if side:
        xx,yy = positions[:,0],positions[:,2]
    else:
        xx,yy = positions[:,0],positions[:,1]

    #top view
    plt.figure()
    plt.scatter(xx,yy,c= weights/np.max(weights),s = 5)
#    plt.xlim([-.05,.1])
#    plt.ylim([0,.15])
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('click center of hip, then mid, then head of mouse!')


    w,h = 570,800
    plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)
    click_points = np.asanyarray(plt.ginput(0))

    if click_points.shape[0] % 2 is not 0:
        print('missing a point')
        click_points = click_points[:-1,:]

    n_clicks = click_points.shape[0]
    start_points = click_points[np.arange(n_clicks)%2==0,:]
    end_points = click_points[np.arange(n_clicks)%2==1,:]
    n_points = start_points.shape[0]

    plt.figure()
    plt.scatter(xx,yy,c= weights/np.max(weights),s = 5)
    for s,e in zip(start_points,end_points):
        plt.plot([s[0],e[0]],[s[1],e[1]],'o-')

    dist = np.linalg.norm(end_points-start_points,axis = 1)

    leg_list = [str(np.round(d,decimals = 3))+" m" for d in dist]

    plt.legend(leg_list)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.title('distance in meters')
#    plt.xlim([-.05,.1])
#    plt.ylim([0,.15])
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    timestr = time.strftime("%Y%m%d-%H%M%S")

    plt.savefig('/home/chrelli/git/3d_sandbox/mycetrack0p4/measurements/'+timestr+'.png')
    plt.show()
    w,h = 570,800
    plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)
    return dist
