f#!/usr/bin/env python3
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
#@njit
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



#@njit
def rotate_body_model(alpha_body,beta_body,gamma_body):
    """
    Returns R_body, to rotate and transform the mouse body model
    alpha,beta,gamma is rotation around x,y,z axis respectively
    https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
    """
    R_alpha = np.empty((3,3))   
    R_alpha[0,:] = [1.,0.,0.]
    R_alpha[1,:] = [0.,np.cos(alpha_body),-np.sin(alpha_body)]
    R_alpha[2,:] = [0.,np.sin(alpha_body),np.cos(alpha_body)]
    R_beta = np.empty((3,3))   
    R_beta[0,:] = [np.cos(beta_body),0.,np.sin(beta_body)]
    R_beta[1,:] = [0.,1.,0.]
    R_beta[2,:] = [-np.sin(beta_body),0.,np.cos(beta_body)]
    R_gamma = np.empty((3,3))   
    R_gamma[0,:] = [np.cos(gamma_body),-np.sin(gamma_body),0.]
    R_gamma[1,:] = [np.sin(gamma_body),np.cos(gamma_body),0.]
    R_gamma[2,:] = [0.,0.,1.]  
    
    return R_alpha@R_beta@R_gamma



#%%
"""
 __  __  ___  _   _ ____  _____   ____   ___  ______   __
|  \/  |/ _ \| | | / ___|| ____| | __ ) / _ \|  _ \ \ / /
| |\/| | | | | | | \___ \|  _|   |  _ \| | | | | | \ V / 
| |  | | |_| | |_| |___) | |___  | |_) | |_| | |_| || |  
|_|  |_|\___/ \___/|____/|_____| |____/ \___/|____/ |_|  

"""    


#@njit
def mouse_body_size_constants(body_scale = 1,use_old=False):
    """
    Now, we make a function, which spits out the constants
    """
    ## HIP is a prolate ellipsoid, centered along the x axis
    a_hip_min = 0.04/2 #m
    a_hip_max = 0.06/2 #m
    b_hip_min = 0.01/2 #m
    b_hip_max = 0.03/2 #m, was 0.046, which was too much
    d_hip = 0.019 #m

    # converting it to the new terminology
    a_hip_0     = a_hip_min #m
    a_hip_delta = a_hip_max - a_hip_min #m
    b_hip_0     = b_hip_min #m
    b_hip_delta = b_hip_max - b_hip_min #m
    
    ## NOSE is prolate ellipsoid, also along the head direction vector
    # here, there is no re-scaling
    a_nose = 0.045/2 #m was .04
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

#@njit
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
    plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)
    
    click_points = np.asanyarray(plt.ginput(3))
    hip_click = click_points[0]
    mid_click = click_points[1]
    nose_click = click_points[2]
    
    plt.show()
    
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
    plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)
    
    plt.show()
    
    return hip_click,mid_click,nose_click


def open_3d():
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    #   3D plot of Sphere
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax
        
def close_3d(ax,positions):
    """
    The positions keyword is a bit silly, but this is just used to estimate 
    the min and max of the axes, so that all are visible
    """
    ax.set_aspect('equal')
    
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
    
    
def add_mouse_to_axes(ax,beta,gamma,s,theta,phi,t_body,color = 'r'):
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

    ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)   

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

    ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)   
    
    c_hip = R_body @ c_hip + t_body
    c_mid = R_body @ c_mid + t_body
    c_nose = R_body @ c_nose + t_body 

    # Add the points for the skeleton
    ax.scatter(c_mid[0],c_mid[1],c_mid[2],c='blue',s=150)    
    ax.scatter(c_hip[0],c_hip[1],c_hip[2],c='k',s=150)    
    ax.scatter(c_nose[0],c_nose[1],c_nose[2],c='blue',s=150)
    
    # and the spine
    cc = np.vstack((c_hip,c_mid,c_nose))    
    ax.plot(cc[:,0],cc[:,1],cc[:,2],color='blue',linewidth=4)




def plot_mouse_body(beta,gamma,s,theta,phi,t_body,positions = None,weights = None):
    # open thie 3d axes
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
        plt.legend([legends[i]])
        plt.axhline(y=hard_lo[i],c='r')
        plt.axhline(y=0,c='k')

        plt.axhline(y=hard_hi[i],c='r')
        
    plt.subplot(6,1,6)
    plt.plot(what_to_plot[5].T)
    plt.axhline(y=0,c='k')

    plt.legend(['$t_x$ (body)','$t_y$ (body)','$t_z$ (body)'])
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



def add_mouse_for_video(ax,beta,gamma,s,theta,phi,t_body,color = 'r'):
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

    h_hip = ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)   

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
    h_nose = ax.plot_wireframe(x, y, z, color='green',alpha = 0.7)   

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


#%%
    
"""
 _   _      _       _             
| | | | ___| |_ __ (_)_ __   __ _ 
| |_| |/ _ \ | '_ \| | '_ \ / _` |
|  _  |  __/ | |_) | | | | | (_| |
|_| |_|\___|_| .__/|_|_| |_|\__, |
             |_|            |___/ 
"""    

def good_guess(hip_click,mid_click,nose_click):   
    # translation vector, which moves the center of the mouse body
    z_guess = 0.022 # guess the z as around 2 cm
    t_body = np.append(hip_click,z_guess)
    
    # set the scaling
    s = 0.5
    
    # guess the rotation as the body model:
    # - alpha is around x axis, none, I guess
    # - beta is around y axis, so elevation
    beta = 0
    # - gamma is around z, so in the xy plane, the left-right angle of the mouse wrt. e^hat_x
        # get the vector
    v_click = mid_click-hip_click
    # make it a 3d vector to use with the handy function
    target = np.append(v_click,0)
    angle_with_x = angle_between(np.array([1.,0,0]),target)
    gamma = angle_with_x
    
    # same with the head   
    theta = 0
    v_click = nose_click-mid_click
    # make it a 3d vector to use with the handy function
    target = np.append(v_click,0)
    angle_with_x = angle_between(np.array([1.,0,0]),target)
    phi = angle_with_x - gamma # NB since phi is with respect to x', not x  
    
    return beta,gamma,s,theta,phi,t_body


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





