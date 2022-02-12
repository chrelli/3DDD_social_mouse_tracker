#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:17:58 2018

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
from pyntcloud import PyntCloud

# import handy Functions
from common_utils import *
from recording_utils import *
from cloud_utils import *
from fitting_utils import *

#from merge_and_filter_clouds import filter_and_downsample_cloud

# h5py for acessing data
import h5py

# ALLSO JIT STUFF

from numba import jit, njit

#%% OK< WHAT DO WE NEED FOR TRACKING

def load_positions_from_h5(file_path,selected_frame,with_rgb=False):
    with h5py.File(file_path, 'r') as hf:
        # get the keys in the file
        what_keys = list(hf.keys())
        this_key = what_keys[selected_frame]
        dset = hf[this_key]
        # make the cloud
        # positions = pd.DataFrame(dset.value, columns=['x', 'y', 'z','red','blue','green'])
        if with_rgb:
            return dset.value,what_keys
        else:
            return dset.value[:,[0,1,2]],what_keys

#%%
#@jit(nopython = True)
@njit
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

rotate_body_model(.02,.1,2)

#
#
#@jit
#def rotation_matrix(eps,phi,ome):
#    ''' WHats up here, is there a mistake??'''
#    # from that spanish book, epsilo9n, phi,omega acorrespond to alpha, beta, gamma 
#    return np.array([[np.cos(phi) * np.cos(ome), np.cos(eps)*np.sin(ome) + np.sin(eps)*np.sin(phi)*np.cos(ome), np.sin(eps)*np.sin(ome) - np.cos(eps)*np.sin(phi)*np.cos(ome)],
#                      [-np.cos(phi) * np.sin(ome), np.cos(eps)*np.cos(ome) - np.sin(eps)*np.sin(phi)*np.sin(ome), np.sin(eps)*np.cos(ome)+np.cos(eps)*np.sin(phi)*np.sin(ome)],
#                      [np.sin(phi) , - np.sin(eps)*np.cos(phi), np.cos(eps)*np.cos(phi) ]])
#
#
#
#rotation_matrix(.02,.1,2)


#%% A few geometry
@njit
def unit_vector(v):
    if np.sum(v) != 0:
        v = v/np.sqrt(v[0]**2+v[1]**2+v[2]**2 )
    return v
       
a=unit_vector(np.array([.2,0.,1.]))

@njit
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

#%%
alpha,beta,gamma,theta_el,phi_lr = .1,.3,.2,1,.4

@njit
def mouse_body_geometry(alpha,beta,gamma,theta_el,phi_lr):
    """
    This function calculates the configuration of the mouse body
    In this configureation, it has four free parameters: azimuth and elevation of the nose/hip
    Returns the points, which define the model: center-points and radii
    theta el is elevation of the head (in xz plane)
    phi lr is head rotation in xy plane
    """
    #scaling of the body model
    body_scale = 0.9
    
    ## MID is simply a sphere
    c_mid = np.array([0,0,0])
    r_mid = .017 * body_scale

    ## HIP is a prolate ellipsoid, centered along the x axis
    a_hip = 0.025 * body_scale
    b_hip = 0.017 * body_scale
    d_hip = 0.72*a_hip
    c_hip = np.array([-d_hip,0,0])

    ## HEAD is a sphere, along the line defined by theta el and phi lr 90% size of r_mid
    # unit vector in head direction:
#    #TODO speed up this?
#    unit_head = np.array([math.cos(theta_el)*math.cos(phi_lr),
#                                    math.cos(theta_el)*math.sin(phi_lr),
#                                    math.sin(theta_el)])
#    
     
    R_body = rotate_body_model(alpha,beta,gamma)
    R_nose = rotate_body_model(alpha,beta+theta_el,gamma+phi_lr)


#    r_head = 0.9 * r_mid
#    d_head = 0.63 * r_mid
#    c_head = d_head * unit_head
#    
    ## NOSE is another prolate ellipsoid, also along the head direction vector
    a_nose = 0.025 * body_scale
    b_nose = 0.015 * body_scale    
    d_nose = 0.9* r_mid #same radius as the mid
    c_nose = np.array([d_nose,0,0])

    Q_hip = R_body @ np.diag(np.array([1/a_hip**2,1/b_hip**2,1/b_hip**2])) @ R_body.T
    Q_nose = R_nose @ np.diag(np.array([1/a_nose**2,1/b_nose**2,1/b_nose**2])) @ R_nose.T
    
    
    # now, just return the coordinates and the radii    
    return R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,r_mid,a_nose,b_nose,Q_hip,Q_nose

#mouse_body_geometry(alpha,beta,gamma,theta_el,phi_lr)

#%%
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

#%%
@njit    
def distance_to_body_3_piece(x0,posx,posy,posz):   
    """
    this calculates the shortest distance from any point to the hull of the mouse, given
    the free parameters in x0. The free parameters are
    - alpha,beta,gamma of body rotation
    - t_body, for centering
    - angles for body posture: theta,phi for hip and nose
    """
    # x0 has the parameters of the function, need to unpack the angles, the translation and the angles
    alpha = x0[0] 
    beta = x0[1] 
    gamma = x0[2]    
    t_body = x0[3:6]
    theta_el = x0[6]
    phi_lr = x0[7]
        
    # and the positions have to be separate vectors for some tuple/scipy bullshit reason
    #TODO cehck if this is really true, could be more momory efficient
    positions = np.vstack((posx,posy,posz)).T
    
    # get the coordinates c of the mouse body in it's own reference frame 
    R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,r_mid,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(alpha,beta,gamma,theta_el,phi_lr)

    # Now, calculate the distance vectors from the origin of the hip, mid and head 
    
    p_hip = (positions - ( R_body @ c_hip + t_body) ).T
    p_mid = (positions - ( c_mid + t_body) ).T
    p_nose = (positions - ( R_nose @ c_nose + t_body) ).T
    
    # and calculate the distances
    
    delta_mid = np.abs( jit_norm(p_mid) - r_mid )   

    delta_hip = np.abs( 1 - 1 / np.sqrt(np.sum(p_hip*(Q_hip @ p_hip),0))  ) *  jit_norm(p_hip) 

    delta_nose = np.abs( 1 - 1 / np.sqrt(np.sum(p_nose*(Q_nose @ p_nose),0))  ) *  jit_norm(p_nose) 

#    distances = np.array(min(list(delta_mid),list(delta_hip),list(delta_nose)) )
    
    distances = np.vstack((delta_mid,delta_hip,delta_nose))

    return distances

#distance_to_body_3_piece(x0,posx,posy,posz)

#%%


def guess_x0():    
    # 40 cm, maybe?
    xy_range = 0.3*np.array([-1,1])
    z_range = 0.15*np.array([0,1])
    
    # THE way the mouse model is defined, 
    # alpha is rotation around x, left-right rotation of the mouse, allow +/- 45 deg?
    # beta is rotation around y, elevation (nose-up) rotation of the mouse, allow +/- 90 deg?
    # gamma is rotation around z is head orientation, all degrees allowed
    alpha_range = np.pi/4 * np.array([-1,1])
    beta_range = np.pi/2 * np.array([-.5,1])
    gamma_range = np.pi * np.array([-1,1])
    
    # pi/6, 30 deg?
    theta_range = np.pi/6 * np.array([-1,1])
    phi_range = np.pi/6 * np.array([-1,1])
    
    # so the boundaries are:
    bounds = np.vstack((alpha_range,beta_range,gamma_range,xy_range,xy_range,z_range,theta_range,phi_range))
    bounds_low = bounds[:,0]
    bounds_high = bounds[:,1]
    
    # Also define the guess, just in the middle
    x0 = np.mean(bounds,axis=1)
    
    return x0,bounds_low,bounds_high


def search_ligth_pr_step():    
    # allow one cm steps?
    xy_range = 0.01*np.array([-1,1])
    z_range = 0.01*np.array([-1,1])
    
    # THE way the mouse model is defined, 
    # alpha is rotation around x, left-right rotation of the mouse, allow +/- 45 deg?
    # beta is rotation around y, elevation (nose-up) rotation of the mouse, allow +/- 90 deg?
    # gamma is rotation around z is head orientation, all degrees allowed
    alpha_range = np.pi/15 * np.array([-1,1])
    beta_range = np.pi/15 * np.array([-1,1])
    gamma_range = np.pi/15 * np.array([-1,1])
    
    # pi/6, 30 deg?
    theta_range = np.pi/15 * np.array([-1,1])
    phi_range = np.pi/15 * np.array([-1,1])
    
    # so the boundaries are:
    bounds = np.vstack((alpha_range,beta_range,gamma_range,xy_range,xy_range,z_range,theta_range,phi_range))
    bounds_low = bounds[:,0]
    bounds_high = bounds[:,1]
    
    return bounds_low,bounds_high




#%% HELLO YES IM A FUNCTION YES HELLO
'''START by loading a frame and clicking the nose, then head'''

def show_cloud_and_ask_for_click(file_path,start_frame):
    full_positions,what_keys = load_positions_from_h5(file_path,start_frame,with_rgb=True)
    
    #fig,ax = open_3d()
    #ax.scatter(full_positions[:,0],full_positions[:,1],full_positions[:,2],c='red')
    
    plt.figure()
    plt.scatter(full_positions[:,0],full_positions[:,1],c=full_positions[:,3:]/255,s=1)
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('click tail and head of mouse!')
    click_points = plt.ginput(2)
    plt.show()
    
    
    # % %  AND now calculate the starting values
    c_mid_est = np.mean(np.array(click_points),axis = 0)
    v_ref_est = np.diff(np.array(click_points),axis = 0)
    def add_vec_from_point(c_mid_est,v_ref_est):
        data = np.vstack((c_mid_est,c_mid_est+v_ref_est))
        plt.plot(data[:,0],data[:,1],c='red')
        plt.plot(data[0,0],data[0,1],c='red',marker='o')
    
    
    plt.figure()
    plt.scatter(full_positions[:,0],full_positions[:,1],c=full_positions[:,3:]/255,s=1)
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    add_vec_from_point(c_mid_est,v_ref_est)
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('estimated center and direction')
    plt.show()
    
    # 2 cm is a pretty good guess for z, so a reasonable x0 (straigt spine, no rotation around ) is:
    # start by 
    x0,bounds_low,bounds_high = guess_x0()  
    
    # translation vector, which moves the center of the mouse body
    x0[3] = c_mid_est[0]
    x0[4] = c_mid_est[1]
    x0[5] = 0.018 # guess the z as around 2 cm
    
    # guess the rotation as the body model:
    # - alpha is around x axis, none, I guess
    # - beta is around y axis, so elevation
    # - gamma is around z, so in the xy plane, the left-right angle of the mouse wrt. e^hat_x
    angle_with_x = angle_between(np.array([1.,0]),v_ref_est.ravel())
    x0[2] = angle_with_x
    
    ''' Returns the estimated starting values'''
    return x0


#%% define function to plot positions and corresponding body model

def plot_positions_with_body(file_path,start_frame,x0):
    positions,what_keys = load_positions_from_h5(file_path,start_frame,with_rgb=True)

    x,y,z = body_test_points()  
#    dd = distance_to_body_surface4(x0,x,y,z)
    dd = np.min( distance_to_body_3_piece(x0,x,y,z), 0 )

    fig, ax = open_3d()
    cut = 0.002
    ax.scatter(x[dd<cut],y[dd<cut],z[dd<cut],c=dd[dd<cut])
    
    if positions.shape[1] == 6:
#        ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=positions[:,3:6]/255,marker='.')    
        ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=positions[:,3:6]/255,alpha =0.2,marker='.')    

    else:
        ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='red',marker='.')    
    
    close_3d(ax,np.vstack((x,y,z)).T)    

def plot_positions_with_body_no_file(positions,x0):

    x,y,z = body_test_points()  
#    dd = distance_to_body_surface4(x0,x,y,z)
    dd = np.min( distance_to_body_3_piece(x0,x,y,z), 0 )

    fig, ax = open_3d()
    cut = 0.002
    ax.scatter(x[dd<cut],y[dd<cut],z[dd<cut],c=dd[dd<cut])
    
    if positions.shape[1] == 6:
#        ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=positions[:,3:6]/255,marker='.')    
        ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=positions[:,3:6]/255,alpha =0.2,marker='.')    

    else:
        ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='red',marker='.')    
    
    close_3d(ax,np.vstack((x,y,z)).T)    



#%% TEST THE SPEED
#def ft():
#    #fit_nongit(positions,x0)
#    #run_naive_fit(positions,x0)
##    distance_to_body_surface4(x0,positions[:,0],positions[:,1],positions[:,2])
#    np.min( distance_to_body_3_piece(x0,positions[:,0],positions[:,1],positions[:,2]), 0 )
#
#import timeit
#t = timeit.Timer(ft)
#t.timeit(1000)


#%% FLOW start with the staring values
# load first frame
file_path = '/home/chrelli/Documents/EXAMPLE H5/mus3_unfiltered_aligned.h5'
#file_path = '/media/chrelli/Data0/mus3_unfiltered_aligned.h5'
start_frame = 147

x0 =  show_cloud_and_ask_for_click(file_path,start_frame)

# plot to see if the estimate was pretty ok
plot_positions_with_body(file_path,start_frame,x0)



#%% NOW, get to the 
def wrapped_distance(x0,x,y,z):
    return np.min( distance_to_body_3_piece(x0,x,y,z), 0 )

def dirty_downsample(positions, delta_space = 0.004):
    lines, count = np.unique(np.round(positions[:,0:3]/delta_space), axis=0, return_counts = True)
    return delta_space * lines, count 

def dirty_cutdown(positions, x0, delta_space = 0.004,space_cut = 0.01):
    positions, weights = dirty_downsample(positions, delta_space)
    dd = wrapped_distance(x0,positions[:,0],positions[:,1],positions[:,2])    
    logic = dd < space_cut
    return positions[logic], weights[logic]

def fit_by_ls(positions,x0,bounds_low,bounds_high):
    from scipy.optimize import least_squares
    """
    positions is just Mx3, x y z values
    """
    res_robust = least_squares(wrapped_distance, x0,ftol = 1e-4, loss='soft_l1',
                               f_scale=0.01,bounds = (bounds_low,bounds_high),args=(positions[:,0],positions[:,1],positions[:,2]))
    

    return res_robust.x,res_robust


def wrap_for_curve(positions,x0):
    return np.min( distance_to_body_3_piece(x0,x,y,z), 0 )




def fit_by_curve(positions,weights,x0,bounds_low,bounds_high):
    from scipy.optimize import curve_fit
    '''
    here's a discussion of how to translate between weights and sigma
    sigma = sqrt(n)/n where n is the number of points in the bin
    '''
    sigma = np.sqrt(weights)/weights
    
    popt, pcov = curve_fit(wrapped_distance)
    
    return popt, pcov
    
    
#%% FOR GENERATING MOVIES

def video_plot(positions,x0,file_path,selected_frame,list_of_centers,list_of_noses):
    # get the body test model
    x,y,z = body_test_points()  
    dd = np.min( distance_to_body_3_piece(x0,x,y,z), 0 )
    # open the figure
    fig, ax = open_3d()
    cut = 0.002
    # plot the body model
#    ax.scatter(x[dd<cut],y[dd<cut],z[dd<cut],c=dd[dd<cut])
    
    # plot the cut down points, red
#    ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='red',marker='.')    
    
    
    positions,what_keys = load_positions_from_h5(file_path,selected_frame,True)
    ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=positions[:,3:6]/255,alpha =0.1,marker='.')           
    
    
    list_of_centers = list_of_centers[~np.isnan(list_of_centers[:,0]),: ]
    list_of_noses = list_of_noses[~np.isnan(list_of_noses[:,0]),: ]

    n_trail = 20
    if list_of_centers.shape[0] < n_trail:
        
        ax.plot(list_of_centers[:,0],list_of_centers[:,1],list_of_centers[:,2],c='hotpink',marker='o',markersize = 2)
        ax.plot(list_of_noses[:,0],list_of_noses[:,1],list_of_noses[:,2],c='lightgreen',marker='o',markersize = 2)
        
    else:

        ax.plot(list_of_centers[-n_trail:,0],list_of_centers[-n_trail:,1],list_of_centers[-n_trail:,2],c='hotpink',marker='o',markersize = 2)
        ax.plot(list_of_noses[-n_trail:,0],list_of_noses[-n_trail:,1],list_of_noses[-n_trail:,2],c='lightgreen',marker='o',markersize = 2)
        

    ax.scatter(list_of_centers[-1,0],list_of_centers[-1,1],list_of_centers[-1,2],c='hotpink',marker='o',s=50)
    ax.scatter(list_of_noses[-1,0],list_of_noses[-1,1],list_of_noses[-1,2],c='lightgreen',marker='o',s=50)


    close_3d(ax,np.vstack((x,y,z)).T)    
    
    ax.view_init(30, 40) 



    fig.canvas.draw()


    
def add_sphere_to_ax(ax,cc,r):
    """
    always works, because of symmetry
    """
    x0, y0, z0 = cc[0],cc[1],cc[2]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)*r
    y=np.sin(u)*np.sin(v)*r
    z=np.cos(v)*r
    x = x + x0
    y = y + y0
    z = z + z0
    ax.plot_wireframe(x, y, z, color="r")

def add_ellipsoid_to_ax(ax,cc,r_major,r_minor,R_here,t_body):
    """
    needs to have another reference point (here, the mid of the mouse) to get the right directionality
    """   
    x0, y0, z0 = cc[0],cc[1],cc[2]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    
    x=np.cos(u)*r_major
    y=np.sin(u)*np.sin(v)*r_minor
    z=np.sin(u)*np.cos(v)*r_minor
    
    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))
        
    posi_rotated = ((R_here @ (posi.T + cc).T ).T + t_body).T
    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]
    

    
    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )
#    
#    x = x + x0 + t_body[0]
#    y = y + y0 + t_body[1]
#    z = z + z0 + t_body[2]    
#    
    
    ax.scatter(x,y,z,c='red')    
    
    ax.plot_wireframe(x, y, z, color="r")   




    
              
def grab_current_fig(folder_name,selected_frame,fit):
    
    fig = plt.gcf()      
    # get the data as an array 
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    data = np.flip(data,2).copy()
    #data = np.flip(data,2)

    frame_height,frame_width,_ = data.shape
        
    plt.close()


    cv2.putText(data, 'h5 frame: '+str(selected_frame)+', Nf: '+str(fit.nfev)+', Nj: '+str(fit.nfev)+', optimality: '+str(round(fit.optimality,8))+' m', 
                (0, round(0.1*frame_height)), cv2.FONT_HERSHEY_SIMPLEX, 
                (0.001*frame_height), [1,10,10],2)

    cv2.imwrite(folder_name+'/selected_frame'+str(selected_frame)+'.png',data)

@njit
def hard_limits():
    ''' defines the absolute hard limits on the values
    The sequence of variables is
    alpha, beta, gamma, t_body, theta, phi
    '''
    low = np.array([-np.pi/400, -1.2*np.pi, -np.inf, -.3, -.3, 0.01, -np.pi/10, -np.pi/3])
    high = np.array([np.pi/400, np.pi/6, np.inf, .3, .3, .1, .6*np.pi, np.pi/3])
    return low, high

@njit
def searchlight():
    ''' 
    takes a fitted value of x_fit and generates a search space for the next iteration
    The sequence of 
    '''
    
    # allow one cm steps?
    xy_range = 0.03*np.array([-1,1])
    z_range = 0.03*np.array([-1,1])
    
    # THE way the mouse model is defined, 
    # alpha is rotation around x, left-right rotation of the mouse, allow +/- 45 deg?
    # beta is rotation around y, elevation (nose-up) rotation of the mouse, allow +/- 90 deg?
    # gamma is rotation around z is head orientation, all degrees allowed
    alpha_range = np.pi/8 * np.array([-1,1])
    beta_range = np.pi/6 * np.array([-1,1])
    gamma_range = np.pi/8 * np.array([-1,1])
    
    # pi/6, 30 deg?
    theta_range = np.pi/8* np.array([-1,1])
    phi_range = np.pi/8 * np.array([-1,1])
    
    # so the boundaries are:
    bounds = np.vstack((alpha_range,beta_range,gamma_range,xy_range,xy_range,z_range,theta_range,phi_range))
    bounds_low = bounds[:,0]
    bounds_high = bounds[:,1]
        
    return bounds_low, bounds_high


@jit
def test_space(x_fit):
    
    hard_lo, hard_hi = hard_limits()
    lo,hi = searchlight()
    # add the position of last best fit    
    lo += x_fit
    hi += x_fit

    lo = np.max((lo,hard_lo),0)
    hi = np.min((hi,hard_hi),0)
    
    return lo,hi


def floor_points(delta_space = .004):
    """
    Returns a grid of points, fine in xy, bit rougher in z, for plotting
    """
    bond = .25
    x_ = np.arange(-bond,bond,delta_space)
    y_ = np.arange(-bond,bond,delta_space)
    z_ = 0
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    x=x.ravel()
    y=y.ravel()
    z=z.ravel()
    return np.vstack((x,y,z)).T

def add_floor_to_positions(positions,delta_space = .004):
    flp = floor_points(delta_space)
    return np.vstack((positions[:,0:3],flp))
    

#%% MAIN

# loop over frames
N_to_track = 1480
#N_to_track = 1
x_fit = x0
#t_hist = np.array(x_fit[3:6])

folder_name = 'hmm11'

if not os.path.exists(folder_name):
    os.mkdir(folder_name)

fitfile_path = folder_name+'/tracking_new.csv'
fitfile = open(fitfile_path,'w')
writer = csv.writer(fitfile, delimiter=',')

_,bounds_low,bounds_high = guess_x0()

list_of_centers = np.empty((N_to_track,3))*np.nan
list_of_noses = np.empty((N_to_track,3))*np.nan

cnt = 0

for selected_frame in tqdm(range(start_frame,start_frame+N_to_track)):
#    print('doing frame '+str(selected_frame)+' of '+str(start_frame+N_to_track)+'...')
    # load frame
    positions,what_keys = load_positions_from_h5(file_path,selected_frame,True)

    #TODO not terribly efficient
#    positions = add_floor_to_positions(positions,delta_space = .004)
    
    # cut down positions by rounding, uniqueness and save the weights
    positions, weights = dirty_cutdown(positions, x_fit, delta_space = 0.004,space_cut = 0.02)
    

    # GET the search ligt boundaries
    
    lo,hi = test_space(x_fit)

    # NOW fit using scipy optimize???
    x_fit,fit = fit_by_ls(positions,x_fit,lo,hi)
#    x_fit,fit = fit_by_curve(positions,x_fit,bounds_low,bounds_high)
    
    writer.writerow(np.hstack((selected_frame,x_fit)))
    
    if False:
        # save for plotting
        list_of_centers[cnt,:] = x_fit[3:6]
            
        def nose_coord(x_fit):
            # x0 has the parameters of the function, need to unpack the angles, the translation and the angles
            alpha = x_fit[0] 
            beta = x_fit[1] 
            gamma = x_fit[2]    
            t_body = x_fit[3:6]
            theta_el = x_fit[6]
            phi_lr = x_fit[7]
            
            R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,r_mid,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(alpha,beta,gamma,theta_el,phi_lr)
            
            body_scale = 0.9
            a_nose = 0.03 * body_scale
            b_nose = 0.013 * body_scale    
            d_nose = 0.9* r_mid #same radius as the mid
            
            coord_nose = ( R_nose @ np.array([d_nose+a_nose,0,0]) ) + t_body
            return coord_nose
            
        list_of_noses[cnt,:] = nose_coord(x_fit)
        
        cnt += 1    

        #plt.ion()
        plt.ioff()
    
        # if selected_frame> 0.9*start_frame+N_to_track:
        video_plot(positions,x_fit,file_path,selected_frame,list_of_centers,list_of_noses)
        grab_current_fig(folder_name,selected_frame,fit)
    
    if selected_frame > 770:
        # save for plotting
        list_of_centers[cnt,:] = x_fit[3:6]
            
        def nose_coord(x_fit):
            # x0 has the parameters of the function, need to unpack the angles, the translation and the angles
            alpha = x_fit[0] 
            beta = x_fit[1] 
            gamma = x_fit[2]    
            t_body = x_fit[3:6]
            theta_el = x_fit[6]
            phi_lr = x_fit[7]
            
            R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,r_mid,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(alpha,beta,gamma,theta_el,phi_lr)
            
            
            
            
            body_scale = 0.9
            a_nose = 0.03 * body_scale
            b_nose = 0.013 * body_scale    
            d_nose = 0.9* r_mid #same radius as the mid
            
            coord_nose = ( R_nose @ np.array([d_nose+a_nose,0,0]) ) + t_body
            return coord_nose
            
        list_of_noses[cnt,:] = nose_coord(x_fit)
        
        def wire_model_plot(positions,x_fit,file_path,selected_frame,list_of_centers,list_of_noses):
            # open thie 3d axes
            fig, ax = open_3d()
            # Get the body configuration in the body reference frame
            alpha = x_fit[0] 
            beta = x_fit[1] 
            gamma = x_fit[2]    
            t_body = x_fit[3:6]
            theta_el = x_fit[6]
            phi_lr = x_fit[7]
            
            R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,r_mid,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(alpha,beta,gamma,theta_el,phi_lr)
            
            # plot the cut down points, red
            ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='blue',marker='.')    
            
            # plot the body wire model            
            add_sphere_to_plot(ax,c_mid+t_body,r_mid)
            add_ellipsoid_to_ax(ax,c_nose,a_nose,b_nose,R_nose,t_body)
            add_ellipsoid_to_ax(ax,c_hip,a_hip,b_hip,R_body,t_body)

            
            positions,what_keys = load_positions_from_h5(file_path,selected_frame,True)
            ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=positions[:,3:6]/255,alpha =0.5,marker='.')           
            
            
            list_of_centers = list_of_centers[~np.isnan(list_of_centers[:,0]),: ]
            list_of_noses = list_of_noses[~np.isnan(list_of_noses[:,0]),: ]
        
            n_trail = 20
            if list_of_centers.shape[0] < n_trail:
                
                ax.plot(list_of_centers[:,0],list_of_centers[:,1],list_of_centers[:,2],c='hotpink',marker='o',markersize = 2)
                ax.plot(list_of_noses[:,0],list_of_noses[:,1],list_of_noses[:,2],c='lightgreen',marker='o',markersize = 2)
                
            else:
        
                ax.plot(list_of_centers[-n_trail:,0],list_of_centers[-n_trail:,1],list_of_centers[-n_trail:,2],c='hotpink',marker='o',markersize = 2)
                ax.plot(list_of_noses[-n_trail:,0],list_of_noses[-n_trail:,1],list_of_noses[-n_trail:,2],c='lightgreen',marker='o',markersize = 2)
                
        
            ax.scatter(list_of_centers[-1,0],list_of_centers[-1,1],list_of_centers[-1,2],c='hotpink',marker='o',s=50)
            ax.scatter(list_of_noses[-1,0],list_of_noses[-1,1],list_of_noses[-1,2],c='lightgreen',marker='o',s=50)
        
        
            close_3d(ax,np.vstack((t_body-.04,t_body+.04)))    
            
            ax.view_init(30, 40) 
        
        
        
            fig.canvas.draw()
            
               
        cnt += 1    

        #plt.ion()
        plt.ioff()
    
        # if selected_frame> 0.9*start_frame+N_to_track:
        
        #video_plot(positions,x_fit,file_path,selected_frame,list_of_centers,list_of_noses)
        wire_model_plot(positions,x_fit,file_path,selected_frame,list_of_centers,list_of_noses)
        grab_current_fig(folder_name,selected_frame,fit)
        
    
fitfile.close()


#%% CHECK the fitted values!
if False:
    raw = np.genfromtxt(fitfile_path ,delimiter = ',')
    
    frames = raw[:,0]    
    alpha,beta,gamma = raw[:,1],raw[:,2],raw[:,3]  
    t_body = raw[:,4:7]
    theta_el = raw[:,7]
    phi_lr = raw[:,8]
        

#%% Compile into a video
if ~False:    
        
    def file_list_from_folder(top_folder,image_type):
        # list of files in the folder, specific to images!
        file_list = os.listdir(top_folder)
        # sort the list
        file_list.sort()
        file_logic = np.empty(len(file_list))
        for num,name in enumerate(file_list):
            file_logic[num]=name.startswith(image_type)
        short_list = list(compress(file_list,file_logic))
        return short_list
    
    #top_folder = '/Users/chrelli/git/3d_sandbox/mycetrack/hmm2/'
    top_folder = '/home/chrelli/git/3d_sandbox/mycetrack/hmm11/'
    
    short_list = file_list_from_folder(top_folder,'selec')
    
    # GRRR HAVE TO PROPERLY SORT THAT 
    num = np.empty(len(short_list))
    for i, nam in enumerate(short_list):
        num[i] = nam[14:-4]
    #
    indices = np.argsort(num).astype(int)
    short_list = [ short_list[i] for i in indices]
    
    
    frame = cv2.imread(top_folder + short_list[0])
    
    out = cv2.VideoWriter(top_folder+'output_selec.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, 
                          (frame.shape[0],frame.shape[1]))
    
    # open the video writer
    for this_name in tqdm(short_list):
        frame = cv2.imread(top_folder + this_name)
        # save frame
        out.write(frame)
    
    # close the video again
    out.release()
     
        