#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:17:28 2018

@author: chrelli
"""

#%% TRY out fitting with GPflow and tensorflow

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


#import math

# small utilities
#import csv
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
#from utils.common_utils import *
#from utils.recording_utils import *
#from utils.cloud_utils import *
#from fitting_utils import *

#from merge_and_filter_clouds import filter_and_downsample_cloud

# h5py for acessing data
#import h5py

# ALLSO JIT STUFF

from numba import jit, njit

#%% set this for profiling

top_folder = '/home/chrelli/Documents/EXAMPLE DATA/May2018_one_mouse'

def read_processed_frame(top_folder,frame,voxel = 0.003, n_padding_digits = 8):
    raw = np.load(top_folder+'/npy/frame_'+str(frame).rjust(n_padding_digits,'0')+'.npy')
    #todo make column order and split
    positions = raw[:,0:3]*voxel
    weights = raw[:,3]
    return positions,weights

#%%

from utils.fitting_utils import *
#from utils.fitting_utils_non_jit import *


def color3d(positions,colors=None):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    correctX,correctY,correctZ = positions[:,0],positions[:,1],positions[:,2]
    
    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        if len(colors.shape)> 1:
            ax.scatter(correctX, correctY, correctZ, zdir='z', s=1, c=colors/255.,rasterized=True)
        else:
            ax.scatter(correctX, correctY, correctZ, zdir='z', s=1, c=colors/np.max(colors),rasterized=True)
    else:
        ax.scatter(correctX, correctY, correctZ, zdir='z', s=1, c='b',rasterized=True)
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    ax.set_title(str(positions.shape[0])+' points',fontsize=16)
    
    X,Y,Z = correctX,correctY,correctZ
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() 
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()    
    w,h = 570,800
    plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)
        
positions,weights = read_processed_frame(top_folder,15000)
color3d(positions,weights)



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
    
    cut = 0.02
    minimum_dist[minimum_dist>cut] = cut
    
    # return the minimum distance
    return minimum_dist

@njit
def wrapped_loss(x0,posx,posy,posz):
    return np.sum(distance_to_1_mouse(x0,posx,posy,posz))

@njit
def wrapped_loss2(x0,posx,posy,posz):
    return np.sum(distance_to_1_mouse(x0,posx,posy,posz)**2)

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
s=.9
# plot the mouse body
plot_mouse_body(beta,gamma,s,theta,phi,t_body,positions = positions,weights = None)


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
    s_range = np.array([0,.5]) #[a.u.]
    
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



hard_lo,hard_hi = hard_limits()



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



box_lo,box_hi = box_bounds()
box_lo,box_hi = 2*box_lo,2*box_hi

#%% START by just fitting one frame, to get starting values!
"""
  _   _   _   _   _   _   _   _     _   _   _   _   _     _   _   _   _   _  
 / \ / \ / \ / \ / \ / \ / \ / \   / \ / \ / \ / \ / \   / \ / \ / \ / \ / \ 
( M | i | n | i | m | i | z | e ) ( s | t | a | r | t ) ( f | r | a | m | e )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/   \_/ \_/ \_/ \_/ \_/   \_/ \_/ \_/ \_/ \_/ 

"""
    
x0_guess = np.hstack((beta,gamma,s,theta,phi,t_body))

from scipy.optimize import minimize
#from scipy.optimize import Bounds


opt = ({'maxiter': 1000})

bounds = [(a,b) for a,b in zip(hard_lo,hard_hi)]

res = minimize(wrapped_loss2, x0_guess, 
               args=(positions[:,0],positions[:,1],positions[:,2]), 
               bounds = bounds,
               method = 'SLSQP',options = opt)   

x_fit = res.x
optimality = res.fun


plot_mouse_body(x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],positions = positions,weights = None)

x_fit_frame = x_fit


#%% try GPFlow in a box around!
import gpflow

#%% Generate samples in the bounding box!

n_points = 1000 #% in each dimension

#TODO very inefficient
n_dim = len(box_lo)
grid = np.empty((n_points,n_dim))
for i in range(8):
    grid[:,i] = np.random.uniform(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_points)



#%% calculate the loss as a function of the test points

X = grid
Y = np.empty((n_points,1))
now = time.time()
for i in range(n_points):
    Y[i]= wrapped_loss2(X[i,:],positions[:,0],positions[:,1],positions[:,2])
print(time.time()-now)    
plt.figure()
plt.plot(Y)    

#%% Load fitting from three frames!
X = grid
Y = np.empty((n_points,1))
now = time.time()
for i in range(n_points):
    this_frame = 15000#+round(np.random.uniform(-1,1))
    positions,weights = read_processed_frame(top_folder,this_frame)
    print(this_frame)
    Y[i]= wrapped_loss2(X[i,:],positions[:,0],positions[:,1],positions[:,2])
print(time.time()-now)    
plt.figure()
plt.plot(Y) 



#%%
positions,weights = read_processed_frame(top_folder,15000)
color3d(positions)

#%%

plt.figure()
plt.scatter(grid[:,6],grid[:,7],c=Y[:,0]/np.max(Y))
plt.show()

#%%
# make a gaussian kernel, with 8 dimensions 
#k = gpflow.kernels.Matern52(1, lengthscales=0.3)
#k = gpflow.kernels.Matern52(n_dim)
#k = gpflow.kernels.RBF(n_dim)

k = gpflow.kernels.RBF(n_dim,lengthscales = box_hi)
#k = gpflow.kernels.Matern52(n_dim,lengthscales = box_hi)

m = gpflow.models.GPR(X,Y,kern=k)
# train it!
gpflow.train.ScipyOptimizer().minimize(m)

#%% Make a function, which plots the cross slice of all dimensions!

def plot_cross(m,x_fit,box_lo,box_hi,n_eval = 20):
    # n_eval is the grid for plotting
    #plt.figure()
    
    n_dim = len(box_lo)
    f, axes = plt.subplots(n_dim, n_dim, figsize=(12, 12))
    
    dim_names = ['beta','gamma','s','theta','phi','t_x','t_y','t_z']
    
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i,j]

            if i==(n_dim-1):
                ax.set_xlabel(dim_names[j])
            if j==0:
#                ax.set_ylabel(dim_names[j])
                ax.set_ylabel(dim_names[i])

            if i==j:
                ax.set_axis_off()
                continue
            else:
                # generate test grid!
                test_i = np.linspace(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_eval)
                test_j = np.linspace(x_fit[j]+box_lo[j],x_fit[j]+box_hi[j],n_eval)
                ii,jj = np.meshgrid(test_i,test_j)
                test_i,test_j = ii.ravel(),jj.ravel()
                
                X_test = np.tile(x_fit,(n_eval**2,1))
                X_test[:,i] = test_i
                X_test[:,j] = test_j
                print('now')
                print(i)
                print(j)                                                
                predMean, predVar = m.predict_y(X_test)
                
#                predMean=np.log(predMean)
                ax.scatter(test_i,test_j,c=predMean.ravel()/np.abs(np.max(predMean)))
                
#                abe= ax.axis('tight')
                ax.set_xlim([x_fit[i]+box_lo[i],x_fit[i]+box_hi[i]])
                ax.set_ylim([x_fit[j]+box_lo[j],x_fit[j]+box_hi[j]])


plot_cross(m,x_fit,box_lo,box_hi)


#%% Another way is to just plot the single value fit of one !


def plot_line(m,x_fit,box_lo,box_hi,n_eval = 20):
    # n_eval is the grid for plotting
    #plt.figure()
    
    n_dim = len(box_lo)
    f, axes = plt.subplots(int(n_dim/2), 2, figsize=(12, 12))
    axes = axes.ravel()
    
    dim_names = ['beta','gamma','s','theta','phi','t_x','t_y','t_z']
    
    for i in range(n_dim):
        
        ax = axes[i]
        
        ax.set_xlabel(dim_names[i])
    
        test_i = np.linspace(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_eval)

        X_test = np.tile(x_fit,(n_eval,1))
        X_test[:,i] = test_i
        
        mean, var = m.predict_y(X_test)
         
        ax.plot(test_i, mean, 'C0', lw=2)
        
        ax.fill_between(test_i,
                     mean.ravel() - 2*np.sqrt(var.ravel()),
                     mean.ravel() + 2*np.sqrt(var.ravel()),
                     color='C0', alpha=0.2)
        
        ax.set_xlabel(dim_names[i])
        ax.set_ylabel('slice of loss function')
    
plot_line(m,x_fit,box_lo,box_hi)

#%%

k_space = gpflow.kernels.RBF(n_dim,lengthscales = box_hi,active_dims=[0,1,2,3,4,5,6,7])
k_time = gpflow.kernels.RBF(1,active_dims = [8])
#k_time = gpflow.kernels.Linear(1,active_dims = [8])

k = k_space*k_time
#k = gpflow.kernels.Matern52(n_dim,lengthscales = box_hi)
# Now, we generate time training data!

n_points = 400 #% in each dimension
n_time = 5
#TODO very inefficient
n_dim = len(box_lo)
X= np.empty((n_points*n_time,n_dim))
for i in range(8):
    X[:,i] = np.random.uniform(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_points*n_time)

# add the time!
Xt = [None]
for i in range(n_time):
    Xt.append([i]*n_points)
Xt = np.concatenate(Xt[1:])-int(np.floor(n_time/2))
# stack space and time
X = np.column_stack((X,Xt))

# now, evaluate the loss
Y = np.empty((n_points*n_time,1))
for i in range(n_points*n_time):
    this_frame = 15000#+round(np.random.uniform(-1,1))
    positions,weights = read_processed_frame(top_folder,this_frame+int(X[i,8]))
    Y[i]= wrapped_loss2(X[i,0:8],positions[:,0],positions[:,1],positions[:,2])

plt.figure()
plt.plot(Y) 


m = gpflow.models.GPR(X,Y,kern=k)
# train it!
gpflow.train.ScipyOptimizer().minimize(m)


#%% NOW, plot across time for everything!!


def plot_time(m,x_fit,box_lo,box_hi,n_eval = 20):
    # n_eval is the grid for plotting
    #plt.figure()
    
    n_dim = len(box_lo)
    f, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    dim_names = ['beta','gamma','s','theta','phi','t_x','t_y','t_z']
    
    for i in range(n_dim):
        ax = axes[i]
        test_i = np.linspace(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_eval)
        test_j = np.linspace(-2,2,5)
        ii,jj = np.meshgrid(test_i,test_j)
        test_i,test_j = ii.ravel(),jj.ravel()
        
        X_test = np.tile(x_fit,(n_eval*5,1))
        X_test[:,i] = test_i
        X_test = np.column_stack((X_test,test_j))
        
        predMean, predVar = m.predict_y(X_test)
 
        ax.scatter(test_i,test_j,c=predMean.ravel()/np.abs(np.max(predMean)))
                
#                abe= ax.axis('tight')
#        ax.set_xlim([x_fit[i]+box_lo[i],x_fit[i]+box_hi[i]])
#        ax.set_ylim([-2,2])



plot_time(m,x_fit,box_lo,box_hi)

#%%

def plot_cross2(m,x_fit,box_lo,box_hi,n_eval = 20):
    # n_eval is the grid for plotting
    #plt.figure()
    
    n_dim = len(box_lo)
    f, axes = plt.subplots(n_dim, n_dim, figsize=(12, 12))
    
    dim_names = ['beta','gamma','s','theta','phi','t_x','t_y','t_z']
    
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i,j]

            if i==(n_dim-1):
                ax.set_xlabel(dim_names[j])
            if j==0:
#                ax.set_ylabel(dim_names[j])
                ax.set_ylabel(dim_names[i])

            if i==j:
                ax.set_axis_off()
                continue
            else:
                # generate test grid!
                test_i = np.linspace(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_eval)
                test_j = np.linspace(x_fit[j]+box_lo[j],x_fit[j]+box_hi[j],n_eval)
                ii,jj = np.meshgrid(test_i,test_j)
                test_i,test_j = ii.ravel(),jj.ravel()
                
                X_test = np.tile(x_fit,(n_eval**2,1))
                X_test[:,i] = test_i
                X_test[:,j] = test_j
                
                X_time = np.zeros(X_test.shape[0])
                
                X_test = np.column_stack((X_test,X_time))
                print('now')
                print(i)
                print(j)                                                
                predMean, predVar = m.predict_y(X_test)
                
#                predMean=np.log(predMean)
                ax.scatter(test_i,test_j,c=predMean.ravel()/np.abs(np.max(predMean)))
                
#                abe= ax.axis('tight')
                ax.set_xlim([x_fit[i]+box_lo[i],x_fit[i]+box_hi[i]])
                ax.set_ylim([x_fit[j]+box_lo[j],x_fit[j]+box_hi[j]])


plot_cross(m,x_fit,box_lo,box_hi)


#%%
def plot_line2(m,x_fit,box_lo,box_hi,n_eval = 20):
    # n_eval is the grid for plotting
    #plt.figure()
    
    n_dim = len(box_lo)
    f, axes = plt.subplots(int(n_dim/2), 2, figsize=(12, 12))
    axes = axes.ravel()
    
    dim_names = ['beta','gamma','s','theta','phi','t_x','t_y','t_z']
    
    for i in range(n_dim):
        
        ax = axes[i]
        
        ax.set_xlabel(dim_names[i])
    
        test_i = np.linspace(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_eval)

        X_test = np.tile(x_fit,(n_eval,1))
        X_test[:,i] = test_i
        
        X_time = np.zeros(X_test.shape[0])
               
        X_test = np.column_stack((X_test,X_time))
        
        mean, var = m.predict_y(X_test)
         
        ax.plot(test_i, mean, 'C0', lw=2)
        
        ax.fill_between(test_i,
                     mean.ravel() - 2*np.sqrt(var.ravel()),
                     mean.ravel() + 2*np.sqrt(var.ravel()),
                     color='C0', alpha=0.2)
        
        ax.set_xlabel(dim_names[i])
        ax.set_ylabel('slice of loss function')
    
plot_line(m,x_fit,box_lo,box_hi)
#%% TRY doint the same thing with gpy now!

import GPy

n_dim = len(box_lo)
#kernel = GPy.kern.RBF(input_dim = n_dim)
#kernel = GPy.kern.RBF(input_dim=n_dim,lengthscale=box_hi,ARD=True)
kernel = GPy.kern.RatQuad(input_dim=n_dim,lengthscale=box_hi,ARD=True)


model = GPy.models.GPRegression(X,Y,kernel=kernel)
model.optimize(messages=True)
#%% NOw, do a gpy version


def plot_cross_GPy(model,x_fit,box_lo,box_hi,n_eval = 20):
    # n_eval is the grid for plotting
    #plt.figure()
    
    n_dim = len(box_lo)
    f, axes = plt.subplots(n_dim, n_dim, figsize=(12, 12))
    
    dim_names = ['beta','gamma','s','theta','phi','t_x','t_y','t_z']
    
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i,j]

            if i==(n_dim-1):
                ax.set_xlabel(dim_names[j])
            if j==0:
#                ax.set_ylabel(dim_names[j])
                ax.set_ylabel(dim_names[i])

            if i==j:
                ax.set_axis_off()
                continue
            else:
                # generate test grid!
                test_i = np.linspace(x_fit[i]+box_lo[i],x_fit[i]+box_hi[i],n_eval)
                test_j = np.linspace(x_fit[j]+box_lo[j],x_fit[j]+box_hi[j],n_eval)
                ii,jj = np.meshgrid(test_i,test_j)
                test_i,test_j = ii.ravel(),jj.ravel()
                
                X_test = np.tile(x_fit,(n_eval**2,1))
                X_test[:,i] = test_i
                X_test[:,j] = test_j
                print('now')
                print(i)
                print(j)                                                
                predMean, predVar = model.predict_noiseless(X_test)
                
#                predMean=np.log(predMean)
                ax.scatter(test_i,test_j,c=predMean.ravel()/np.abs(np.max(predMean)))
                
#                abe= ax.axis('tight')
                ax.set_xlim([x_fit[i]+box_lo[i],x_fit[i]+box_hi[i]])
                ax.set_ylim([x_fit[j]+box_lo[j],x_fit[j]+box_hi[j]])


plot_cross_GPy(model,x_fit,box_lo,box_hi)



#%%
plt.figure()
#plt.scatter(test_i,test_j)
plt.scatter(test_i,test_j,c=predMean.ravel()/np.max(predMean))
ax = plt.gca()
ax.axis('tight')

#%% plot a 2d slice of the model!

nx=5
ny=6
Y_p = np.empty(Y.shape)
for i in range(n_points):
    x_new = x_fit.copy()
    x_new[nx] = X[i,nx]
    x_new[ny] = X[i,ny]
    Y_p[i],_ = m.predict_f(x_new.reshape(-1,1).T)

#%%

plt.figure()
plt.scatter(X[:,nx],X[:,ny],c=Y_p[:,0]/np.max(Y_p))
plt.show()    
    
    


#%%


m.predict_y(x_fit.reshape(-1,1))

#%%
ax = fig.add_subplot(132, title='pcolormesh: actual edges',
        aspect='equal')

X, Y = np.meshgrid(xedges, yedges)

ax.pcolormesh(X, Y, H)





#%%

N = 12
X = np.random.rand(N,1)
Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3
plt.plot(X, Y, 'kx', mew=2)
plt.show()




#%%
k = gpflow.kernels.Matern52(1, lengthscales=0.3)
m = gpflow.models.GPR(X, Y, kern=k)
m.likelihood.variance = 0.01

#%%
def plot(m):
    xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
    plt.xlim(-0.1, 1.1)
    
plot(m)


