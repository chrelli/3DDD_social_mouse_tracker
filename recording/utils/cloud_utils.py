#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:49:18 2018

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
#import cv2

# for recording and connecting to the intel realsense librar
#import pyrealsense as pyrs

#import multiprocessing
from multiprocessing import Process

# for cloud handling
#from pyntcloud import PyntCloud

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *

#%% HANDY FUNCTIONS


def load_calib_trace(which_device,top_folder):
    raw = np.genfromtxt(top_folder+'/central_point_'+str(which_device)+'.csv',delimiter=',')
    positions = raw[:,1:4]
    return positions

def load_transformation(which_device,top_folder):
    # try reading, if it doesn't work fall back on the most recent calibration which _HAS_one
    if os.path.exists(top_folder+'/transformation_'+str(which_device)+'.csv'):
        transform = np.genfromtxt(top_folder+'/transformation_'+str(which_device)+'.csv',delimiter=',')
        t = np.genfromtxt(top_folder+'/translation_'+str(which_device)+'.csv',delimiter=',')
        R = np.genfromtxt(top_folder+'/rotation_'+str(which_device)+'.csv',delimiter=',')
    # actually, no - too dangerous, change later
    else:
        print('WARNING: Most recent calibration is no good, used another recent one!')
        parent = os.path.split(top_folder)[0]
        # look for all calibration folders in the parent
        folder_list = next(os.walk(parent))[1]
        logic_list = [x[0:11] == 'calibration' for x in folder_list]
                 
        led_list = list(compress(folder_list,logic_list)) 
        led_list.sort(reverse=True)
        # get the last one
        for folder in led_list:
            if os.path.exists(folder+'/transformation_'+str(which_device)+'.csv'):
                transform = np.genfromtxt(folder+'/transformation_'+str(which_device)+'.csv',delimiter=',')
                t = np.genfromtxt(folder+'/translation_'+str(which_device)+'.csv',delimiter=',')
                R = np.genfromtxt(folder+'/rotation_'+str(which_device)+'.csv',delimiter=',')
                break
    
    return R,t,transform





def all_calibration_folders():
    # RETURNS a list of recording folders sorted to newest first!
    # simply look for the most recent LED folder!
    top_folder = '/media/chrelli/Data0'
    # get a list of the folders in that directory
    folder_list = next(os.walk(top_folder))[1]
    logic_list = [x[0:11] == 'calibration' for x in folder_list]
    
    led_list = list(compress(folder_list,logic_list)) 
    led_list.sort(reverse=True)
    # get the last one
    full_folders_0 = ['/media/chrelli/Data0/'+i for i in led_list] #TODO fix this shit!
    full_folders_1 = ['/media/chrelli/Data1/'+i for i in led_list] #TODO fix this shit!

    return full_folders_0,full_folders_1








def apply_rigid_transformation(positions,R,t):
    # takes postions as a Nx3 vector and applies rigid transformation
    # make matrices
    A = np.asmatrix(positions)
    R = np.asmatrix(R)
    t = np.asmatrix(t).T

    # Matrix way:
    n = A.shape[0]
    A2 = np.matmul(R,A.T) + np.tile(t, (1, n))
    
    # print(str(i)+' after transform: '+str(A2.shape))
    # make it an array?
    return np.asarray(A2.T)



#%% cloud handling

#def merge_four_clouds(cloud0,cloud1,cloud2,cloud3):
#    # a function to combine two point clouds into one!
#    color0 = Color("blue")
#    color1 = Color("red")
#    color2 = Color("orange")
#    color3 = Color("lightgreen")
#    
#    rgb0=color0.get_rgb()
#    rgb1=color1.get_rgb()
#    rgb2=color2.get_rgb()
#    rgb3=color3.get_rgb()
#  
#    points0 = cloud0.points
#    points1 = cloud1.points    
#    points2 = cloud2.points
#    points3 = cloud3.points
#
#    colors0 = np.ones((points0.shape[0],3))*rgb0
#    colors1 = np.ones((points1.shape[0],3))*rgb1
#    colors2 = np.ones((points2.shape[0],3))*rgb2
#    colors3 = np.ones((points3.shape[0],3))*rgb3
#    
#    fullcolors = (np.vstack((colors0,colors1,colors2,colors3)) * 255).astype(np.uint8)
#
#    # adcd the points together
#    fullpoints = pd.concat((points0,points1,points2,points3),axis=0)
#    # and add the rgb values
#    fullpoints[['red', 'blue', 'green']] = pd.DataFrame(fullcolors, index=fullpoints.index)
#    # and make a full cloud
#    fullcloud = PyntCloud(fullpoints)
#    # and plot it
#    return fullcloud




#%% custom cloud plotting example

def easy3d(positions):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    correctX,correctY,correctZ = positions[:,0],positions[:,1],positions[:,2]
    
    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c='b',rasterized=True)
    
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    
    X,Y,Z = correctX,correctY,correctZ
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()    
    
    



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
    
    


def weight3d(positions,weights):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    correctX,correctY,correctZ = positions[:,0],positions[:,1],positions[:,2]
    
    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c=weights/np.max(weights),rasterized=True)
    
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    
    X,Y,Z = correctX,correctY,correctZ
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() 
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()    
    
    
    plt.show()


def plot_cloud(cloud):
        
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
#    selecta = (Z>0.40)*(Z<1.2)
    selecta = ~np.isnan(Z)

    if 'red' in cloud.points:
        # get the colors
        C = cloud.points[['red','green','blue']].values
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
    plt.show()


def plot_cloud_raw(cloud):
        
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
    
    
    if 'red' in cloud.points:
        # get the colors
        C = cloud.points[['red','green','blue']].values
        ax.scatter(X, Y, Z, zdir='z', s=1, c=C/255.0,rasterized=True)
    else:
        ax.scatter(X, Y, Z, zdir='z', s=1, c='b',rasterized=True)
    
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    plt.show()
    


def plot_cloud_home(cloud):
        
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
    
    
    if 'red' in cloud.points:
        # get the colors
        C = cloud.points[['red','green','blue']].values
        ax.scatter(X, Y, Z, zdir='z', s=10, c=C/255.0,rasterized=True)
    else:
        ax.scatter(X, Y, Z, zdir='z', s=10, c='b',rasterized=True)
    
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    
    
    fig.tight_layout(pad=0)
    ax.set_axis_off()
    plt.show()
        
    
    
def plot_cloud_eq(cloud):
        
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
    
    
    if 'red' in cloud.points:
        # get the colors
        C = cloud.points[['red','green','blue']].values
        ax.scatter(X, Y, Z, zdir='z', s=1, c=C/255.0,rasterized=True)

    else:
        ax.scatter(X, Y, Z, zdir='z', s=1, c='b',rasterized=True)
    
    ax.set_aspect('equal')
    
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)

    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()    





    
