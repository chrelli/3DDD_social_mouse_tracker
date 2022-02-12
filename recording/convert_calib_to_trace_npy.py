#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:01:29 2018

@author: chrelli

CONVERT THE CALIBRATION into a 3d shape!

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

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *
from utils.cloud_utils import *


#%% Parse some inputs

import argparse

parser = argparse.ArgumentParser(description='Applies HSV/background filtering and converts the tracked calibration data to a calibration trace of estimated xzy coordinates for each frame in the calibration data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--ncams', type=int, default = 4 , choices=[1,2,3,4],
                    help='number of cameras to stream')

parser.add_argument('--estimate', type=str, default = 'ransacfixed' , choices=['closestpoint','fitsphere','ransacsphere','ransacfixed'],
                    help='choose how to estimate the center of the ping pong ball. Default is ransacfixed, slower but worth it. 5-6 it/s right now')

parser.add_argument("--force", help="forces a re-calculation of the trace even if it was already done",
                    action="store_false")


args = parser.parse_args()



#%% check if the folders exist
from utils.reading_utils import most_recent_calibration_folders

top_folder_0,top_folder_1 = most_recent_calibration_folders()

check_folder_if_present(top_folder_0)
check_folder_if_present(top_folder_1)

# DROP a file to disk to indicate the chosen estimation method! (extremely retro)

# this is some weirdo string saving stuff
dat = np.array(args.estimate)
np.savetxt(top_folder_0+'/trace_estimation.txt',dat.reshape(1,),delimiter=" ", fmt="%s")


#%% GENERATE custom pyntcloud class to fit fixed sphere!
import numpy as np
import pandas as pd
from pyntcloud.geometry.models.base import GeometryModel

class SphereFixedRadius(GeometryModel):

    def __init__(self, center=None, radius=None):
        self.center = center
        self.radius = radius

    def from_k_points(self, points):

        # fit numerically to the four points!
        from scipy.optimize import minimize
        r = 0.02 # we know the radius!
        def errorfun(center):
            dist = np.linalg.norm(points-center,axis = 1)
            err = np.sum( (r - dist)**2 ) # minimize the sum of sqaures
            return err

        initial_guess = np.mean(points,axis=0)
        result = minimize(errorfun, initial_guess)

        self.center = result.x
        self.radius = r

    def from_point_cloud(self, points):
        """
        Least Squares fit. numerically
        Parameters
        ----------
        points: (N, 3) ndarray
        """
       # fit numerically to the four points!
        from scipy.optimize import minimize
        r = 0.02 # we know the radius!
        def errorfun(center):
            dist = np.linalg.norm(points-center,axis = 1)
            err = np.sum( (r - dist)**2 ) # minimize the sum of sqaures
            return err

        initial_guess = np.mean(points,axis=0)
        result = minimize(errorfun, initial_guess)

        self.center = result.x
        self.radius = r

    def get_projections(self, points, only_distances=False):
        vectors = points - self.center
        lengths = np.linalg.norm(vectors, axis=1)
        distances = np.abs(lengths - self.radius)
        if only_distances:
            return distances
        scales = self.radius / lengths
        projections = (scales[:, None] * vectors) + self.center
        return distances, projections


from pyntcloud.ransac.models import RansacModel

class RansacSphereFixedRadius(RansacModel, SphereFixedRadius):

    def __init__(self, max_dist=1e-3):
        super().__init__(max_dist=max_dist)
        self.k = 4

    def are_valid(self, k_points):
        # check if points are coplanar
        x = np.ones((4, 4))
        x[:-1, :] = k_points.T
        if np.linalg.det(x) == 0:
            return False
        else:
            return True

# also make a more loose model for ransac fitting

from pyntcloud.ransac.models import RansacModel
from pyntcloud.geometry.models.sphere import Sphere

class RansacSphereLoose(RansacModel, Sphere):

    def __init__(self, max_dist=3e-3):
        super().__init__(max_dist=max_dist)
        self.k = 4

    def are_valid(self, k_points):
        # check if points are coplanar
        x = np.ones((4, 4))
        x[:-1, :] = k_points.T
        if np.linalg.det(x) == 0:
            return False
        else:
            return True


#%% define a few sphere plotting functions


def fix_axes_properly(positions):
    X,Y,Z = positions[:,0], positions[:,1], positions[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


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
    plt.show()


def sphere_plot_BRUTE(positions):

    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from scipy.optimize import minimize
    r = 0.02 # we know the radius!
    def errorfun(center):
        dist = np.linalg.norm(positions-center,axis = 1)
        err = np.sum( (r - dist)**2 ) # minimize their distances!
        return err

    initial_guess = [np.mean(positions[:,0]),np.mean(positions[:,1]),np.mean(positions[:,2])]

    result = minimize(errorfun, initial_guess)
    cc = result.x

    # cc= initial_guess

    x0, y0, z0 = cc[0],cc[1],cc[2]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)*r
    y=np.sin(u)*np.sin(v)*r
    z=np.cos(v)*r
    x = x + x0
    y = y + y0
    z = z + z0

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], zdir='z', s=20, c='b',rasterized=True)
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    plt.show()


def sphere_plot_RANSAC(positions):

    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # import pyntcloud
    from pyntcloud import PyntCloud
    # convert the filtered points to a pyntcloud
    # add to a dataframe
    points = pd.DataFrame(positions,columns=['x', 'y', 'z'])
    # and generate points
    cloud = PyntCloud(points)

    # now use the pyntcloud ransac to fit a sphere!
    from pyntcloud.ransac.fitters import single_fit
    inliers, best_model = single_fit(cloud.points.values, RansacSphereLoose, return_model=True)


    r = best_model.radius
    cc = best_model.center

    # cc= initial_guess

    x0, y0, z0 = cc[0],cc[1],cc[2]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)*r
    y=np.sin(u)*np.sin(v)*r
    z=np.cos(v)*r
    x = x + x0
    y = y + y0
    z = z + z0

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], zdir='z', s=20, c='b',rasterized=True)
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    plt.show()



def sphere_plot_RANSACFIXED(positions):

    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # import pyntcloud
    from pyntcloud import PyntCloud
    # convert the filtered points to a pyntcloud
    # add to a dataframe
    points = pd.DataFrame(positions,columns=['x', 'y', 'z'])
    # and generate points
    cloud = PyntCloud(points)

    # now use the pyntcloud ransac to fit a sphere!
    from pyntcloud.ransac.fitters import single_fit
    inliers, best_model = single_fit(cloud.points.values, RansacSphereFixedRadius, return_model=True)


    r = best_model.radius
    cc = best_model.center

    # cc= initial_guess

    x0, y0, z0 = cc[0],cc[1],cc[2]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)*r
    y=np.sin(u)*np.sin(v)*r
    z=np.cos(v)*r
    x = x + x0
    y = y + y0
    z = z + z0

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], zdir='z', s=20, c='b',rasterized=True)
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)

    #fix_axes_properly(positions)
    plt.show()

# pyntcloud has added a kind of stupid print out function which prints the points out, if they are not coplanar. So I have to add it here and fix


from pyntcloud.ransac.samplers import RandomRansacSampler

def single_fit(points, model, sampler=RandomRansacSampler,
               model_kwargs={},
               sampler_kwargs={},
               max_iterations=100,
               return_model=False,
               n_inliers_to_stop=None):
    """ RANdom SAmple Consensus for fitting model a single model to points.
    points: ndarray
        (N, M) ndarray where N is the number of points and M is the number
        scalar fields associated to each of those points.
        M is usually 3 for representing the x, y, and z coordinates of each point.
    model: Ransac_Model
        Class (NOT INSTANCE!) representing the model that will be fitted to points.
        Check ransac/models for reference.
    sampler: Ransac_Sampler
        Class (NOT INSTANCE!) used to sample points on each iteration.
        Check ransac/samplers for reference.
    model_kwargs: dict, optional
        Default: {}
        Arguments that will be used on model's instantiation.
        Variable according to passed model.
    sampler_kwargs: dict, optional
        Default: {}
        Arguments that will be used on sampler's instantiation.
        Variable according to passed sampler.
    max_iterations: int, optional
        Default: 100
        Maximum number of iterations.
    return_model: bool, optional (default False)
        Whether the best fitted model will be returned or not.
    n_inliers_to_stop: int, optional
        Default None
        If the model fits a number of inliers > n_inliers_to_stop the loop will end.
    """

    model = model(**model_kwargs)
    sampler = sampler(points, model.k, **sampler_kwargs)

    # TODO this is a CHRELLI hack, replace with an actual better ransac at some point!
    best_inliers = []
    n_best_inliers = 0
    if n_inliers_to_stop is None:
        n_inliers_to_stop = len(points)

    for i in range(max_iterations):

        k_points = sampler.get_sample()

        if not model.are_valid(k_points):
            # This is silly, why force it to print every time??
            # print(k_points)

            continue

        model.fit(k_points)

        all_distances = model.get_distances(points)

        inliers = all_distances <= model.max_dist

        n_inliers = np.sum(inliers)

        # TODO added hack here to set larger or equa;l to
        if n_inliers >= n_best_inliers:
            n_best_inliers = n_inliers
            best_inliers = inliers

            if n_best_inliers > n_inliers_to_stop:
                break

    if return_model:
        model.least_squares_fit(points[best_inliers])
        return best_inliers, model

    else:
        return best_inliers







#%% MAIN funciton
def convert_calib_to_xyz(which_device,top_folder):

    # should really make the name a variable!
    if os.path.exists(top_folder+'/central_point_'+str(which_device)+'.csv') and args.force:
        print('already done, skipping (pass --force to recalculate)')
        return

    # load the values for filtering
    hsv_values,gray_values = read_hsv_file(which_device,top_folder)
    # and load the camera parameters
    cam_params = read_cam_params(which_device,top_folder)


    ###################
    # Block for reading from npy saving
    ###################
    d_list = get_file_shortlist(which_device,top_folder+'/npy_raw','d')
    cad_list = get_file_shortlist(which_device,top_folder+'/npy_raw','cad')

    n_frames = np.min((len(d_list),len(cad_list)))

    print(str(n_frames)+' frames to process from dev'+str(which_device)+'.')





    # initialize the background subtration
    bgSubThreshold = 100
    fgbg = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=bgSubThreshold,detectShadows=False)

    # loop over the frams
    with open(top_folder+'/central_point_'+str(which_device)+'.csv', 'w') as csvfile:
        pointwriter = csv.writer(csvfile, delimiter=',')
        for frame_counter in (range(n_frames)):

            if frame_counter%100 ==0:
                print('device ' + str(which_device) +' done '+str(frame_counter)+' of '+str(n_frames) +'...')
            elif frame_counter == (n_frames-1):
                print('device ' + str(which_device) +' doing last frame of '+str(n_frames) +'!')



            #for frame_counter in tqdm(range(n_frames)):
            # set default point to zeros
            # format of the point is frame,x,y,z,r,r_ball,N,err
            point = [frame_counter,0,0,0,0,0,0,0]

            # load the two frames
            ###########
            # numpy reading
            ###########
            d = np.load(top_folder+'/npy_raw/'+d_list[frame_counter])
            cad = np.load(top_folder+'/npy_raw/'+cad_list[frame_counter])

            rval = (cad is not None) and (d is not None)
#            d = cv2.imread(top_folder+'/'+d_list[frame_counter],2)
#            cad = cv2.imread(top_folder+'/'+cad_list[frame_counter])

            # convert to BGR and step the background to get the foreground mask
            # cad = cv2.cvtColor(cad,cv2.COLOR_RGB2BGR)
            # this is faster
            cad = np.flip(cad,2).copy()

            fgmask = fgbg.apply(cad)

            # now, we run the filtering step to get the mask
            mask = mask_stepper(cad,hsv_values,gray_values,fgmask)

            # only do this if there is a masked region to look at:
            # use 10 pixels as a cutoff for the smallest
            min_number_of_pixels = 10
            if (mask.any() and mask.sum() >= 255*min_number_of_pixels):
                # print('frame #'+str(frame_counter))

                # use the above function to get the largest region - speed up with contours?
                mask = fill_largest_region(mask)

                # and the indices od the mask, these are in pixels indices i,j
                pi,pj = np.where(mask)
                # get the depth of the masked pixels as a raveled list
                dij = d[pi,pj]

                # remove all the instances where depth is zero
                pi,pj,dij = clean_by_pd(pi,pj,dij)

                # convert the filtered pixels to world coordinates
                x_m,y_m,z_m = pixel_2_world(pi,pj,dij,cam_params)

                # now estimate the center of the ball
                if args.estimate == 'closestpoint':
                    # calculate the range, i.e. distance to distance from the camera, pythagoras
                    r_m = world_2_range(x_m,y_m,z_m)

                    # now simply select the min index, the closest point
                    min_idx = r_m.argmin()

                    # add the ping pong radius to all points, bit dirty
                    #TODO clean up this at some point
                    x_m,y_m,z_m = add_ping_pong_radius(x_m,y_m,z_m,r_m)

                    # and this is the closeset point
                    # format is frma, x,y,z,r,N,err
                    point = [frame_counter,x_m[min_idx],y_m[min_idx],z_m[min_idx],r_m[min_idx],len(x_m),0]

                elif args.estimate == 'fitsphere':
                    # import pyntcloud
                    from pyntcloud import PyntCloud
                    # convert the filtered points to a pyntcloud
                    # first get the postions
                    positions = np.transpose(np.vstack((x_m,y_m,z_m)))

                    # positions = positions[cloud.points.is_sphere==1,:]
                    from scipy.optimize import minimize
                    r = 0.02 # we know the radius!
                    def errorfun(center):
                        dist = np.linalg.norm(positions-center,axis = 1)
                        err = np.sum( (r - dist)**2 ) # minimize their distances!
                        return err

                    initial_guess = [np.mean(positions[:,0]),np.mean(positions[:,1]),np.mean(positions[:,2])]

                    result = minimize(errorfun, initial_guess)
                    # optional plotting
#                    sphere_plot_BRUTE(positions)

                    # sphere_plot_RANSAC(positions)


                    if result.success:
                        cc = result.x
                        n_pixels = positions.shape[0]
                        #TODO figure out where it is best to put the normalization
                        norm_err = result.fun/n_pixels
                        # format of the point is frame,x,y,z,r,N,err
                        point = np.hstack((frame_counter,cc,np.linalg.norm(cc),r,n_pixels,norm_err))

#                elif args.estimate == 'ransacsphere':
#                    # import pyntcloud
#                    from pyntcloud import PyntCloud
#                    # convert the filtered points to a pyntcloud
#                    # first get the postions
#                    positions = np.transpose(np.vstack((x_m,y_m,z_m)))
#                    # add to a dataframe
#                    points = pd.DataFrame(positions,columns=['x', 'y', 'z'])
#                    # and generate points
#                    cloud = PyntCloud(points)
#
#                    # now use the pyntcloud ransac to fit a sphere!
#                    from pyntcloud.ransac.fitters import single_fit
#
#                    inliers, best_model = single_fit(cloud.xyz, RansacSphereLoose, return_model=True)
#                    point = np.hstack((frame_counter,best_model.center,np.linalg.norm(best_model.center),best_model.radius,np.sum(inliers),np.sum(inliers)/len(inliers)))
#
#                    # sphere_plot_RANSAC(positions)

                elif args.estimate == 'ransacfixed':
                    # import pyntcloud, move outside?
                    #from pyntcloud import PyntCloud
                    # convert the filtered points to a pyntcloud
                    # first get the postions
                    positions = np.transpose(np.vstack((x_m,y_m,z_m)))
                    # add to a dataframe
                    #points = pd.DataFrame(positions,columns=['x', 'y', 'z'])
                    # and generate points
                    #cloud = PyntCloud(points)
                    # now use the pyntcloud ransac to fit a sphere!
                    #from pyntcloud.ransac.fitters import single_fit
                    inliers, best_model = single_fit(positions, RansacSphereFixedRadius, return_model=True)
                    point = np.hstack((frame_counter,best_model.center,np.linalg.norm(best_model.center),best_model.radius,np.sum(inliers),np.sum(inliers)/len(inliers)))

                    # potential plots
                    # sphere_plot_RANSACFIXED(positions)
                    # sphere_plot_RANSACFIXED(positions[inliers,:])

            pointwriter.writerow(point)

#%% READ
def read_device_0():
    print('processing camera 1...')
    which_device = 0
    top_folder = top_folder_0
    convert_calib_to_xyz(which_device,top_folder)

def read_device_1():
    print('processing camera 2...')
    which_device = 1
    top_folder = top_folder_0
    convert_calib_to_xyz(which_device,top_folder)

def read_device_2():
    print('processing camera 3...')
    which_device = 2
    top_folder = top_folder_1
    convert_calib_to_xyz(which_device,top_folder)

def read_device_3():
    print('processing camera 4...')
    which_device = 3
    top_folder = top_folder_1
    convert_calib_to_xyz(which_device,top_folder)


#%% Parallelize
#%% run the processes on independent cores
from multiprocessing import Process
if __name__ == '__main__':
    if args.ncams == 4:
        print('starting 4 cams, with multiprocessing!')
        # start 4 worker processes
        Process(target=read_device_0).start()
        Process(target=read_device_1).start()
        Process(target=read_device_2).start()
        Process(target=read_device_3).start()
        # oops, don't need this: Process(target=blink_using_firmata).start()

    elif args.ncams == 3:
        print('starting 3 cams, with multiprocessing!')
        Process(target=read_device_0).start()
        Process(target=read_device_1).start()
        Process(target=read_device_2).start()

    elif args.ncams == 2:
        print('starting 2 cams, with multiprocessing!')
        Process(target=read_device_0).start()
        Process(target=read_device_1).start()

    elif args.ncams == 1:
        print('starting 1 cam, with multiprocessing!')
        Process(target=read_device_0).start()



#%% please replace with a beter way!
#def getopts(argv):
#    opts = {}  # Empty dictionary to store key-value pairs.
#    while argv:  # While there are arguments left to parse...
#        if argv[0][0] == '-':  # Found a "-name value" pair.
#            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
#        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
#    return opts
#
#if __name__ == '__main__':
#    from sys import argv
#    myargs = getopts(argv)
#
#    if '-ncams' in myargs:  # Example usage.
#        print(myargs['-ncams'])
#
#    print(myargs)
#


##%% run the processes on independent cores
#if __name__ == '__main__':
#    if '-ncams' in myargs:  # Example usage.
#        if myargs['-ncams']=='1':
#            print('hello 1 cam!')
#            Process(target=read_device_0).start()
#
#        if myargs['-ncams']=='2':
#            print('hello 2 cams!')
#            Process(target=read_device_0).start()
#            Process(target=read_device_1).start()
#
#        if myargs['-ncams']=='3':
#            print('hello 3 cams!')
#            Process(target=read_device_0).start()
#            Process(target=read_device_1).start()
#            Process(target=read_device_2).start()
#
#        if myargs['-ncams']=='4':
#            print('hello 4 cams!')
#            Process(target=read_device_0).start()
#            Process(target=read_device_1).start()
#            Process(target=read_device_2).start()
#            Process(target=read_device_3).start()
#        else:
#            print('ncams not a valid choice')
#            serv.stop()
#    else:
#        print('no number given, just starting one camera')
#        print('not using the multiprocessing library')
#        read_device_0()
