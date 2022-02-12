#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:32:20 2018

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

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *
from utils.cloud_utils import *



#%% arguments

import argparse

parser = argparse.ArgumentParser(description='uses LSE to generate a rigid transform to align the view of all cameras',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--noshow", help="disables showing the plots, will just dump the csv and png's",
                    action="store_false")

args = parser.parse_args()



#%% check if the folders exist

from utils.reading_utils import most_recent_calibration_folders

top_folder_0, top_folder_1 = most_recent_calibration_folders()

check_folder_if_present(top_folder_0)
check_folder_if_present(top_folder_1)

# debugging step, just for mac
if sys.platform == 'darwin':
    top_folder_0 = '/Volumes/Seagate Backup Plus Drive/example_data/calibration'
    top_folder_1 = '/Volumes/Seagate Backup Plus Drive/example_data/calibration'

    check_folder_if_present(top_folder_0)
    check_folder_if_present(top_folder_1)


#%% LOAD the saved estimation method!

estimation_method = np.genfromtxt(top_folder_0+'/trace_estimation.txt',dtype=str)

print('estimation method was: '+str(estimation_method))


#"""
## IDEA is the following:
#Use the RANSAC fitting of the fixed size sphere
#Use the count of in/outliers to estimate if the ball was a good fit! Kill bad points.
#Apply statistical outlier filter to remove obvious trash points.
#
## optional: choose the reference camera as the one which has the highest number of good frames
#
#Once only good points are left, we use RANSAC to fit the transformation matrix!
#
#First fit with squared error, THEN refinen with ransac?
#
#It doesn't matter if it takes time, as long as it happens...
#"""


#%% load the shifted time stamps!

def load_shifted_stamps(which_device,top_folder):
    # loads the frame numbers, time stamps, and unix time stamps, all are rescaled to start at zero!
    rawdata = np.genfromtxt(top_folder+'/shiftedstamps_'+str(which_device)+'.csv', delimiter=',')

    # this checks if the text file has the same row twice!
#    next_frame_new = np.hstack(( np.diff(rawdata[:,0]) > 0,True))

#    fn = rawdata[:,0]
#    ts = rawdata[:,1]
#    nix = rawdata[:,2]
#    leds = np.genfromtxt(top_folder+'/ledstamps_'+str(which_device)+'.csv', delimiter=',')
#
    #TODO fix this more elegantly, if frames are overwritten!
#    f_counter = rawdata[next_frame_new,0]
#    fn = rawdata[next_frame_new,0]
#    ts = rawdata[next_frame_new,1]
#    nix = rawdata[next_frame_new,2]
#    leds = rawdata[next_frame_new,3]
    f_counter = rawdata[:,0]
    fn = rawdata[:,1]
    ts = rawdata[:,2]
    nix = rawdata[:,3]
    leds = rawdata[:,4]

    return fn,ts,nix,leds

def load_raw_calib_trace(which_device,top_folder):
    raw_data = np.genfromtxt(top_folder+'/central_point_'+str(which_device)+'.csv', delimiter=',')
    frame,x,y,z,r = raw_data[:,0],raw_data[:,1],raw_data[:,2],raw_data[:,3],raw_data[:,4]

    # load stamps and trajectory for cloud i and reference j

#    # set the radius to nan??
#    r[np.where(r==0)]=np.nan

    fn,ts,nix,leds = load_shifted_stamps(which_device,top_folder)

    # add the ping pong radius!
    # x,y,z = add_ping_pong_radius(x,y,z,r)

    #also clean these by r!

    # now is the time to load extra stuff!
    if estimation_method == 'ransacfixed':
        #here the format was: point = np.hstack((frame_counter,best_model.center,np.linalg.norm(best_model.center),best_model.radius,np.sum(inliers),np.sum(inliers)/len(inliers)))
        # so frame, xyz, r_cam, r_ball, inliner_n, inlier_pct
        extras = raw_data[:,[5,6,7]]
        # looks like a good idea to cut the inlier pct to ca 20? Maybe 100 pixes as well?

    else:
        extras = raw_data[:,[5,6,7]]

    return frame,x,y,z,r,fn,ts,nix,leds,extras

#%% FUNCTIONS to align and plot point clouds:

# calculate the rigid transform, use knowledge about corresponding points
def estimate_rigid_transformation(A, B):
    # takes two arrays, which are N-by-3 since that is the pandas way
    # returns R an t as matrices, - do I want that or not??
    import numpy as np
    # make sure athat they are the same size
    assert A.shape == B.shape

    # convert to numpy matrices, since we do linag in this function
    A = np.asmatrix(A)
    B = np.asmatrix(B)

    N = A.shape[0]; # total points

    # get the centroids of the clouds
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the clouds
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # do matrix multiplication
    H = np.matmul(AA.transpose(),BB)
    U, S, Vt = np.linalg.svd(H)

    # recapitulate the rotation
    R = Vt.transpose() * U.transpose()

    # special reflection case, check determinant
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = R = Vt.transpose() * U.transpose()

    # recapitulate the translation
    t = np.matmul(-R,centroid_A.transpose()) + centroid_B.transpose()

    return R, t


def print_transformation_error(A,B,R,t):
    # convert to numpy matrices, since we do linag in this function
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    # total n of points
    n = A.shape[0]
    # calculate the rotated version of A!
    A2 = np.matmul(R,A.T) + np.tile(t, (1, n))
    A2 = A2.T

    # Find the error
    err = A2 - B

    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = np.sqrt(err/n)

    print("Rotation")
    print(R)
    print("Translation")
    print(t)

    print("RMSE:"+str( rmse))
    print("If RMSE is near zero, the function is correct!")


def plot_transformation_error(A,B,R,t):
    # convert to numpy matrices, since we do linag in this function
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    # total n of points
    n = A.shape[0]
    # calculate the rotated version of A!
    A2 = np.matmul(R,A.T) + np.tile(t, (1, n))
    A2 = A2.T

    # Find the error
    err = A2 - B

    single_point_norm = np.linalg.norm(err,axis=1)
    meanerr=np.mean(single_point_norm)

    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = np.sqrt(err/n)

    plt.figure()
    plt.plot(A,'.')
    plt.plot(B,'.')
    plt.title('corresponding points, after cleaning and 10x usampling')
    plt.legend(('x_i','y_i','z_i','x_j','y_j','z_j'),loc=4)
    plt.xlabel('# frame')
    plt.ylabel('xyz [m]')
    plt.show()

    plt.figure()
    plt.plot(B,'.')
    plt.plot(A2,'.')
    plt.title('After alignment, spoint_err: '+str(round(meanerr*1e3,2))+' mm, RMSE: '+str( rmse))
    plt.legend(('x_i','y_i','z_i','x_j','y_j','z_j'),loc=4)
    plt.xlabel('# frame')
    plt.ylabel('xyz [m]')
    plt.show()



#%% function to calculate the transformation
#which_device,top_folder,ref_device,ref_folder=3,top_folder_1,0,top_folder_0

def calculate_transformation(which_device,top_folder,ref_device,ref_folder):
    # load stamps and trajectory for cloud i and reference j
    #TODO this was done so fast and badly, glean it up later
    frame_i,x_i,y_i,z_i,r_i,fn_i,ts_i,nix_i,leds_i,extras_i = load_raw_calib_trace(which_device,top_folder)
    frame_j,x_j,y_j,z_j,r_j,fn_j,ts_j,nix_j,leds_j,extras_j = load_raw_calib_trace(ref_device,ref_folder)
#
#    # HERE WE ADD A CLEANING STEP!
#    def purge_dropped_frames(which_device,top_folder,fn_i):
#
#        # PURGE DROPPED FRAMES!
#        d_list = get_file_shortlist(which_device,top_folder,'d')
#        cad_list = get_file_shortlist(which_device,top_folder,'cad')
#
#
#        time_stamps
#


    # now is the time to load extra stuff!
    def cleaning(frame_i,x_i,y_i,z_i,r_i,fn_i,ts_i,nix_i,leds_i,extras_i):
        if estimation_method == 'ransacfixed':
            #here the format was: point = np.hstack((frame_counter,best_model.center,np.linalg.norm(best_model.center),best_model.radius,np.sum(inliers),np.sum(inliers)/len(inliers)))
            # so frame, xyz, r_cam, r_ball, inliner_n, inlier_pct
            r_ball = extras_i[:,0]
            inlier_n = extras_i[:,1]
            inlier_pct = extras_i[:,2]

#            good = (r_i>0.5)*(r_i < 1.5) * (inlier_n > 25) * (inlier_pct>0.2)
            good = (r_i>0.2)*(r_i < .65) * (inlier_n > 25) * (inlier_pct>0.2)


            # possible that the ir was saved, but not the led
            # print(len(good))
            # print(leds_j.shape)
            # print(extras_j.shape)


            if len(fn_i) > len(good):
                fn_i = fn_i[:len(good)]
            if len(nix_i) > len(good):
                nix_i = nix_i[:len(good)]
            if len(ts_i) > len(good):
                ts_i = ts_i[:len(good)]
            if len(leds_i) > len(good):
                leds_i = leds_i[:len(good)]
            # better:
            # for xx in [fn_i,ts_i,nix_i,leds_i]:
                # if len(xx) > len(good):
                    # xx = xx[:len(good)]
            if extras_i.shape[0] > len(good):
                extras_i = extras_i[:len(good),:]


            # if len(good) < len(frame_i):
            #     all_f = np.ones_like(good) * False
            #     all_f[:len(good)] = good
            #     good = all_f
            # elif len(good) > len(frame_i):
            #     all_f = np.ones_like(good) * False
            #     all_f = good[:len(all_f)]
            #     good = all_f


            # looks like a good idea to cut the inlier pct to ca 20? Maybe 100 pixes as well?

            return frame_i[good],x_i[good],y_i[good],z_i[good],r_i[good],fn_i[good],ts_i[good],nix_i[good],leds_i[good],extras_i[good,:]


    frame_i,x_i,y_i,z_i,r_i,fn_i,ts_i,nix_i,leds_i,extras_i = cleaning(frame_i,x_i,y_i,z_i,r_i,fn_i,ts_i,nix_i,leds_i,extras_i)
    frame_j,x_j,y_j,z_j,r_j,fn_j,ts_j,nix_j,leds_j,extras_j = cleaning(frame_j,x_j,y_j,z_j,r_j,fn_j,ts_j,nix_j,leds_j,extras_j)


    # filter trajectories to remove statistical outliers from both
    # STILL TO DO!
    ShowPlots = args.noshow
    if ShowPlots:

        plt.plot(ts_i,x_i)
        plt.plot(ts_j,x_j)
        plt.title('X')
        plt.show()

        plt.plot(ts_i,y_i)
        plt.plot(ts_j,y_j)
        plt.title('Y')
        plt.show()

        plt.plot(ts_i,z_i)
        plt.plot(ts_j,z_j)
        plt.title('Z')
        plt.show()

        plt.plot(ts_i,r_i)
        plt.plot(ts_j,r_j)
        plt.title('R')
        plt.show()



    # use scipy to interpolate with spline to generate an estimated trajectory at
    # generate interpolation functions from sp
    from scipy.interpolate import interp1d # due to some scipy weirdness

    what_kind_of_interpolation = 'linear'
    fx_i = interp1d(ts_i,x_i,kind=what_kind_of_interpolation)
    fx_j = interp1d(ts_j,x_j,kind=what_kind_of_interpolation)
    fy_i = interp1d(ts_i,y_i,kind=what_kind_of_interpolation)
    fy_j = interp1d(ts_j,y_j,kind=what_kind_of_interpolation)
    fz_i = interp1d(ts_i,z_i,kind=what_kind_of_interpolation)
    fz_j = interp1d(ts_j,z_j,kind=what_kind_of_interpolation)
    fr_i = interp1d(ts_i,r_i,kind=what_kind_of_interpolation)
    fr_j = interp1d(ts_j,r_j,kind=what_kind_of_interpolation)


    # time step now:
    dt = np.median(np.diff(ts_j))
    # upsample 10 times!
    newdt = .1*dt
    maxtime = np.min([np.max(ts_i),np.max(ts_j)])
    mintime = np.max([np.min(ts_i),np.min(ts_j)])
    # generate a new time for both
    xnew = np.arange(mintime,maxtime,newdt)

    # now generate traces on common time!

    x_in = fx_i(xnew)
    x_jn = fx_j(xnew)
    y_in = fy_i(xnew)
    y_jn = fy_j(xnew)
    z_in = fz_i(xnew)
    z_jn = fz_j(xnew)

    r_in = fr_i(xnew)
    r_jn = fr_j(xnew)

    if ShowPlots:

        plt.plot(xnew,x_in)
        plt.plot(xnew,x_jn)
        plt.title('X interpolated')
        plt.show()

        plt.plot(xnew,y_in)
        plt.plot(xnew,y_jn)
        plt.title('Y interpolated')
        plt.show()

        plt.plot(xnew,z_in)
        plt.plot(xnew,z_jn)
        plt.title('Z interpolated')
        plt.show()

        plt.plot(xnew,r_in)
        plt.plot(xnew,r_jn)
        plt.title('R interpolated')
        plt.show()

    # clean by r should happen at the end, not before!!
    # here, it would be good to add a statistical outlier removal or something!
    def calculate_good_logic(r):
        # remove the first and last 20% of the trace
        pctage = 0.20
        r[0:int(np.round(pctage*len(r)))]=0
        r[-int(np.round(pctage*len(r))):-1]=0
        # and take only traces where r is between 50 and 150 cm from view
        range_logic = (r>0.2)*(r < .65)

        # run a box filter across the inverse to open in up around bad frames!
        n_box = 30 # n frames box filter
        opened_by_box = np.convolve(~range_logic,np.ones(n_box)/n_box,mode = 'same')

        # the good frames are the ones where the inverse of the opened range logic remains zero
        good_logic = opened_by_box == 0

        return good_logic

    good_logic_i = calculate_good_logic(r_in)
    good_logic_j = calculate_good_logic(r_jn)
    # bitwise and
    good_logic = good_logic_i & good_logic_j

    if ShowPlots:

        plt.plot(good_logic_i)
        plt.plot(good_logic_j)
        plt.plot(good_logic)
        plt.legend(('dev i','dev j','both'))
        plt.show()

    # now use rigid transform to calculate the rotation tensor!
    def xyz_2_points(x,y,z,good_logic):
        points = np.vstack((x,y,z))
        points = points[:,good_logic]
        return points.transpose()

    points_i = xyz_2_points(x_in,y_in,z_in,good_logic)
    points_j = xyz_2_points(x_jn,y_jn,z_jn,good_logic)

    if ShowPlots:
        plt.plot(points_i,'.')
        plt.plot(points_j,'.')
        plt.title('corresponding points, after cleaning and 10x usampling')
        plt.legend(('x_i','y_i','z_i','x_j','y_j','z_j'),loc=4)
        plt.xlabel('# frame')
        plt.ylabel('xyz [m]')
        plt.show()
        # estimate the jitter??

    # USE svd to estimate first
    R,t = estimate_rigid_transformation(points_i,points_j)

    # the refine using scipy: http://scipy-cookbook.readthedocs.io/items/robust_regression.html
    # start guess is the non-robust version! MAke an array
    x0 = np.array(np.hstack((R,t)))
    x0_list = np.reshape(x0,(x0.size,))

    def target_fun(x0_list,points_i,points_j):
        # target function, which just resturns the residual
        x0 = np.reshape(x0_list,(3,4))
        R = x0[:,:3]
        t = x0[:,3]

        differences = points_j-apply_rigid_transformation(points_i,R,t)
        return np.linalg.norm(differences,axis=1)

    from scipy.optimize import least_squares
    inlier_cutoff = 0.002
    res_lsq = least_squares(target_fun, x0_list, args=(points_i,points_j), loss='soft_l1', f_scale=inlier_cutoff)

    x_found = np.reshape(res_lsq.x  ,(3,4))

    R_scipy = np.matrix(x_found[:,0:3])
    t_scipy = np.matrix(x_found[:,3]).T

    if ShowPlots:
        #plt.figure()
        plot_transformation_error(points_i,points_j,R,t)
        plt.ylabel('SVD ESTIMATE')
        plt.show()

        #plt.figure()
        plot_transformation_error(points_i,points_j,R_scipy,t_scipy)
        plt.ylabel('ROBUST REGRESSION ESTIMATE, f_scale:'+str(inlier_cutoff))
        plt.show()

        plt.figure()
        plt.plot(res_lsq.fun)
        plt.xlabel('iteration')
        plt.ylabel('distance in cm')
        plt.title('convergence history')
        plt.axhline(res_lsq.fun[0])
        plt.legend(['trajectory','svd estimate'])
        plt.show()

    # overwrite the old R and t!
    R = R_scipy
    t = t_scipy


    transformation =  np.vstack((np.hstack((R,t)),np.mat([0,0,0,1])))
    # top_folder = '/Users/chrelli/git/3d_sandbox/matlab'
    np.savetxt(top_folder+'/rotation_'+str(which_device)+'.csv',R,delimiter=',')
    np.savetxt(top_folder+'/translation_'+str(which_device)+'.csv',t,delimiter=',')
    np.savetxt(top_folder+'/transformation_'+str(which_device)+'.csv',transformation,delimiter=',')


#    print_transformation_error(points_i,points_j,R,t)
#    plot_transformation_error(points_i,points_j,R,t)



#%% run the function across the four cams
calculate_transformation(0,top_folder_0,0,top_folder_0)
calculate_transformation(1,top_folder_0,0,top_folder_0)
calculate_transformation(2,top_folder_1,0,top_folder_0)
calculate_transformation(3,top_folder_1,0,top_folder_0)


#%% for debugging
#
#A,B = points_i,points_j
#
##R_est,t_est = rigid_transform_3D(A, B)
#
#R_est,t_est = estimate_rigid_transformation(A, B)
#print_transformation_error(A,B,R_est,t_est)
#plot_transformation_error(A,B,R_est,t_est)
