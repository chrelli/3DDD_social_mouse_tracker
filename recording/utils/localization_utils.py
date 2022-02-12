#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:00:10 2018

@author: chrelli
"""
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
# import pyrealsense as pyrs

#import multiprocessing
from multiprocessing import Process

# for cloud handling
from pyntcloud import PyntCloud

# import handy Functions
#from utils.common_utils import *
#from utils.recording_utils import *
#from utils.cloud_utils import *


#%% Geometric functinos to find rotation angle and rotation matrix

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def rotation_angle(v1,v2):
    # first make sure that they are unit vectors!
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # do the clipping to avoid rounding errors at the edge, which makes arccos complain
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle

def rotation_matrix(axis, theta):
    return sp.linalg.expm(np.cross(np.eye(3), axis/sp.linalg.norm(axis)*theta))


#%% Defina function, which fits a plane to a pyntcloud and returns the coordinates of the plane!

def fit_plane_to_cloud(cloud):
    # import the ransac model class and the Plane geometry from pyntcloud
    from pyntcloud.ransac.models import RansacModel
    from pyntcloud.geometry.models.plane import Plane
    # define the ransac plane class myself
    class RansacPlane(RansacModel, Plane):

        def __init__(self, max_dist=1e-4):
            super().__init__(max_dist=max_dist)
            self.k = 10

        def are_valid(self, k_points):
            # any 3 points define a plane
            return True

    # now use the pyntcloud ransac to fit a sphere!
    from pyntcloud.ransac.fitters import single_fit
    inliers, best_model = single_fit(cloud.xyz, RansacPlane, return_model=True)

    this_normal = best_model.normal
    this_point = best_model.point

    return this_normal,this_point


def fit_plane_to_positions_pyntcloud(positions):
    # actually, it doesn't have to be a cloud, could just be positions
    # import the ransac model class and the Plane geometry from pyntcloud
    from pyntcloud.ransac.models import RansacModel
    from pyntcloud.geometry.models.plane import Plane
    # define the ransac plane class myself
    class RansacPlane(RansacModel, Plane):

        def __init__(self, max_dist= 1e-3):
            super().__init__(max_dist=max_dist)
            self.k = 20

        def are_valid(self, k_points):
            # any 3 points define a plane
            return True

    # now use the pyntcloud ransac to fit a sphere!
    from pyntcloud.ransac.fitters import single_fit
    inliers, best_model = single_fit(positions, RansacPlane, return_model=True)
#    inliers, best_model = single_fit(positions, RansacPlane,max_iterations=100,n_inliers_to_stop=None, return_model=True)
    this_normal = best_model.normal
    this_point = best_model.point

    return this_normal,this_point


from sklearn import linear_model
def fit_plane_to_positions_sklearn(positions):
    # actuallly, I don't even have to use pyntcloud, the sklearn ransac is prettyyyy good on it's own!
    # make the ransac regressor
    clt_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    # set the residual threshold
    clt_ransac.residual_threshold = 3e-3

    # and apply the fit to find a linear prediction of z = f(x,y), i.e. a plane
    clt_ransac.fit(positions[:,[0,1]], positions[:,2])
    # calculate the predicted plane, not really nescc, but w/e
    yhat_ransac = clt_ransac.predict(positions[:,[0,1]])
    positions_hat = np.vstack((positions[:,[0,1]].T,yhat_ransac)).T

    inlier_mask = clt_ransac.inlier_mask_

    # print('n inliers: ' +str(sum(inlier_mask)) +' of ' +str(len(inlier_mask)))

    # this is the beta value
    coefficients = clt_ransac.estimator_.coef_
    # this is the intercept with z ( f(0,0) )
    intercept_with_z = clt_ransac.estimator_.intercept_
    intercept = np.array([0,0,intercept_with_z])

    # so we have the followng vectors from the intercept along the plane:
    vec_xz = unit_vector( np.array([1,0,coefficients[0]]) )
    vec_yz = unit_vector( np.array([0,1,coefficients[1]]) )

    # so the normal to the vector is the cross product
    this_normal = np.cross(vec_xz,vec_yz,axis=0)
    # take the normal to be the average of the point cloud
    this_point = np.mean(positions_hat,axis=0)

    return this_normal,this_point

from sklearn import linear_model
def fit_plane_to_positions_sklearn_weighted(positions,weights):
    # actuallly, I don't even have to use pyntcloud, the sklearn ransac is prettyyyy good on it's own!
    # make the ransac regressor
    clt_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    # set the residual threshold
    clt_ransac.residual_threshold = 3e-3

    # and apply the fit to find a linear prediction of z = f(x,y), i.e. a plane
    clt_ransac.fit(positions[:,[0,1]], positions[:,2],sample_weight=weights)
    # calculate the predicted plane, not really nescc, but w/e
    yhat_ransac = clt_ransac.predict(positions[:,[0,1]])
    positions_hat = np.vstack((positions[:,[0,1]].T,yhat_ransac)).T

    inlier_mask = clt_ransac.inlier_mask_

    # print('n inliers: ' +str(sum(inlier_mask)) +' of ' +str(len(inlier_mask)))

    # this is the beta value
    coefficients = clt_ransac.estimator_.coef_
    # this is the intercept with z ( f(0,0) )
    intercept_with_z = clt_ransac.estimator_.intercept_
    intercept = np.array([0,0,intercept_with_z])

    # so we have the followng vectors from the intercept along the plane:
    vec_xz = unit_vector( np.array([1,0,coefficients[0]]) )
    vec_yz = unit_vector( np.array([0,1,coefficients[1]]) )

    # so the normal to the vector is the cross product
    this_normal = np.cross(vec_xz,vec_yz,axis=0)
    # take the normal to be the average of the point cloud
    this_point = np.mean(positions_hat,axis=0)

    return this_normal,this_point
