#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:18:58 2018

@author: chrelli
"""

# Demo getting the KRLS-t to work!

#%%



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
from utils.fitting_utils import *

#from merge_and_filter_clouds import filter_and_downsample_cloud

# h5py for acessing data
#import h5py

# ALLSO JIT STUFF

from numba import jit, njit


tracking_holder = np.load("utils/raw_tracking_no_bounds_full.npy")

# call the fitted values for X (is N body dimension x M time steps)


#%% Try to generate an estimate! Just xy for now!
        
        
xx = tracking_holder[-3,:]
yy = tracking_holder[-2,:]
zz = tracking_holder[-1,:]


#response variable is the next value!

plt.figure()
plt.plot(xx,yy)
plt.show()

plt.figure()
plt.plot(xx)

#%% Now, try generating the time embedded data!

#%% Generate training data by time embedding!

N_train = 2000
embedding = 5

def time_embedding(X,embedding):
    # X is a column vector!
    N = X.shape[0]
    X_embedded = np.zeros((N,embedding))
    for i in range(embedding):
        X_embedded[i:,i] = X[:(N-i)]
    return X_embedded

X = time_embedding(xx[:N_train],embedding)
Y = xx[1:(N_train+1)]

# add extra time dimension to the start for Xt
Xt = np.column_stack((np.arange(X.shape[0]),X))

#%% from matlab we have
#sigma_est,reg_est,lambda_est = 0.1631, 1.1680e-08,1.0000
#sigma_est,reg_est,lambda_est = 0.3775, 2.4780e-08,.9999

#sigma_est,reg_est,lambda_est = 14, 2.4780e-04,.999

#sigma_est =  0.2215
#reg_est = 4.449468e-09
#lambda_est =  1.0000

sigma_est =  0.1902
reg_est =  0.7567e-07
lambda_est = 0.9999

# Now make the kernel function!
from utils.gaussian import Gaussian
from utils.krlst import krlst

# make the kernel function with the appropriate sigma!

kern = Gaussian(sigma = sigma_est)
# make the regressor!
reg = krlst(kern)
reg.Lambda = lambda_est

#reg.Lambda = 0.99

reg.sn2 = reg_est


# % % Loop over the data and predict!

y_max = []
loops = np.linspace(100,len(Y)-100,num = 20)
for loop_from in loops:
    y_pred = [0]
#    loop_from = 200
    # at 400, we stop adding 'real' data, and just recursively add predicted data!
    for i,y in tqdm(enumerate(Y)):
        if i < loop_from:
            # train with real data!
            reg.train(X[i,:],y)
            X_train = X[i,:]
           
            if i>0:
                y_guess = float(reg.evaluate(X[i,:])[0])
                y_pred.append(y_guess)
                # get this ready for the prediction!
                # initialize X_train for the next!
                X_train = X[i+1,:]
                     
        else:
            # estimate the guess
            y_guess = float(reg.evaluate(X_train)[0])
            # add to list
            y_pred.append(y_guess)
            # and update X_train 
            # now, just do it recursively!
            #train here?
    #        reg.train(X_train,y_guess)
            if i == loop_from + 20:
                continue
            
            X_train = np.hstack((y_guess,X_train[:-1]))
    y_max.append(y_pred)
        

#% %
plt.close('all')
plt.figure()
plt.plot(Y)
for y_pred in y_max:
    plt.plot(y_pred)
for loop_from in loops:    
    plt.axvline(x=loop_from-1)
#plt.xlim([loop_from-100,loop_from+100])
plt.show()


#%% Super naiive linear regression
from sklearn import linear_model
regr = linear_model.LinearRegression()

y_pred = [0]
y_pred2 = [0,0]
y_pred3 = [0,0,0]

loop_from = 2000
# at 400, we stop adding 'real' data, and just recursively add predicted data!
for i,y in enumerate(Y):
    regr = linear_model.LinearRegression()
    regr.fit(np.arange(embedding).reshape(-1,1),X[i,:],0.9**np.arange(embedding))
    y_pred.append(regr.predict(np.array([-1]).reshape(-1,1)))
    y_pred2.append(regr.predict(np.array([-2]).reshape(-1,1)))
    y_pred3.append(regr.predict(np.array([-3]).reshape(-1,1)))
#% %
plt.close('all')
plt.figure()
plt.plot(Y)
plt.plot(y_pred)
plt.plot(y_pred2)
plt.plot(y_pred3)
plt.axvline(x=loop_from)
plt.show()
    
#%% Try just with KRLS

from utils.krlst import KRLS





#%%

def compute_RBF(mat1, mat2, sigma = 0.016):

    trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
    trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat2]).T

    k1 = trnorms1 * np.mat(np.ones((mat2.shape[0], 1), dtype=np.float64)).T

    k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T

    k = k1 + k2

    k -= 2 * np.mat(mat1 * mat2.T)

    k *= - 1./(2 * np.power(sigma, 2))

    return np.exp(k)


#%%
    x_c = np.reshape(x,(-1,1))
    x_m = np.matrix(x).T
    
#%%
    