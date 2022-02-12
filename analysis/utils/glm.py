# IDEA: Add neck to the posture map?
from IPython import get_ipython

# %matplotlib qt
# %load_ext autoreload
# %autoreload 2

import time, os, sys, shutil
from utils.fitting_utils import *

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#%matplotlib notebook
#%matplotlib widget
import scipy.stats as stats

from itertools import compress # for list selection with logical
from tqdm import tqdm

from multiprocessing import Process

# and pytorch
import torch

import sys, os, pickle
# import cv2
from colour import Color
import h5py
from tqdm import tqdm, tqdm_notebook
import glob
import itertools
import matplotlib.patches as patches


# test_selecta = []
# # TODO maybe there is a more efficient way to do this? Only have to do it once, so w/e..
# for i_fold in range(n_folds):
#     test_index = []
#     for i_skip in range(n_skips):
#         test_index.append(i_fold + i_skip * n_folds )
#     test_selecta.append(test_index)

# def fold_indices_compl(i_fold,test_selecta,time_index,):
#     # find the indices for the fold
#     test_logic = np.isin(fold_index,test_selecta[i_fold])
#     test_index = time_index[test_logic]
#     train_index = time_index[~test_logic]
#     return train_index,test_index

def fold_indices(i_fold,n_time,n_folds=10,n_skips=3):
    # everthing wrapped in one function
    n_chunks = n_folds*n_skips
    chunk_size = int(n_time/n_chunks)
    time_index = np.arange(n_time)
    fold_index = np.floor(np.linspace(0,n_folds*n_skips,n_time))    
    test_index = []
    for i_skip in range(n_skips):
        test_index.append(i_fold + i_skip * n_folds )
    test_logic = np.isin(fold_index,test_index)
    test_index = time_index[test_logic]
    train_index = time_index[~test_logic]    
    return train_index,test_index

def roughness_penalty_1d(param,p,beta = 5e1):
    # param_sub is the subset of the parameters that are associated with this predictor
    # just use the same beta as in the paper
    n_param = p['nsub']
    param_sub = param[p['param_subidx']]
    # we want a matrix with -1 on the diag and 1 abve the diag
    # the size is (n_param -1)x(n_param)
    # let's use eye, will be easy to pytochify later
    # first, we a square with -1!
    D1 = -np.eye(n_param,dtype = 'int')
    # now, we drop the bottom row of D
    D1 = D1[:-1,:]
    # then we add a slightly smaller eye to get the diagonal above
    D1[:,1:] += np.eye(n_param-1,dtype = 'int')
    # now, we can get DD1
    DD1 = np.matmul(D1.transpose(),D1)
    # and calc the value and gradient
    J = beta * .5 * np.inner( param_sub , np.matmul(DD1,param_sub) )
    J_g = beta * np.matmul(DD1,param_sub)
    J_h = beta * DD1
    return J, J_g,J_h

def roughness_penalty_1d_circ(param,p,beta = 5e1):
    # param_sub is the subset of the parameters that are associated with this predictor
    # just use the same beta as in the paper
    n_param = p['nsub']
    param_sub = param[p['param_subidx']]
    # we want a matrix with -1 on the diag and 1 abve the diag
    # the size is (n_param -1)x(n_param)
    # let's use eye, will be easy to pytochify later
    # first, we a square with -1!
    D1 = -np.eye(n_param,dtype = 'int')
    # now, we drop the bottom row of D
    D1 = D1[:-1,:]
    # then we add a slightly smaller eye to get the diagonal above
    D1[:,1:] += np.eye(n_param-1,dtype = 'int')
    # now, we can get DD1
    DD1 = np.matmul(D1.transpose(),D1)
    
    # NOW remember to make the boundary conditions circular
    DD1[0,:] = np.roll(DD1[1,:],-1)
    DD1[-1,:] = np.roll(DD1[-2,:],1)
    
    # and calc the value and gradient
    J = beta * .5 * np.inner( param_sub , np.matmul(DD1,param_sub) )
    J_g = beta * np.matmul(DD1,param_sub)
    J_h = beta * DD1
    return J, J_g,J_h




def neg_ll(param,A,binned_spikes,model_stack):
    X = A
    Y = binned_spikes

    # calculate the firing rate
    u = np.matmul(X,param)
    rate = np.exp(u)

    # loop over the predictors and add the roughness penalty to the model stack
    for p in model_stack:
        if p['type'] == '1d':
            J, J_g,J_h = roughness_penalty_1d(param,p)
            p['J'] = J
            p['J_g'] = J_g        
            p['J_h'] = J_h        
        elif p['type'] == '1d_circ':
            J, J_g,J_h = roughness_penalty_1d_circ(param,p)
            p['J'] = J
            p['J_g'] = J_g        
            p['J_h'] = J_h   
    # calc the function value, the gradient and the hessian
    f = np.sum(rate-Y*u) + np.sum([p['J'] for p in model_stack])
#     f_g = np.matmul(X.transpose() , (rate - Y) ) + np.hstack([p['J_g'] for p in model_stack])  # matmul maybe not so pythonic
    
#     # approx the hessian
#     rX = X*rate[:,np.newaxis]
#     # use the * to expand the list to individual arguments for block_diag
#     J_h_list = [p['J_h'] for p in model_stack]
#     f_h = np.matmul(rX.transpose(),X) + sp.linalg.block_diag(*J_h_list)
    
    return f



def jac_ll(param,A,binned_spikes,model_stack):
    X = A
    Y = binned_spikes

    # calculate the firing rate
    u = np.matmul(X,param)
    rate = np.exp(u)

    # loop over the predictors and add the roughness penalty to the model stack
    for p in model_stack:
        if p['type'] == '1d':
            J, J_g,J_h = roughness_penalty_1d(param,p)
            p['J'] = J
            p['J_g'] = J_g        
            p['J_h'] = J_h        
        elif p['type'] == '1d_circ':
            J, J_g,J_h = roughness_penalty_1d_circ(param,p)
            p['J'] = J
            p['J_g'] = J_g        
            p['J_h'] = J_h   

    # calc the function value, the gradient and the hessian
#     f = np.sum(rate-Y*u) + np.sum([p['J'] for p in model_stack])
    f_g = np.matmul(X.transpose() , (rate - Y) ) + np.hstack([p['J_g'] for p in model_stack])  # matmul maybe not so pythonic
    
#     # approx the hessian
#     rX = X*rate[:,np.newaxis]
#     # use the * to expand the list to individual arguments for block_diag
#     J_h_list = [p['J_h'] for p in model_stack]
#     f_h = np.matmul(rX.transpose(),X) + sp.linalg.block_diag(*J_h_list)
    
    return f_g


def hess_ll(param,A,binned_spikes,model_stack):
    X = A
    Y = binned_spikes

    # calculate the firing rate
    u = np.matmul(X,param)
    rate = np.exp(u)

    # loop over the predictors and add the roughness penalty to the model stack
    for p in model_stack:
        if p['type'] == '1d':
            J, J_g,J_h = roughness_penalty_1d(param,p)
            p['J'] = J
            p['J_g'] = J_g        
            p['J_h'] = J_h        
        elif p['type'] == '1d_circ':
            J, J_g,J_h = roughness_penalty_1d_circ(param,p)
            p['J'] = J
            p['J_g'] = J_g        
            p['J_h'] = J_h   

    # calc the function value, the gradient and the hessian
#     f = np.sum(rate-Y*u) + np.sum([p['J'] for p in model_stack])
#     f_g = np.matmul(X.transpose() , (rate - Y) ) + np.hstack([p['J_g'] for p in model_stack])  # matmul maybe not so pythonic
    
    # approx the hessian
    rX = X*rate[:,np.newaxis]
    # use the * to expand the list to individual arguments for block_diag
    J_h_list = [p['J_h'] for p in model_stack]
    f_h = np.matmul(rX.transpose(),X) + sp.linalg.block_diag(*J_h_list)
    
    return f_h



# even without cahing etc the hessian helps a lot!
# res = minimize(fun = neg_ll,x0 = param_start,args = (A,binned_spikes,model_stack) ,method='BFGS')
# res = minimize(fun = neg_ll,jac=jac_ll,x0 = param_start,args = (A,binned_spikes,model_stack) ,method='BFGS')
# res = minimize(fun = neg_ll,jac=jac_ll,x0 = param_start,args = (A,binned_spikes,model_stack) ,method='Newton-CG')
# res = minimize(fun = neg_ll,jac=jac_ll,hess=hess_ll,x0 = param_start,args = (A,binned_spikes,model_stack) ,method='trust-ncg')

# plt.plot(res.x)




# calculate the fit scores
def calc_fit_scores(param_fit,A,binned_spikes,verbose=False):
    #calculates a lot of model metrics

    # compute the llh increase from "mean firing rate model"
    # rename to r and n
    r = np.exp(np.matmul(A,param_fit))
    n = binned_spikes
    mean_spikes = np.mean(binned_spikes)

    # calculate smooth firing rates
    kernel = easy_kernel(60)
    fr_smooth = np.convolve(binned_spikes,kernel,'same')
    fr_hat_smooth = np.convolve(r,kernel,'same')

    # calculate explained variance and mse
    sse = np.sum((fr_hat_smooth-fr_smooth)**2)
    sst = np.sum((fr_smooth-np.mean(fr_smooth))**2)
    varExplained = 1-(sse/sst)
    mse = sse/len(fr_smooth)

    # calculate correlation
    pearson_r, pearson_p = sp.stats.pearsonr(fr_smooth,fr_hat_smooth)

    # calculate the log_llh_test stuff
    log_llh_model = np.sum(r - n*np.log(r) + np.log(sp.special.factorial(n)) )/np.sum(n)
    log_llh_mean = np.sum(mean_spikes - n*np.log(mean_spikes) + np.log(sp.special.factorial(n)) )/np.sum(n)
    log_llh = np.log(2) * (-log_llh_model + log_llh_mean)

    # counts 
    n_spikes, n_samples = np.sum(n), len(n)
    
    if verbose:
        plt.figure()
        plt.plot(fr_smooth,'r',label='real spikes')
        plt.plot(fr_hat_smooth,'g',label='model spikes')
        plt.xlim([0,10000])
        plt.show()

    return varExplained, pearson_r, log_llh, mse, n_spikes, n_samples

from scipy.linalg import block_diag
from scipy.optimize import minimize
from .analysis_tools import easy_kernel

def fit_model_spec(model_spec,predictor_stack,binned_spikes,n_folds =10):
    # takes a predictor stack and the binned spikes as data
    # the model spec is a list of the indices of predictors to include
    # in the model, e.g. [0, 2]
    # returns the model stack [maybe get rid of this?]

    # pack a model stack with the selected predictors
    model_stack = [predictor_stack[i] for i in model_spec]
    # also add the parameter subindices for the roughness later!
    n_predictors = len(model_stack)
    running_count = 0
    for p in model_stack:
        p['param_subidx'] = running_count + np.arange(p['nsub'])
        running_count += p['nsub']

    # We can make A once and for all
    A = np.concatenate([p['A'] for p in model_stack],axis = 1)
    # initialize the parameters! 
    n_time,n_params = A.shape
    param_start = np.random.normal(size=n_params)*1e-3 # some random non-zero numbers
    param = param_start

    
    # fit the model for every fold
    param_holder = []
    for i_fold in range(n_folds):
#         print("fitting fold #{}...".format(i_fold))
        train_index,test_index = fold_indices(i_fold,n_time,n_folds=10,n_skips=3)

        # fit the data on training data,
        res = minimize(fun = neg_ll,jac=jac_ll,hess=hess_ll,x0 = param_start,args = (A[train_index,:],binned_spikes[train_index],model_stack) ,method='trust-ncg',tol = 1e-3)
        param_fit = res.x
        param_holder.append(param_fit)
        
        # for the subsequent fits, just start at the last best fit
        param_start = param_start

    # pack the result in the param_holder
    model_scores = {"param_holder": param_holder,
                    "train":{'varExplained':[],'pearson_r':[],'log_llh':[],'mse':[],'n_spikes':[],'n_samples':[]},
                    "test":{'varExplained':[],'pearson_r':[],'log_llh':[],'mse':[],'n_spikes':[],'n_samples':[]} }


    # loop over the folds and calculate the scores
    for i_fold in range(n_folds):
        # get the index for the fold
        train_index,test_index = fold_indices(i_fold,n_time,n_folds=10,n_skips=3)
        # loop over test and train and get the scores, save to the dict
        for index,name in zip([train_index,test_index],["train","test"]):
            varExplained, pearson_r, log_llh, mse, n_spikes, n_samples = calc_fit_scores(param_holder[i_fold],A[index,:],binned_spikes[index])
            model_scores[name]['varExplained'].append(varExplained)
            model_scores[name]['pearson_r'].append(pearson_r)
            model_scores[name]['log_llh'].append(log_llh)
            model_scores[name]['mse'].append(mse)
            model_scores[name]['n_spikes'].append(n_spikes)
            model_scores[name]['n_samples'].append(n_samples)    
    
    # after the folds, we calc a few things:
    for name in ["train","test"]:
        log_llh = model_scores[name]['log_llh']
        model_scores[name]['mean_log_llh'] = np.mean(log_llh)
        # do a signed rank test, to see if model is better than the flat model
        w_baseline,pval_baseline = sp.stats.wilcoxon(log_llh,alternative='greater')
        model_scores[name]['w_baseline'] = w_baseline
        model_scores[name]['pval_baseline'] = pval_baseline

    return model_scores, model_stack
