
from utils.fitting_utils import *


import time, os, sys, shutil

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#%matplotlib notebook
#%matplotlib widget

from itertools import compress # for list selection with logical
from tqdm import tqdm

from multiprocessing import Process

# ALLSO JIT STUFF
from numba import jit, njit

# and pytorch
import torch



#%% function to do the clipping

def pytorch_clip(part,min,max):
    #will clip a 1 X N torch array like numpy
    # really nonsense that this is not a native pytorch function
    return torch.max(torch.min(part, max), min)

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
    # gamma_range = np.array([None,None])


    # now set the range for the spine scaling
    s_range = np.array([0,1]) #[a.u.]

    # theta is the head pitch (head up/down, around y' axis)
    # so theta = 0 is horizonal, theta<0 is looking up, theta>0 is looking down
    theta_range = np.pi * np.array([-1/2,1])

    # phi is the head yaw, i.e. from left to right
    # allow a bit more than 90 degress left/right
    phi_range = np.pi *0.67* np.array([-1,1])

    # so the boundaries are:
    bounds = np.vstack((beta_range,gamma_range,
                        s_range,
                        theta_range,phi_range,
                        x_range,y_range,z_range))
    hard_lo = bounds[:,0]
    hard_hi = bounds[:,1]

    return hard_lo,hard_hi

hard_lo,hard_hi = hard_limits()
hard_lo = torch.tensor(np.tile(hard_lo,2),dtype=torch.float32)
hard_hi = torch.tensor(np.tile(hard_hi,2),dtype=torch.float32)

def enforce_hard_limits(part,hard_lo,hard_hi):
    # clamps the part inside the
    return torch.max(torch.min(part, hard_hi), hard_lo)




#~~~~~~~~~~~~~~~~~~~~~~~~
# Some helper functions for calculating the loss
#~~~~~~~~~~~~~~~~~~~~~~~~

def make_inner_Q(aa,bb,n_particles):
    # helper functoin
    # makes the inner diagonal Q matrix for the symmetric ellipsoid
    aa = 1./aa**2
    bb = 1./bb**2
    #Q_inner = torch.diagonal(torch.stack([aa,bb,bb],dim=0),dim1=0,dim2=1)
    #Q_inner = torch.diag(torch.stack([aa,bb,bb],dim=1))
    # TODO this seems kind of idiotic - can it be done with einsum??
    # this didn't work
    #torch.einsum('a,ai->aii',[Q_inner,torch.stack([aa,bb,bb],dim=1)])
    # maybe this is an idea? https://github.com/pytorch/pytorch/issues/1791
    Q_inner = torch.zeros(n_particles, 3, 3)
    Q_inner[:,0,0] = aa
    Q_inner[:,[1,2],[1,2]] = bb.unsqueeze(1)

    return Q_inner

def make_yz_rotation(beta,gamma,one,zero):
    # helper function
    # makes a rotation matrix, only around y and z angles
    # first, calcul,ate
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # makes a rotation matrix, only around y and z angles
    rot_beta = torch.stack([torch.stack([cos_beta,   zero,      sin_beta], dim=1),
                      torch.stack([zero,       one,       zero],      dim=1),
                      torch.stack([-sin_beta,  zero,      cos_beta], dim=1)], dim=1)
    rot_gamma = torch.stack([torch.stack([cos_gamma,  -sin_gamma, zero],      dim=1),
                      torch.stack([sin_gamma,  cos_gamma,  zero],      dim=1),
                      torch.stack([zero,       zero,       one],       dim=1)], dim=1)

    # now, these are also n-particles x 3 x 3, or batchzie x 3 x 3 in tf lingo
    # do batch-wise matrix multiplication with einsum
    rot_xy = torch.einsum('aij,ajk->aik',[rot_beta,rot_gamma])
    return rot_xy

def make_xyz_rotation(alpha,beta,gamma,one,zero):
    # helper function
    # makes a rotation matrix, only around y and z angles
    # first, calcul,ate
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # makes a rotation matrix, only around y and z angles
    rot_alpha = torch.stack([torch.stack([one,   zero,      zero], dim=1),
                          torch.stack([zero,       cos_alpha,       -sin_alpha],      dim=1),
                          torch.stack([zero,  sin_alpha,      cos_alpha], dim=1)], dim=1)
    rot_beta = torch.stack([torch.stack([cos_beta,   zero,      sin_beta], dim=1),
                      torch.stack([zero,       one,       zero],      dim=1),
                      torch.stack([-sin_beta,  zero,      cos_beta], dim=1)], dim=1)
    rot_gamma = torch.stack([torch.stack([cos_gamma,  -sin_gamma, zero],      dim=1),
                      torch.stack([sin_gamma,  cos_gamma,  zero],      dim=1),
                      torch.stack([zero,       zero,       one],       dim=1)], dim=1)

    # now, these are also n-particles x 3 x 3, or batchzie x 3 x 3 in tf lingo
    # do batch-wise matrix multiplication with einsum
    rot_xy = torch.einsum('aij,ajk->aik',[rot_beta,rot_gamma])
    rot_xyz = torch.einsum('aij,ajk->aik',[rot_alpha,rot_xy])
    return rot_xyz




def batch_diagonal_old(input):
    # from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # waiting for this to close: https://github.com/pytorch/pytorch/issues/12160
    # strided not documented https://github.com/pytorch/pytorch/issues/9928
    # could batch a (n)-D to (n+1)-D diagonal matrix
    # works in  2D -> 3D
    # initialize
    # todo, even better would be diagonal https://pytorch.org/docs/stable/torch.html?highlight=diagonal#torch.diagonal
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    strides = [output.stride(i) for i in torch.arange(output.dim())]
    #output.as_strided(input.size(), [multiply(strides[:-1]) , output.size(-1) + 1]).copy_(input)
    output.as_strided(input.size(), [output.stride(0) , output.size(-1) + 1]).copy_(input)

    return output

def batch_diagonal(input):
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # pad the zero matrix
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal
    output.as_strided(input.size(), strides ).copy_(input)
    return output

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

def batch_diagonal(input):
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # pad the zero matrix
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal
    output.as_strided(input.size(), strides ).copy_(input)
    return output

def batch_diagonal_2D(input):
    # from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    output.as_strided(input.size(), [output.stride(0), output.size(input.dim()) + 1]).copy_(input)
    return output


def x0_to_mouse_body(part):
    '''
    Takes the particles and returns the location of centers and the width of the hip
    '''
    # hip centers
    # and nose centers
    # calculates the distance to one mouse, given positions and particles
    s = part[:,2]
    beta = part[:,0]
    gamma = part[:,1]
    theta = part[:,3]
    phi = part[:,4]
    t_body = part[:,5:8]

    # calculate vectors holding the hip values!
    # the values are n_particles long
    a_hip = a_hip_0 + a_hip_delta * s
    b_hip = b_hip_0 + b_hip_delta * (1.-s)
    d_hip = .75 * a_hip

    # we need to do cos an sin on the angles!
    # and other places too over and over, let's just keep them in mem?
    one  = torch.ones_like(s)
    zero = torch.zeros_like(s)


    R_body = make_yz_rotation(beta,gamma,one,zero)
    R_head = make_yz_rotation(theta,phi,one,zero)
    # also batch-wise here!
    R_nose = torch.einsum('aij,ajk->aik',[R_body,R_head])

    # THESE COORDINATES ARE WIHTIN THE REF FRAME OF THE MOUSE
    # c_nose is just d_nose, rotated by R_nose:
    # so for ever 2D matrix, it's c_nose =  R_nose @ np.array([d_nose,0,0])
    # and this only selects the first column of R_head and sums it
    c_mid = torch.stack([d_hip,zero,zero],dim=1)
    c_nose = c_mid + d_nose * R_head[:,:,0]

    # NOW HERE IS THE CONVERTION

    # Now, calculate the distance vectors from the origin of the hip, mid and head, in the real world
    c_hip = t_body
    c_nose = torch.einsum('aij,aj->ai',[R_body,c_nose]) + t_body
    # returns the coordinates and the current b, the shortest axis of the ellipsoid
    return c_hip,c_nose,b_hip


def smallest_distance_between_centers(part):
    '''
    A function which takes particles and returns an L2 loss on the amount of overlap of the balls
    '''
    c_hip_0,c_nose_0,b_hip_0 = x0_to_mouse_body(part)
    c_hip_1,c_nose_1,b_hip_1 = x0_to_mouse_body(part[:,8:])
    # first, we calculate the distances between the centers of the ellipsoids
    # nose2nose
    d_n0n1 = torch.norm(c_nose_0-c_nose_1,dim=1)
    # nose2hip
    d_n0h1 = torch.norm(c_nose_0-c_hip_1,dim=1)
    d_n1h0 = torch.norm(c_nose_1-c_hip_0,dim=1)
    # hip2hip
    d_h0h1 = torch.norm(c_hip_0-c_hip_1,dim=1)
    # make a list of the actual distance between the centers
    d_actual = torch.stack([d_n0n1,d_n0h1,d_n1h0,d_h0h1])
    # make a list of the minimum allowed distance between
    cutoff_barrier = torch.stack([(b_nose + b_nose)*torch.ones_like(b_hip_1),b_nose+b_hip_1,b_nose+b_hip_0,b_hip_0+b_hip_1])
    # clip the overlap
    ball_to_ball = torch.min(d_actual-cutoff_barrier)
    # do a kind of L2 loss, which we add everywhere
    return ball_to_ball

# part2 = torch.stack((part,part,part)).squeeze(1)
# part = part2
# ball_cost(part2)
